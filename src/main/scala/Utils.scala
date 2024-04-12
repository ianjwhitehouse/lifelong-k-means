import java.io.{FileInputStream, FileWriter}

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.Row
import SparkUtils._
import scala.annotation.tailrec
import scala.math._

object Utils {

  var (algorithm,alpha,numTargets) =
    try {
      val prop = new java.util.Properties()
      prop.load(new FileInputStream("/force/config.properties"))
      (
        prop.getProperty("algorithm"),
        prop.getProperty("clustering.alpha").toDouble,
        prop.getProperty("clustering.numTargets").toInt
      )
    } catch { case e: Exception =>
      e.printStackTrace()
      sys.exit(1)
    }
  //************************************************************************************************
  def selectIndexes(vec: Vector, sel: Array[Int]) : Array[Double] = {
    def selected: Array[Double] = sel.map(index => vec.apply(index))
    selected
  }
  //************************************************************************************************
  def selectIndexes(vec: Vector, sel: Array[Int], plant: Int, day: Int): (Array[Double], Double, Double) = {
    def selected: Array[Double] = sel.map(index => vec.apply(index))
    (selected,vec.apply(plant),vec.apply(day))
  }
  //************************************************************************************************
  def distance(xs: Array[Double], ys: Array[Double]) = {
    val res = sqrt((xs zip ys).map { case (x,y) => pow(y - x, 2) }.sum)
    //println(res)
    res
  }
  //************************************************************************************************
  def sumArray(x: Array[Double], y: Array[Double]): Array[Double] = {
    x.zip(y).map {
      case(x,y) => x+y
    }
  }
  //************************************************************************************************
  def generateCouplesFromList(list:List[(Array[Double], Double, Double)]): List[((Array[Double], Double, Double), (Array[Double], Double, Double))] ={
    @tailrec
    def generateCoupleTailRec(list:List[(Array[Double], Double, Double)], acc: List[((Array[Double],Double,Double),(Array[Double],Double,Double))]):List[((Array[Double],Double,Double),(Array[Double],Double,Double))]={
      list match {
        case head::Nil=> acc
        case head :: tail =>
          val couples=tail.map(x=>(head,x))
          generateCoupleTailRec(tail, acc ++ couples)
      }
    }
    generateCoupleTailRec(list, List())
  }
  //************************************************************************************************
  def writeFile(fileName: String, content: String): Unit = {
    val fw = new FileWriter(fileName, true)
    try {
      fw.write(content + "\r\n")
    }
    catch { case e: Exception =>
      e.printStackTrace()
      sys.exit(1)
    }
    fw.close()
  }
  //************************************************************************************************
  /**
    *
    * @param v1         Object vector contained in a cluster
    * @param v2         Object vector contained in a cluster
    * @return           Sum vector (v1+v2) obtained as sum for each feature
    */

  def sumVectors(v1:Vector,v2:Vector): Vector =  {
    val v =
      for(i <- 0 to v1.size-1)
        yield v1(i)+v2(i)
    Vectors.dense(v.toArray)
  }
  //************************************************************************************************
  def rowToVec(x: Row):Array[Double] = {
    /***
      *  Convert a row to an array excluding the initial value (id)
      */

    val size = x.length

    val y =
      for(i <- 1 to size-1)
        yield x.get(i).asInstanceOf[Double]

    y.toArray
  }
  //************************************************************************************************
  def cyclicSim (d1: Int, d2: Int): Double = {

    val (minDay,maxDay) =
      if (d1<d2) (d1,d2)                // 6 360
      else (d2,d1)

    val dist1: Double = Math.abs(minDay - maxDay)                 // 354
    val dist2: Double = Math.abs((-1 * maxDay) + minDay + 366)    // -359 + 6 + 366   13
    val mod1: Double = dist1 % 366                                // 12
    val mod2: Double = dist2 % 366                                // 13
    val min: Double = 0.0
    val max: Double = 183.0
    var n: Double = 0.0
    if (mod1 < mod2) {
      n = (mod1 - min) / (max - min)
      n = 1 - n
    }
    else {
      n = (mod2 - min) / (max - min)
      n = 1 - n
    }
    n
  }
  //************************************************************************************************
  def denormalizer(dataset: String, idplant: Double): Int = {

    val c = dataset match {
      case("PV_ITALY") => {
        val idplant_min = 1
        val idplant_max = 41
        val denorm =  ((idplant * (idplant_max - idplant_min)) + idplant_min).toInt
        //print("idplant: " + idplant + " denorm: " + denorm)
        denorm
      }
      case("PV_NREL") => {
        val idplant_min = 1
        val idplant_max = 48
        val denorm =  ((idplant * (idplant_max - idplant_min)) + idplant_min).toInt
        //print("idplant: " + idplant + " denorm: " + denorm)
        denorm
      }
      case("WIND_NREL") => {
        val idplant_min = 1
        val idplant_max = 5
        val denorm =  ((idplant * (idplant_max - idplant_min)) + idplant_min).toInt
        //print("idplant: " + idplant + " denorm: " + denorm)
        denorm
      }
    }
    c
  }
  //************************************************************************************************
  def calculateErrors(predicted: Array[Double], expected: Array[Double]): (Double, Double) = {

    // RMSE
    var sum = 0.0
    var sumExp = 0.0
    val temp: Double = Math.pow(10, 3)

    //println("Predicted " + predicted.mkString(",") + " - Expected " + expected.mkString(","))

    for(i <- 0 to predicted.length-1) {
      val diffsquare_norm = Math.pow(predicted.apply(i) - expected.apply(i), 2)
      sum = sum + diffsquare_norm

      var diffExp = Math.pow(((predicted.apply(i)-expected.apply(i))/(expected.apply(i))), 2)

      if(java.lang.Double.isNaN(diffExp)||java.lang.Double.isInfinite(diffExp))
        diffExp=0.0

      sumExp = sumExp + diffExp
    }

    var rmse = Math.sqrt(sum/expected.length)
    rmse = Math.round(rmse * temp) / temp

    // MAE (Mean Absolute Error)
    sum = 0.0

    for(i <- 0 to predicted.length-1) {
      var diff = 0.0

      diff = Math.abs(predicted.apply(i)-expected.apply(i))
      sum = sum + diff
    }

    var mae = sum / expected.length
    mae = Math.round(mae * temp) / temp

    //println("RMSE   " + rmse + " | MAE   " + mae)
    (rmse,mae)
  }
  //************************************************************************************************
/*
def cosineSimilarity(x: Array[Double], y: Array[Double]): Double = {
  require(x.size == y.size)
  dotProduct(x, y)/(magnitude(x) * magnitude(y))
}
 */
/*
 * Return the dot product of the 2 arrays
 * e.g. (a[0]*b[0])+(a[1]*a[2])
 */
/*
def dotProduct(x: Array[Double], y: Array[Double]): Double = {
  (for((a, b) <- x zip y) yield a * b) sum
}
*/

/*
 * Return the magnitude of an array
 * We multiply each element, sum it, then square root the result.
 */
/*
def magnitude(x: Array[Double]): Double = {
  math.sqrt(x map(i => i*i) sum)
}
*/

//************************************************************************************************
//************************************************************************************************
def customSimilarity(vecA: Vector, vecB: Vector): Double = {
  val vecAarray = vecA.toArray
  val vecBarray = vecB.toArray
/*
   println(vecAarray.foreach(println))
   println(vecBarray.foreach(println))
   println("alpha " + alpha)
   println("vecAarray size " + vecAarray.size)
   println("numTargets " + numTargets)
   println("* pezzo1 : " + ((1.0-alpha)/(vecAarray.size-numTargets)))
   println("* pezzo2 : " + sumNonTargets(vecAarray,vecBarray))
   println("* pezzo3: " + (alpha/numTargets))
   println("* pezzo4: " + (diffTargets(vecAarray,vecBarray)))
   println()*/

  if(numTargets==1)
    (((1.0-alpha)/(vecAarray.size-1)) * sumNonTarget(vecAarray,vecBarray)) + (alpha * (1- diffTarget(vecAarray,vecBarray)))
  else
    (((1.0-alpha)/(vecAarray.size-numTargets)) * sumNonTargets(vecAarray,vecBarray)) + ((alpha/numTargets) * (diffTargets(vecAarray,vecBarray)))
}
//************************************************************************************************
//************************************************************************************************
def sumAbsoluteDiff(vecA: Array[Double], vecB: Array[Double]): Double = {
  vecA.zip(vecB).map { case (a, b) => 1 - (Math.abs(a-b))}.sum
}
//************************************************************************************************
def sumNonTarget(vecA: Array[Double], vecB: Array[Double]): Double = {
  vecA.slice(0, vecA.size-1).zip(vecB.slice(0, vecB.size-1)).map { case (a, b) => 1 - (Math.abs(a-b))}.sum
}
//************************************************************************************************
def diffTarget(vecA: Array[Double], vecB: Array[Double]): Double = {
  val targetA = vecA.apply(vecA.size-1)
  val targetB = vecB.apply(vecB.size-1)
  Math.abs(targetA-targetB)
}
//************************************************************************************************
// Methods for multi-target
//************************************************************************************************
def sumNonTargets(vecA: Array[Double], vecB: Array[Double]): Double = {
  val vecAnonTargets = vecA.take(vecA.size-numTargets)
  val vecBnonTargets = vecB.take(vecB.size-numTargets)
  vecAnonTargets.zip(vecBnonTargets).map { case (a, b) => 1 - (Math.abs(a-b))}.sum
}
//************************************************************************************************
def diffTargets(vecA: Array[Double], vecB: Array[Double]): Double = {
  val vecAtargets = vecA.takeRight(numTargets)
  val vecBtargets = vecB.takeRight(numTargets)
  vecAtargets.zip(vecBtargets).map { case (a, b) => 1 - (Math.abs(a-b))}.sum
}
//************************************************************************************************


}
