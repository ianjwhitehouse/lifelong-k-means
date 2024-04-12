
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.evaluation.MulticlassMetrics

import scala.collection.mutable.ListBuffer
import scala.math.{pow, sqrt}

object LifelongKMeans {

  val conf = new SparkConf()
    .setMaster("local[2]")
    .setAppName("CountingSheep")

  // import conf.implicits._

  val sc = new SparkContext(conf)

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  def main(args: Array[String]): Unit = {

    val trainPath = "data/sensorless_drive_diagnosis/streaming_train"
    val testPath = "data/sensorless_drive_diagnosis/streaming_test"
    val numClusters = 2
    val numIterations = 20
    val numfeatures = 49

    val trainData = sc.textFile(trainPath)
    val trainingData = trainData.map(s => Vectors.dense(s.split(',').map(_.toDouble).take(numfeatures))).cache()

    println(trainingData.count().toInt)

    // ****************************************************************************************
    // TO DO: Program key-value store: (cluster_id, (mean, stddev, max))
    // TO DO: Each new cluster created: Compute statistics and add them
    // TO DO: Caveat: data is not added if recognized
    // TO DO: Alternative: Find another way to recognize out-of-distribution: plain distance from centroid

    // TODO: Save files with predictions and cluster ids so that they
    // TODO: Try UCI multi-class datasets

    // TODO: Add complexity to the method?
    // ****************************************************************************************

    //val trainingData = sparkConf.sparkContext.textFile(trainPath).map(x => Vectors.dense(stringToVec(x)))
    //val testData = sparkConf.sparkContext.textFile(testPath).map(x => Vectors.dense(stringToVec(x)))

    var initial_model: KMeansModel = KMeans.train(trainingData, numClusters, numIterations)

    var model: KMeansModel = initial_model
    var clusterCenters: Array[linalg.Vector] = model.clusterCenters

    //var preds = model.predict(testingData)
    //val centersBC: Broadcast[Array[Vector]] = sparkConf.sparkContext.broadcast(clusterCenters)

    // *********
    // Old approach: calculates threshold based on variance of overall model calculated with training data points
    // Calculate average distance between each training instance and top k neighbors
//    val trainDistances: RDD[Double] = trainingData.map(trainInstance => {
//      val centroidID = model.predict(trainInstance)
//      val centroidVec = clusterCenters.apply(centroidID)
//      (euclDistance(trainInstance.toArray, centroidVec.toArray))
//    })

//    // Tentative new
//    val trainDistances: RDD[(Int, linalg.Vector, Double)] = trainingData.map(trainInstance => {
//      val centroidID = model.predict(trainInstance)
//      val centroidVec = clusterCenters.apply(centroidID)
//      (centroidID, trainInstance, euclDistance(trainInstance.toArray, centroidVec.toArray))
//    })
//
//    // For each centroid, set of instances that should create new clusters and calculate centroid for each group
//    val trainInstGrouped: RDD[(Int, Array[Double])] = trainDistances.map {
//      case (centroid, instance, distance) => (centroid, instance.toArray)
//    }.reduceByKey((x, y) => sumDivideVectors(x, y))
//
//    //

    // TODO: need to extract average and standard deviation of each cluster
    // TODO: need to save (cluster_id, avg, std_dev)
    // TODO: at testing time, add entries to this data structure or update existing values with incremental average
//
//    val mean: Double = trainDistances.sum / trainDistances.count()
//    val devs: RDD[Double] = trainDistances.map(v => (v - mean) * (v - mean))
//    val stddev: Double = Math.sqrt(devs.sum / (trainDistances.count() - 1))
//    val max: Double = mean + 2 * stddev
//
//    println("Training data stats:")
//    println("Mean " + mean)
//    println("Sddev " + stddev)
//    println("Max " + max)
//    println("_____________________")
    // *********

    var centroidDistances = new ListBuffer[Double]()
//    (numClusters*(numClusters-1))/2
    for (x <- 0 to clusterCenters.size-1) {
      for (y <- x+1 to clusterCenters.size-1) {
        centroidDistances += euclDistance(clusterCenters.apply(x).toArray, clusterCenters.apply(y).toArray)
        //println(dist)
      }
    }
    var minDistCentroids = centroidDistances.min
    print(minDistCentroids)
    minDistCentroids = math.sqrt(minDistCentroids)/10

    println("Initial threshold for cluster creation: " + minDistCentroids)

    // *************************
    // Testing - prediction mode

    var ground_truth_id: Double = 2.0   // Starts from cluster 3 - each batch presents a new cluster
//
//    val testing_batches = Array("Web_Shell_k_5_rate_0_iter_3.csv",
//              "Web_Shell_k_5_rate_0_iter_4.csv",
//              "Web_Shell_k_5_rate_0_iter_5.csv")


    val testing_batches = Array("3.csv",
              "4.csv",
              "5.csv",
              "6.csv",
              "7.csv",
              "8.csv",
              "9.csv",
              "10.csv",
              "11.csv")


    testing_batches.map {
      file =>
        println()
        println("Processing " + file + " ...")
        val testFile = testPath + "/" + file
        val testData: RDD[String] = sc.textFile(testFile)
        val testingData: RDD[linalg.Vector] = testData.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()
//        val testingData = processedTest.map(s => (s.apply(.take(numfeatures), s.takeRight(1)))
        val ground_truth = testData.map(s => Vectors.dense(s.split(',').map(_.toDouble).takeRight(1))).cache()

        val testDistancesWithData: RDD[(Int, linalg.Vector, Double)] = testingData.map(testInstance => {
          val centroidID: Int = model.predict(testInstance)
          val centroidVec: linalg.Vector = clusterCenters.apply(centroidID)
          (centroidID, testInstance, euclDistance(testInstance.toArray, centroidVec.toArray))
        })

        val outOfDistributionEntries: RDD[(Int, linalg.Vector, Double)] = testDistancesWithData.filter {
          case (centroid, instance, distance) => distance > minDistCentroids
        }

        println("Candidate objects for new clusters: " + outOfDistributionEntries.count() + " / " + testData.count())

        // For each centroid, set of instances that should create new clusters and calculate centroid for each group
        val groupedInstances: RDD[(Int, Array[Double])] = outOfDistributionEntries.map {
          case (centroid, instance, distance) => (centroid, instance.toArray)
        }.reduceByKey((x, y) => sumDivideVectors(x, y))
        //val newCentroids = groupedInstances.reduceByKey((x,y) => sumVectors(x,y))

        println("New clusters to be created: " + groupedInstances.count())

        // Model update phase : add centroid to clusterCenters

        val newCentroids: Array[linalg.Vector] = groupedInstances.map(a => org.apache.spark.mllib.linalg.Vectors.dense(a._2)).collect()
        //val allCentroids: Array[linalg.Vector] = clusterCenters ++ newCentroids
        //allCentroids.foreach(println)
        //model = new KMeansModel(allCentroids)
        clusterCenters = clusterCenters ++ newCentroids
        model = new KMeansModel(clusterCenters)

        // Prediction phase
        // TODO: instead of ground_truth_id, join predictions (pair rdd) with ground_truth (pair rdd)
        val preds_labels: RDD[(Double, Double)] = model.predict(testingData).map(x => (x.toDouble, ground_truth_id))
        val metrics = new MulticlassMetrics(preds_labels)
        println("Lifelong model:")
        model.predict(testingData).collect.foreach(x=> print(x + " "))
        println()
        println("Weighted Precision: " + metrics.weightedPrecision + " - Weighted Recall: " + metrics.weightedRecall + " - Weighted F1: " + metrics.weightedFMeasure)
        println()

        val preds_labels_bl: RDD[(Double, Double)] = initial_model.predict(testingData).map(x => (x.toDouble, ground_truth_id))
        val metrics_bl = new MulticlassMetrics(preds_labels_bl)
        println("Baseline:")
        initial_model.predict(testingData).collect.foreach(x=> print(x + " "))
        println()
        println("Weighted Precision: " + metrics_bl.weightedPrecision + " - Weighted Recall: " + metrics_bl.weightedRecall + " - Weighted F1: " + metrics_bl.weightedFMeasure)
        println()

        // Update distances which define new threshold for cluster creation

        centroidDistances = new ListBuffer[Double]()
        for (x <- 0 to clusterCenters.size-1) {
          for (y <- x+1 to clusterCenters.size-1) {
            centroidDistances += euclDistance(clusterCenters.apply(x).toArray, clusterCenters.apply(y).toArray)
          }
        }

        // Average (no clusters)
        // minDistCentroids = centroidDistances.sum / centroidDistances.length

        // Min (too many clusters - 2 each iteration)
        // minDistCentroids = centroidDistances.min

        // Second centroid (No clusters)
        // minDistCentroids = centroidDistances.sorted.apply(1)

        // In between first and second centroid (No clusters - clusters too far away?)
        // minDistCentroids = centroidDistances.sorted.apply(0) + centroidDistances.sorted.apply(1) / 2

        // Difference between second and first centroid (No clusters - clusters too far away?)
        minDistCentroids = centroidDistances.sorted.apply(1) - centroidDistances.sorted.apply(0)

        // First centroid + fraction
        minDistCentroids = centroidDistances.min + (centroidDistances.min / 2)

        println("New threshold for cluster creation: " + minDistCentroids)

        // TBD: Remove ground truth id counter increment - use ground truth in files

        ground_truth_id = ground_truth_id + 1
    }

    // ———————————————————


    /* val newCentroids = groupedInstances.aggregateByKey( (x,y) => {
          sumVectors(x,y)
        })*/

    //outlierEntries.collect.foreach(println)

    // Group instances by centroid and find new centroid for each group (average)
    // Add centroids to model
    // model = new KMeansModel(centers.map(_.vector))

    // Keep processing new data


  }


  /*    println(preds.count())
      println(preds.foreach(x => print(x + " ")))

      testFile = testPath + "/Web_Shell_k_5_rate_10_iter_4.csv"
      testData = sparkConf.sparkContext.textFile(testFile)
      testingData = testData.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()
      preds = model.predict(testingData)
      println(preds.count())
      println(preds.foreach(x => print(x + " ")))

      testFile = testPath + "/Web_Shell_k_5_rate_10_iter_5.csv"
      testData = sparkConf.sparkContext.textFile(testFile)
      testingData = testData.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()
      preds = model.predict(testingData)
      println(preds.count())
      println(preds.foreach(x => print(x + " ")))
      */

  //************************************************************************************************
  def stringToVec(x: String): Array[Double] = {
    /** *
     * Convert a string to an array
     */

    val x2 = x.split(",")
    val size = x2.length

    val y =
      for (i <- 0 to size - 1)
        yield x2.apply(i).toDouble

    y.toArray
  }

  //************************************************************************************************
  def euclDistance(xs: Array[Double], ys: Array[Double]) = {
    sqrt((xs zip ys).map { case (x, y) => pow(y - x, 2) }.sum)
  }

  //************************************************************************************************
  def sumVectors(vecA: Array[Double], vecB: Array[Double]): Array[Double] = {
    //vecA.slice(0, vecA.size-1).zip(vecB.slice(0, vecB.size-1)).map { case (a, b) => 1 - (Math.abs(a-b))}.sum
    vecA.zip(vecB).map { case (a, b) => a + b }
  }

  //************************************************************************************************
  def sumDivideVectors(vecA: Array[Double], vecB: Array[Double]): Array[Double] = {
    //vecA.slice(0, vecA.size-1).zip(vecB.slice(0, vecB.size-1)).map { case (a, b) => 1 - (Math.abs(a-b))}.sum
    vecA.zip(vecB).map { case (a, b) => (a + b) / 2 }
  }

}