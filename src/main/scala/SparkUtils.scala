import java.io.FileInputStream
import java.util.Properties

import org.apache.spark.sql.{SparkSession, DataFrame, SQLContext, SaveMode}
import org.apache.spark.{SparkConf, SparkContext}

object SparkUtils {

  val (memory, master, dbJar) =
    try {
      val prop = new Properties()
      prop.load(new FileInputStream("/force/config.properties"))
      (
        prop.getProperty("spark.memory"),
        prop.getProperty("spark.master"),
        prop.getProperty("db.jarPath")
      )
    } catch { case e: Exception =>
      e.printStackTrace()
      sys.exit(1)
    }

  val sparkConf: SparkSession = SparkSession.builder
    .master(master)
    .appName("FORCE")
    .config("spark.executor.memory ", memory)
    .config("spark.driver.maxResultSize", "10g")
    .config("spark.serializer","org.apache.spark.serializer.KryoSerializer")
    .getOrCreate()

  val sc : SparkContext = sparkConf.sparkContext
  sc.addFile(dbJar)

  val sqlC : SQLContext = sparkConf.sqlContext

      /*val sparkConf = new SparkConf()
        .setMaster(master)
        .setAppName("FORCE")
        .set("spark.executor.memory ", memory)
        .set("spark.storage.memoryFraction", "0.3")
        .set("spark.driver.maxResultSize", "10g")
        .set("spark.serializer","org.apache.spark.serializer.KryoSerializer")

      val sc: SparkContext = new SparkContext(sparkConf)
      sc.addFile(dbJar)

      val sqlContext = new SQLContext(sc)
*/

  def getSparkSession() : SparkSession = { sparkConf }
  def getSparkContext(): SparkContext = { sc }
  def getSQLContext(): SQLContext = { sparkConf.sqlContext }

  //************************************************************************************************
  def selectDataFrameFromTxt(fileName: String): DataFrame = {

    val df: DataFrame = sparkConf.sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true")       // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .option("delimiter",",")
      .load(fileName)

    df
  }

}