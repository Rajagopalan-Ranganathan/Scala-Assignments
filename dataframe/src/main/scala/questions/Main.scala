package questions

import org.apache.spark.sql.SparkSession

import org.apache.log4j.Logger
import org.apache.log4j.Level

object Main {

  def main(args: Array[String]) = {

    Logger.getLogger("org").setLevel(Level.OFF)
      Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession.builder
      .master("local[*]")
      .appName("main")
      .config("spark.driver.memory", "5g")
      .config("spark.dynamicAllocation.enabled","true")
      .config("spark.shuffle.service.enabled","true")
      .getOrCreate()

      val filePath = getClass().getResource("/2008.csv").toString
      val airportsPath = getClass().getResource("/airports.csv").toString
      val carriersPath = getClass().getResource("/carriers.csv").toString

    val processor = new AirTrafficProcessor(spark, filePath, airportsPath, carriersPath)
    val data = processor.loadDataAndRegister(filePath)

    //println(data.schema)
    //data.collect().foreach(println)
    // println("<<<security>>>")
    // processor.cancelledDueToSecurity().show()
    // println("<<<weather dealy>>>")
    // processor.longestWeatherDelay().show()
    // println("<<<didn't fly>>>")
    //processor.didNotFly().show()
    // println("<<<from vegas to jfk>>>")
    // processor.flightsFromVegasToJFK().show()
    // println("<<<time taxiing>>>")
    // processor.timeSpentTaxiing().show()
    // println("<<<median>>>")
    // processor.distanceMedian().show()
    // println("<<<percentile>>>")
    // processor.score95().show()
    // println("<<<cancelled flights>>>")
    // processor.cancelledFlights().show()
    //println("least squares: " + processor.leastSquares())
    

    spark.stop()
  }

}