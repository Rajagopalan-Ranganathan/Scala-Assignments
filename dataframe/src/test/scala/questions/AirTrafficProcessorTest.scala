package questions


import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row

import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.scalatest._
import Matchers._
import org.scalatest.concurrent.TimeLimitedTests
import org.scalatest.concurrent.Interruptor
import org.scalatest.time.SpanSugar._



/**
* Tests for AirTrafficProcessor
*
*
*/
class AirTrafficProcessorTest extends FlatSpec with GivenWhenThen with AppendedClues with TimeLimitedTests {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    Logger.getLogger("hive").setLevel(Level.OFF)


    //if time limit is exceeded, terminate test
    override val defaultTestInterruptor = new Interruptor {
        override def apply(testThread: Thread): Unit = {
            testThread.stop()
        }
    }
    
    //time limit for each test
    def timeLimit = 100 seconds

    //initialize spark
    val spark = SparkSession.builder
      .master("local[*]")
      .appName("main")
      .config("spark.driver.memory", "5g")
      .config("spark.dynamicAllocation.enabled","true")
      .config("spark.shuffle.service.enabled","true")
      .getOrCreate()

    import spark.implicits._
    
    val flightPath = getClass().getResource("/sample.csv").toString().drop(5)
    val airportPath = getClass().getResource("/airports.csv").toString().drop(5)
    val carrierPath = getClass().getResource("/carriers.csv").toString().drop(5)
    //initialize AirTrafficProcessor
    val processor = new AirTrafficProcessor(spark, flightPath, airportPath, carrierPath)

    val df = spark.read.parquet(getClass().getResource("/sample.parquet/").toString().drop(5))

    info("----------------------\nusing sample.csv file\n----------------------")

    "loadDataAndRegister" should "load the csv file register it as "+
            "a table 'airtraffic' and finally return a DataFrame" in {
        val df2 = processor.loadDataAndRegister(flightPath)
        val dfC = df.collect().map(_.toSeq.toSet)

        spark.catalog.tableExists("airtraffic") should be (true) withClue("register table airtraffic")

        //df2.collect().foreach(x => dfC.contains(x) should equal (true) withClue("Can't find value: " + x + " from " + dfC))
    }
    //override sql table
    df.createOrReplaceTempView("airtraffic")

    "flightCount" should "calculate the total flighttime per airplane" in {
        val res: Array[(String,Int)] = Array(
            ("N961DL",2), ("N397DA",1), ("N3732J",1), ("N921DL",1),
            ("N973DL",1), ("N904DA",1), ("N968DL",1), ("N3741S",1), ("N908DA",1)).sortBy(x => (x._1,x._2))
        val df2 = processor.flightCount(df).collect().map(x => (x.getString(0),x.getLong(1))).sortBy(x => (x._1,x._2))

        res.deep should equal (df2.deep)
    }


    "cancelledDueToSecurity" should "produce a DataFrame having all of the "+
    "flights which were cancelled due to security" in {

        processor.cancelledDueToSecurity(df).collect().isEmpty should equal (true) withClue("Should be empty")
    }


    "longestWeatherDelay" should "calculate the maximux weather delay" in {

        processor.longestWeatherDelay(df).collect().deep should equal (Array(Row(null))) withClue("Should be empty(null row)")
    }

    "didNotFly" should "calculate which airliners didn't fly" in {
        processor.didNotFly(df).count() should equal (1490) withClue("Count should be: 1490")
    }


    "flightsFromVegasToJFK" should "tell which airliners travel from Vegas "+
    "to JFK and how often" in {
        processor.flightsFromVegasToJFK(df).collect().isEmpty should equal (true) withClue("should be empty")
    }

    "timeSpentTaxiing" should "tell by airport how long airplanes taxi "+
    "by average" in {
        val res = Array( ("CVG",6.166666666666667), ("TPA",9.25), ("SNA",14.5), ("SLC",14.5))
        processor.timeSpentTaxiing(df).collect().map(x => (x.getString(0),x.getDouble(1))).deep should equal (res.deep)
    }

    "distanceMedian" should "calculate median distance" in {
        processor.distanceMedian(df).collect().map(_.getDouble(0)) should equal (Array(719.0))
    }

    "score95" should "calculate CarrierDelay below which are 95% of the observations." in {
        processor.score95(df).collect().map(_.getDouble(0)) should equal (Array(0.0))
    }



    "cancelledFlights" should "tell how likely is it that flight will be "+
     "cancelled in a specific airport" in {
        processor.cancelledFlights(df).collect().isEmpty should equal (true) withClue("should be empty")
     }

     "leastSquares" should "estimate the WeatherDelay" in {
        processor.leastSquares(df) should equal ((0.0,0.0))
     }

     "runningAverage" should "calculate the running average for DepDelay" in {
        processor.runningAverage(df).collect().deep should equal (Array(Row("2008-05-10",3.7)).deep)
     }
}