package questions

import org.apache.spark.sql.SparkSession
//import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, SQLContext}
import org.apache.spark.sql.functions
import org.apache.spark.sql.expressions.Window
import scala.reflect.ClassTag
import org.apache.spark.sql.functions._
/** AirTrafficProcessor provides functionalites to
* process air traffic data
* Spark SQL tables available from start:
*   - 'carriers' airliner information
*   - 'airports' airport information
*
* After implementing the first method 'loadData'
* table 'airtraffic',which provides flight information,
* is also available. You can find the raw data from /resources folder.
* Useful links:
* http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.sql.Dataset
* http://spark.apache.org/docs/latest/sql-programming-guide.html
* https://github.com/databricks/spark-csv
*
* We are using data from http://stat-computing.org/dataexpo/2009/the-data.html
*
* @param spark reference to SparkSession
* @param filePath path to the air traffic data
* @param airportsPath path to the airport data
* @param carriersPath path to the carrier data
*/
class AirTrafficProcessor(spark: SparkSession,
    filePath:String, airportsPath: String, carriersPath: String) {

    import spark.implicits._

    /*
    * Table names for SQL
    * use airTraffic for table name in loadData
    * method.
    */
    val airTraffic = "airtraffic"
    val carriers = "carriers"
    val airports = "airports"

    // load the files and create tables
    // for spark sql
    //DO NOT EDIT
    val carriersTable = spark.read
        .option("header","true")
        .option("inferSchema", "true")
        .csv(carriersPath)
    carriersTable.createOrReplaceTempView(carriers)
    //DO NOT EDIT
    val airportsTable = spark.read
        .option("header","true")
        .option("inferSchema", "true")
        .csv(airportsPath)
    airportsTable.createOrReplaceTempView(airports)

    /** load the data and register it as a table
    * so that we can use it later for spark SQL.
    * File is in csv format, so we are using
    * Spark csv library.
    *
    * Use the variable airTraffic for table name.
    *
    * Example on how the csv library should be used can be found:
    * https://github.com/databricks/spark-csv
    *
    * Note:
    *   If you just load the data using 'inferSchema' -> 'true'
    *   some of the fields which should be Integers are casted to
    *   Strings. That happens, because NULL values are represented
    *   as 'NA' strings in this data.
    *
    *   E.g:
    *   2008,7,2,3,733,735,858,852,DL,1551,N957DL,85,77,42,6,-2,CAE,
    *   ATL,191,15,28,0,,0,NA,NA,NA,NA,NA
    *
    *   Therefore you should remove 'NA' strings and replace them with
    *   NULL. Option 'nullValue' in csv library is useful. However you
    *   don't need to take empty Strings into account.
    *   For instance TailNum field contains empty strings.
    *
    * Correct Schema:
    *   |-- Year: integer (nullable = true)
    *   |-- Month: integer (nullable = true)
    *   |-- DayofMonth: integer (nullable = true)
    *   |-- DayOfWeek: integer (nullable = true)
    *   |-- DepTime: integer (nullable = true)
    *   |-- CRSDepTime: integer (nullable = true)
    *   |-- ArrTime: integer (nullable = true)
    *   |-- CRSArrTime: integer (nullable = true)
    *   |-- UniqueCarrier: string (nullable = true)
    *   |-- FlightNum: integer (nullable = true)
    *   |-- TailNum: string (nullable = true)
    *   |-- ActualElapsedTime: integer (nullable = true)
    *   |-- CRSElapsedTime: integer (nullable = true)
    *   |-- AirTime: integer (nullable = true)
    *   |-- ArrDelay: integer (nullable = true)
    *   |-- DepDelay: integer (nullable = true)
    *   |-- Origin: string (nullable = true)
    *   |-- Dest: string (nullable = true)
    *   |-- Distance: integer (nullable = true)
    *   |-- TaxiIn: integer (nullable = true)
    *   |-- TaxiOut: integer (nullable = true)
    *   |-- Cancelled: integer (nullable = true)
    *   |-- CancellationCode: string (nullable = true)
    *   |-- Diverted: integer (nullable = true)
    *   |-- CarrierDelay: integer (nullable = true)
    *   |-- WeatherDelay: integer (nullable = true)
    *   |-- NASDelay: integer (nullable = true)
    *   |-- SecurityDelay: integer (nullable = true)
    *   |-- LateAircraftDelay: integer (nullable = true)
    *
    * @param path absolute path to the csv file.
    * @return created DataFrame with correct column types.
    */
    def loadDataAndRegister(path: String): DataFrame = {
        val dataFrame = spark.read
                          .format("com.databricks.spark.csv")
                          .option("header", "true")         
                          .option("inferSchema", "true")    
                          .option("treatEmptyValuesAsNulls", "false")
                          .option("nullValue", "NA")    
                          .load(path)
        dataFrame.registerTempTable(airTraffic)
        return dataFrame
    }

    //USE can use SPARK SQL or DataFrame transformations

    /** Gets the number of flights for each
    * airplane. 'TailNum' column is unique for each
    * airplane so it should be used. DataFrame should
    * also be sorted by count in descending order.
    *
    * Result looks like:
    *   +-------+-----+
    *   |TailNum|count|
    *   +-------+-----+
    *   | N635SW| 2305|
    *   | N11150| 1342|
    *   | N572UA| 1176|
    *   | N121UA|    8|
    *   +-------+-----+
    *
    * @param df Air traffic data
    * @return DataFrame containing number of flights per
    * TailNum. DataFrame is sorted by count. Column names
    * are TailNum and count
    */
    def flightCount(df: DataFrame): DataFrame = {
         df.registerTempTable("airTraffic")
         return spark.sql("SELECT TailNum, COUNT(*) as count FROM airTraffic GROUP BY TailNum ORDER BY count DESC")
    }


    /** Which flights were cancelled due to
    * security reasons the most?
    *
    * Example output:
    * +---------+----+
    * |FlightNum|Dest|
    * +---------+----+
    * |     4285| DHN|
    * |     4790| ATL|
    * |     3631| LEX|
    * |     3632| DFW|
    * +---------+----+
    *
    * @return Returns a DataFrame containing flights which were
    * cancelled due to security reasons (CancellationCode = 'D').
    * Columns FlightNum and Dest are included.
    */
    def cancelledDueToSecurity(df: DataFrame): DataFrame = {
        df.registerTempTable("flight")
        return spark.sql("SELECT FlightNum, Dest FROM flight WHERE Cancelled = 1 AND CancellationCode = 'D'")
    }

    /** What was the longest weather delay between January
    * and march (1.1-31.3)?
    *
    * Example output:
    * +----+
    * | _c0|
    * +----+
    * |1148|
    * +----+
    *
    * @return DataFrame containing the highest delay which
    * was due to weather.
    */
    def longestWeatherDelay(df: DataFrame): DataFrame = {
         df.registerTempTable("df")
         spark.sql("SELECT MAX(WeatherDelay) FROM df WHERE Month >= 1 AND Month <= 3")
    }

    /** Which airliners didn't fly.
    * Table 'carriers' has this information.
    *
    * Example output:
    * +--------------------+
    * |         Description|
    * +--------------------+
    * |      Air Comet S.A.|
    * |   Aerodynamics Inc.|
    * |  Combs Airways Inc.|
    * |   Evanston Aviation|
    * |Lufthansa Cargo A...|
    * +--------------------+
    *
    * @return airliner descriptions
    */
    def didNotFly(df: DataFrame): DataFrame = {

//     	SELECT 
// baseFeeds.siteURL, feedFinishes.timeTaken, feedFinishes.timeSubmitted
// FROM feedFinishes 
// JOIN baseFeeds 
// ON feedFinishes.GUID = baseFeeds.GUID 
// WHERE baseFeeds.siteURL LIKE '%www.example.com%' 
// GROUP BY
// baseFeeds.siteURL, feedFinishes.timeTaken, feedFinishes.timeSubmitted


            // df.registerTempTable("df")
            // spark.sql("SELECT  c.Description FROM carriers c, df d WHERE c.Code != d.UniqueCarrier GROUP BY c.Description")
         // val shouldHaveFlyed = spark.sql("SELECT c.Description FROM carriers c, df d WHERE c.Code = d.UniqueCarrier")
          //carriersTable.except(shouldHaveFlyed).selectExpr("Description")

                  df.registerTempTable("airlines")
        //return spark.sql("SELECT DISTINCT carrier.Description FROM carriers carrier, airlines airline WHERE carrier.Code != airline.UniqueCarrier")
        carriersTable.except(spark.sql("SELECT carrier.* FROM carriers carrier, airlines airline WHERE carrier.Code = airline.UniqueCarrier"))
        .selectExpr("Description")
    }

    /** Find the airliners which travel
    * from Vegas to JFK. Sort them in descending
    * order by number of flights and append the
    * DataFrame with the data from carriers.csv.
    * Spark SQL table 'carriers' contains this data.
    * Vegas iasa code: LAS
    * JFK iasa code: JFK
    *
    *   Output should look like:
    *
    *   +--------------------+----+
    *   |         Description| Num|
    *   +--------------------+----+
    *   |     JetBlue Airways|1824|
    *   |Delta Air Lines Inc.|1343|
    *   |US Airways Inc. (...| 948|
    *   |American Airlines...| 366|
    *   +--------------------+----+
    *
    * @return DataFrame containing Columns Description
    * (airliner name) and Num(number of flights) sorted
    * in descending order.
    */
    def flightsFromVegasToJFK(df: DataFrame): DataFrame = {
           df.registerTempTable("df")
          spark.sql("SELECT Description, COUNT(*) FROM df, carriers" +
                    " WHERE Origin = 'LAS' AND Dest = 'JFK' AND UniqueCarrier = Code GROUP BY Description")
    }

    /** How much time airplanes spend on moving from
    * gate to the runway and vise versa at the airport on average.
    * This method should return a DataFrame containing this
    * information per airport.
    * Columns 'TaxiIn' and 'TaxiOut' tells time spend on taxiing.
    * Order by time spend on taxiing in ascending order. 'TaxiIn'
    * means time spend on taxiing in departure('Origin') airport
    * and 'TaxiOut' spend on taxiing in arrival('Dest') airport.
    * Column name's should be 'airport' for ISA airport code and
    * 'taxi' for average time spend taxiing
    * We are only interested in flights that takeoff and land inside
    * USA
    *
    * DataFrame should look like:
    *   +-------+-----------------+
    *   |airport|             taxi|
    *   +-------+-----------------+
    *   |    BRW|5.084736087380412|
    *   |    OME|5.961294471783976|
    *   |    OTZ|6.866595496262391|
    *   |    DAL|6.983973195822733|
    *   |    HRL|7.019248180512919|
    *   |    SCC|7.054629009320311|
    *   +-------+-----------------+
    *
    * @return DataFrame containing time spend on taxiing per
    * airport ordered in ascending order.
    */
    def timeSpentTaxiing(df: DataFrame): DataFrame = {
               val taxiinOriginDF = df.select($"Origin".alias("airport"), $"TaxiIn".alias("taxi"))
        val taxiinDestDF = df.select($"Dest".alias("airport"), $"TaxiOut".alias("taxi"))
        taxiinOriginDF.unionAll(taxiinDestDF)
                    .groupBy($"airport").avg("taxi")
                    .select($"airport", $"avg(taxi)".alias("taxi"))
                    .orderBy(functions.asc("taxi"))
    }

    /** What is the median travel distance?
    * Field Distance contains this information.
    *
    * Example output:
    * +-----+
    * |  _c0|
    * +-----+
    * |581.0|
    * +-----+
    *
    * @return DataFrame containing the median value
    */
    def distanceMedian(df: DataFrame): DataFrame = {
        df.registerTempTable("df")
        spark.sql("SELECT Percentile(Distance, 0.5) FROM df")
    }

    /** What is the carrier delay, below which 95%
    * of the observations may be found?
    *
    * Example output:
    * +----+
    * | _c0|
    * +----+
    * |77.0|
    * +----+
    *
    * @return DataFrame containing the carrier delay
    */
    def score95(df: DataFrame): DataFrame = {
        df.registerTempTable("df")
        spark.sql("SELECT Percentile(CarrierDelay, 0.95) FROM df")
    }


    /** From which airport are flights cancelled the most?
    * What percentage of flights are cancelled from a specific
    * airport?
    * cancelledFlights combines flight data with
    * location data. Returns a DataFrame containing
    * columns 'airport' and 'city' from 'airports'
    * and the sum of number of cancelled flights divided
    * by the number of all flights (we get percentage)
    * from 'airtraffic' table. Name should be 'percentage'.
    * Lastly result should be ordered by 'percentage' and
    * secondly by 'airport' both in descending order.
    *
    *
    *
    *
    *   Output should look something like this:
    *   +--------------------+--------------------+--------------------+
    *   |             airport|                city|          percentage|
    *   +--------------------+--------------------+--------------------+
    *   |  Telluride Regional|           Telluride|   0.211340206185567|
    *   |  Waterloo Municipal|            Waterloo| 0.17027863777089783|
    *   |Houghton County M...|             Hancock| 0.12264150943396226|
    *   |                Adak|                Adak| 0.09803921568627451|
    *   |Aspen-Pitkin Co/S...|               Aspen| 0.09157716223855286|
    *   |      Sioux Gateway |          Sioux City| 0.09016393442622951|
    *   +--------------------+--------------------+--------------------+
    *
    * @return DataFrame containing columns airport, city and percentage
    */
    def cancelledFlights(df: DataFrame): DataFrame = {
        
    df.registerTempTable("airlines")

    val cancelled = spark.sql(" SELECT a.airport, COUNT(*) as count_cancelled FROM airlines airline, airports a WHERE airline.Origin = a.iata AND airline.Cancelled = 1 GROUP BY a.airport")

    val flights = spark.sql("SELECT  a.airport, a.city, COUNT(*) as total_flights FROM airlines airline, airports a WHERE airline.Origin = a.iata GROUP BY a.airport, a.city")

    cancelled.registerTempTable("cancelled")
    flights.registerTempTable("flights")

    return spark.sql(" SELECT cancelled.airport, flights.city, (cancelled.count_cancelled / flights.total_flights) as percentage FROM cancelled, flights WHERE cancelled.airport = flights.airport ORDER BY percentage DESC, cancelled.airport DESC")


    }

    /**
    * Calculates the linear least squares approximation for relationship
    * between DepDelay and WeatherDelay.
    *  - First filter out entries where DepDelay < 0
    *  - there are definitely multiple data points for a single
    *    DepDelay value so calculate the average WeatherDelay per
    *    DepDelay
    *  - Calculate the linear least squares (c+bx=y) where
    *    x equals DepDelay and y WeatherDelay. c is the constant term
    *    and b is the slope.
    *
    *  
    * @return tuple, which has the constant term first and the slope second
    */
    def leastSquares(df: DataFrame):(Double, Double) = {
    //         df.registerTempTable("df")
    // val f: Traversable[(Double, Double)] = df
    //   .filter($"DepDelay" >= 0 && $"WeatherDelay" >= 0).groupBy($"DepDelay")
    //   .agg(avg($"WeatherDelay").alias("avgWD"))
    //   .select("DepDelay", "avgWD")
    //   .as[(Double, Double)].collect()

    // val n = f.size

    // if (n < 2) (0, 0)
    // else {

    //   val (sumX, sumY, sumXY, sumXX) = (
    //     f.map { case (dd, _) => dd }.sum,
    //     f.map { case (_, wd) => wd }.sum,
    //     f.map { case (dd, wd) => wd * dd }.sum,
    //     f.map { case (dd, _) => dd * dd }.sum)

    //   val b = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX)
    //   val c = (sumY - b * sumX) / n
    //   (c, b)
    // }


        val filteredValues = df.filter($"DepDelay" >= 0).groupBy($"DepDelay").avg("WeatherDelay").select($"DepDelay", $"avg(WeatherDelay)".alias("WeatherDelay"))

        val depDelay_Mean = filteredValues.select(functions.mean($"DepDelay")).first().get(0)
        val weatherDelay_Mean = filteredValues.select(functions.mean($"WeatherDelay")).first().get(0)
        val xyMeans = filteredValues.withColumn( "X", filteredValues("DepDelay") - depDelay_Mean ).withColumn( "Y", filteredValues("WeatherDelay") - weatherDelay_Mean )
        val result = xyMeans.withColumn("numerator", xyMeans("Y") * xyMeans("Y")).withColumn( "denominator", xyMeans("X") * xyMeans("X") )

        val slope = result.select(functions.sum($"numerator") / functions.sum($"denominator")).first().get(0).asInstanceOf[Double]
        val constant = weatherDelay_Mean.asInstanceOf[Double] - (slope * depDelay_Mean.asInstanceOf[Double])
        val resultTuple = (constant, slope)
        return resultTuple

    }

    /**
    * Calculates the running average for DepDelay per day.
    * Average should be taken from both sides of the day we
    * are calculating the running average. You should take
    * 5 days from both sides of the day you are counting the running
    * average into consideration. For instance
    * if we want to calculate the average for the 10th of
    * January we have to take days: 5,6,7,8,9,10,11,12,13,14 and
    * 15 into account.
    *
    * Complete example
    * Let's assume that data looks like this
    *
    * +----+-----+---+--------+
    * |Year|Month|Day|DepDelay|
    * +----+-----+---+--------+
    * |2008|    3| 27|      12|
    * |2008|    3| 27|      -2|
    * |2008|    3| 28|       3|
    * |2008|    3| 29|      -5|
    * |2008|    3| 29|      12|
    * |2008|    3| 30|       5|
    * |2008|    3| 31|      47|
    * |2008|    4|  1|      45|
    * |2008|    4|  1|       2|
    * |2008|    4|  2|      -6|
    * |2008|    4|  3|       0|
    * |2008|    4|  3|       4|
    * |2008|    4|  3|      -2|
    * |2008|    4|  4|       2|
    * |2008|    4|  5|      27|
    * +----+-----+---+--------+
    *
    *
    * When running average is calculated
    * +----------+------------------+
    * |      date|    moving_average|
    * +----------+------------------+
    * |2008-03-27|13.222222222222221|
    * |2008-03-28|              11.3|
    * |2008-03-29| 8.846153846153847|
    * |2008-03-30| 8.357142857142858|
    * |2008-03-31|               9.6|
    * |2008-04-01|               9.6|
    * |2008-04-02|10.307692307692308|
    * |2008-04-03|10.916666666666666|
    * |2008-04-04|              12.4|
    * |2008-04-05|13.222222222222221|
    * +----------+------------------+
    *
    * For instance running_average for 
    * date 2008-04-03 => (-5+12+5+47+45+2-6+0+4-2+2+27)/12
    * =10.916666
    *
    *
    * @return DataFrame with schema 'date' (YYYY-MM-DD) (string type) and 'moving_average' (double type)
    * ordered by 'date'
    */
    def runningAverage(df: DataFrame): DataFrame = {
    //         val wind = Window.orderBy($"timestamp".asc).rangeBetween(-5 * 24 * 60 * 60, 5 * 24 * 60 * 60)
    // val date = concat_ws(".", lpad($"DayOfMonth", 2, "0"), lpad($"Month", 2, "0"), $"Year")

    // df.withColumn("timestamp", unix_timestamp(date, "dd.MM.yyyy")).groupBy("timestamp")
    //   .agg(sum($"DepDelay").alias("sd_delay"), count($"DepDelay").as("cd_delay"))
    //   .withColumn("moving_average", sum($"sd_delay").over(wind) / sum($"cd_delay").over(wind))
    //   .withColumn("date", from_unixtime($"timestamp", "yyyy-MM-dd"))
    //   .select("date", "moving_average")


        val w = Window.partitionBy("date").orderBy("date").rowsBetween(-5, 5)

        val result = df.groupBy($"Year", $"Month", $"DayofMonth")
            .agg(functions.sum("DepDelay"), functions.count("DepDelay"))
            .select( functions.to_date(functions.concat_ws("-", $"Year", $"Month", $"DayofMonth")).alias("date"),
                        $"sum(DepDelay)".alias("sumDepDelay"), $"count(DepDelay)".alias("countDepDelay") )
            .withColumn( "moving_average", functions.sum("sumDepDelay").over(w)/functions.sum("countDepDelay").over(w) )
            .select($"date", $"moving_average")
        return result
    }


}
/**
*
*  Change the student id
*/
object AirTrafficProcessor {
    val studentId = "XXXXXX"
}