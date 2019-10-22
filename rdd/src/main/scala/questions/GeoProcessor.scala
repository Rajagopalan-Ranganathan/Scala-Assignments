package questions

import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import scala.math._

import org.apache.spark.graphx._
import org.apache.spark.graphx.Edge
import org.apache.spark.graphx.Graph


/** GeoProcessor provides functionalites to 
* process country/city/location data.
* We are using data from http://download.geonames.org/export/dump/
* which is licensed under creative commons 3.0 http://creativecommons.org/licenses/by/3.0/
*
* @param spark reference to SparkSession 
* @param filePath path to file that should be modified
*/
class GeoProcessor(spark: SparkSession, filePath:String) {

    //read the file and create an RDD
    //DO NOT EDIT
    val file = spark.sparkContext.textFile(filePath)

    /** filterData removes unnecessary fields and splits the data so
    * that the RDD looks like RDD(Array("<name>","<countryCode>","<dem>"),...))
    * Fields to include:
    *   - name
    *   - countryCode
    *   - dem (digital elevation model)
    *
    * @return RDD containing filtered location data. There should be an Array for each location
    */
    def filterData(data: RDD[String]): RDD[Array[String]] = {
        /* hint: you can first split each line into an array.
        * Columns are separated by tab ('\t') character. 
        * Finally you should take the appropriate fields.
        * Function zipWithIndex might be useful.
        */
        val dataList = List(1, 8, 16)
        return data.map(lines => lines.split('\t').zipWithIndex.filter{ case (value, idx) => dataList.contains(idx)}.map(_._1))
    }


    /** filterElevation is used to filter to given countryCode
    * and return RDD containing only elevation(dem) information
    *
    * @param countryCode code e.g(AD)
    * @param data an RDD containing multiple Array[<name>, <countryCode>, <dem>]
    * @return RDD containing only elevation information
    */
    def filterElevation(countryCode: String,data: RDD[Array[String]]): RDD[Int] = {
        return data.filter(record => record(1).contains(countryCode)).map(value => value(2).toInt)
    }



    /** elevationAverage calculates the elevation(dem) average
    * to specific dataset.
    *
    * @param data: RDD containing only elevation information
    * @return The average elevation
    */
    def elevationAverage(data: RDD[Int]): Double = {
        return  data.sum/data.count
        
    }

    /** mostCommonWords calculates what is the most common 
    * word in place names and returns an RDD[(String,Int)]
    * You can assume that words are separated by a single space ' '. 
    *
    * @param data an RDD containing multiple Array[<name>, <countryCode>, <dem>]
    * @return RDD[(String,Int)] where string is the word and Int number of 
    * occurrences. RDD should be in descending order (sorted by number of occurrences).
    * e.g ("hotel", 234), ("airport", 120), ("new", 12)
    */
    def mostCommonWords(data: RDD[Array[String]]): RDD[(String,Int)] = {
        return data.flatMap(lines => lines(0).split(' ')).map( word => (word,1)).reduceByKey(_ + _ ).sortBy(_._2,false)
        
    }

    /** mostCommonCountry tells which country has the most
    * entries in geolocation data. The correct name for specific
    * countrycode can be found from countrycodes.csv.
    *
    * @param data filtered geoLocation data
    * @param path to countrycode.csv file
    * @return most common country as String e.g Finland or empty string "" if countrycodes.csv
    *         doesn't have that entry.
    */
    def mostCommonCountry(data: RDD[Array[String]], path: String): String = {
        val countryCodeRDD = data.map(lines => (lines(1),1)).reduceByKey(_ + _ ).sortBy(_._2,false).first._1
        val ccToCountryFile = spark.sparkContext.textFile(path)
        val ccRDD = ccToCountryFile.map(lines => lines.split(','))
          .filter(cc =>cc(1).equals(countryCodeRDD)).map(country => country(0))
        if (ccRDD.count > 0) {
            return ccRDD.first
        }
        return ""

    }

//
    /**
    * How many hotels are within 10 km (<=10000.0) from
    * given latitude and longitude?
    * https://en.wikipedia.org/wiki/Haversine_formula
    * earth radius is 6371e3 meters.
    *
    * Location is a hotel if the name contains the word 'hotel'.
    * Don't use feature code field!
    *
    * Important
    *   if you want to use helper functions, use variables as
    *   functions, e.g
    *   val distance = (a: Double) => {...}
    *
    * @param lat latitude as Double
    * @param long longitude as Double
    * @return number of hotels in area
    */
    def hotelsInArea(lat: Double, long: Double): Int = {
            val distanceBetweenXandY = (latX: Double, lonX: Double, latY: Double, lonY: Double) => {
            val LatitudeDelta=(latY - latX).toRadians
            val LongitutdeDelta=(lonY - lonX).toRadians

            //Return statement - gives error, so just return
            2 * asin(sqrt(pow(sin(LatitudeDelta/2),2) + pow(sin(LongitutdeDelta/2),2) * cos(latX.toRadians) * cos(latY.toRadians))) * 6371

        }

        val recordRDD = file.map(lines => lines.split('\t'))

        return recordRDD.filter( row => row(1).toLowerCase.contains("hotel")).map( row=> (row(1),distanceBetweenXandY(lat,long, row(4).toDouble, row(5).toDouble)))
        .filter(record => record._2 <= 10).map(_=> 1).count.toInt

    }

    //GraphX exercises

    /**
    * Load FourSquare social graph data, create a
    * graphx graph and return it.
    * Use user id as vertex id and vertex attribute.
    * Use number of unique connections between users as edge weight.
    * E.g
    * ---------------------
    * | user_id | dest_id |
    * ---------------------
    * |    1    |    2    |
    * |    1    |    2    |
    * |    2    |    1    |
    * |    1    |    3    |
    * |    2    |    3    |
    * ---------------------
    *         || ||
    *         || ||
    *         \   /
    *          \ /
    *           +
    *
    *         _ 3 _
    *         /' '\
    *        (1)  (1)
    *        /      \
    *       1--(2)--->2
    *        \       /
    *         \-(1)-/
    *
    * Hints:
    *  - Regex is extremely useful when parsing the data in this case.
    *  - http://spark.apache.org/docs/latest/graphx-programming-guide.html
    *
    * @param path to file. You can find the dataset
    *  from the resources folder
    * @return graphx graph
    *
    */
    def loadSocial(path: String): Graph[Int,Int] = {
        val regExPattern = """\s*(\d+)\s*\|\s*(\d+)\s*""".r

         val recordsRDD = spark.sparkContext.textFile(path).flatMap(regExPattern.unapplySeq(_)).map(value => value.map(value => value.toLong))
 
         val vertexRDD = recordsRDD.flatMap(vertex => vertex).distinct().map(value => (value, value.toInt))
 
         val edgeRDD = spark.sparkContext.makeRDD(recordsRDD.countByValue().map { case (edg, n) =>Edge[Int] (edg.head, edg(1), n.toInt) }.toSeq)
 
 
         Graph(vertexRDD, edgeRDD)
    }

    /**
    * Which user has the most outward connections.
    *
    * @param graph graphx graph containing the data
    * @return vertex_id as Int
    */
    def mostActiveUser(graph: Graph[Int,Int]): Int = {
        graph.outDegrees.map(_.swap).sortByKey(false).map(_.swap).first._1.toInt
    }

    /**
    * Which user has the highest pageRank.
    * https://en.wikipedia.org/wiki/PageRank
    *
    * @param graph graphx graph containing the data
    * @return user with highest pageRank
    */
    def pageRankHighest(graph: Graph[Int,Int]): Int = {
            graph.pageRank(0.000001).vertices.map(_.swap).sortByKey(false).map(_.swap).first._1.toInt
    }

}
/**
*
*  Change the student id
*/
object GeoProcessor {
    val studentId = "601153"
}