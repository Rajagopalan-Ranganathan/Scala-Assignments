package questions

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.graphx.Edge
import org.apache.spark.graphx.Graph

import org.scalatest._
import Matchers._

import org.scalatest.concurrent.TimeLimitedTests
import org.scalatest.concurrent.Interruptor
import org.scalatest.time.SpanSugar._

/**
* GeoProcessorTest provides tests for GeoProcessor.
*
*
*/
class GeoProcessorTest extends FlatSpec with GivenWhenThen with AppendedClues with TimeLimitedTests {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    override val defaultTestInterruptor = new Interruptor {
        override def apply(testThread: Thread): Unit = {
            testThread.stop()
        }
    }
    
    //time limit for each test
    def timeLimit = 100 seconds

    val spark = SparkSession.builder
      .master("local")
      .appName("main")
      .config("spark.driver.memory", "5g")
      .getOrCreate()

    val processor = new GeoProcessor(spark, getClass().getResource("/sample.txt").toString)
    val filtered = Array(
                    Array("Courtyard Long Beach Downtown","US","19"),
                    Array("Comfort Inn & Suites Near Universal Studios","US","208"),
                    Array("Best Western Meridian Inn & Suites. Anaheim-Orange","US","42"),
                    Array("Eurostars Dylan","US","39"),
                    Array("Best Western San Mateo Los Prados Inn","US","1"),
                    Array("Americas Best Value Inn Extended Stay Civic Center Ex.Abigail","US","25"),
                    Array("Holiday Inn Express Hotel & Suites San Francisco Fishermans Wharf","US","7"),
                    Array("Aida Shared Bathroom","US","17"),
                    Array("Citigarden Formerly Ramada Airport North","US","6"),
                    Array("Holiday Inn Express & Suites Sfo North","US","5"),
                    Array("Holiday Inn In'Tl Airport","US","7"),
                    Array("Galaxy Motel","US","6"),
                    Array("Ymca Greenpoint Shared Bathroom","US","8"),
                    Array("Ymca Flushing Shared Bathroom","US","21"),
                    Array("Seton","US","13"),
                    Array("The Roosevelt","US","29"),
                    Array("The Village At Squaw Valley","US","1896"),
                    Array("Ritz Carlton Highlands Lake Tahoe","US","1967"),
                    Array("The Tuscan A Kimpton Hotel","US","8"),
                    Array("Comfort Inn Watsonville","US","37"),
                    Array("Days Inn & Suites Santa Barbara","US","43"),
                    Array("Best Western Plus Pepper Tree","US","61"),
                    Array("Hyatt","US","9"),
                    Array("Comfort Inn By The Bay - San F","US","41"),
                    Array("Comfort Inn & Suites Airport","US","5"),
                    Array("Indigo Santa Barbara","US","6"),
                    Array("Canary Hotel","US","25"),
                    Array("Motel 6 Ventura Beach 218","US","6"),
                    Array("Comfort Inn Ventura Beach","US","5"),
                    Array("Crowne Plaza Ventura","US","14"),
                    Array("Four Seasons The Biltmore Santa Barbara","US","26"),
                    Array("Inn By The Harbor - Santa Barb","US","11"),
                    Array("Fess Parker'S Doubletree - San","US","3"),
                    Array("Best Western Encinita Lodge","US","50"),
                    Array("Best Western Plus Pepper Tree Inn","US","61"),
                    Array("Canary A Kimpton Hotel","US","22"),
                    Array("Hotel Goleta - Santa Barbara","US","17"),
                    Array("Holiday Inn Express Santa Barbara","US","11"),
                    Array("Dolphin Bay Resort & Spa","US","24"),
                    Array("Best Western Plus Sonora Oaks","US","653"),
                    Array("Comfort Inn & Suites Sfo Airport","US","6"),
                    Array("Hilton Garden Inn San Francisco Arpt North","US","12"),
                    Array("Embassy Suites San Fran","US","4"),
                    Array("Best Western Plus Grosvenor Ai","US","4"),
                    Array("Comfort Inn & Suites San Franc","US","6"),
                    Array("Days Inn San Francisco South Oyster Point Airport","US","12"),
                    Array("Baymont Inn & Suites Miami Airport West","US","7"),
                    Array("Best Western Premier Miami International Airport","US","11"),
                    Array("Residence Inn Miami Airport South","US","10"),
                    Array("Ramada Inn Miami Airport North","US","9"))
    val data = spark.sparkContext.parallelize(filtered)
    val elevation = Array(19,208,42,39,1,25,7,17,6,5,7,6,8,21,
            13,29,1896,1967,8,37,43,61,9,41,5,6,25,6,5,14,26,
            11,3,50,61,22,17,11,24,653,6,12,4,4,6,12,7,11,10,9)

    "filterData" should "filter out unnecessary fields and return RDD[Array[String]]" in {
   
        processor.filterData(processor.file).collect().deep should equal (filtered.deep)
    }

    "filterElevation" should "return an RDD[Int] containing elevation "+
        "information about given continent" in {

        processor.filterElevation("US",data).collect().deep should equal (elevation.deep)
    }

    "elevationAverage" should "calculate the average" in {

       processor.elevationAverage(spark.sparkContext.parallelize(elevation)) should equal (110.7)


    }

    "mostCommonWords" should "produce an ordered wordcount" in {
        val res = Array(("Inn",22),("&",10),("Best",9),("Airport",9),("Suites",9),
        ("San",8),("Western",8),("Comfort",7),("The",6),("Santa",6),("Hotel",5),("Barbara",5),("Plus",4),
        ("Holiday",4),("Miami",4),("-",4),("North",4),("Francisco",3),("Express",3),("Bathroom",3),("Beach",3),
        ("Ventura",3),("Shared",3),("Kimpton",2),("Ramada",2),("Ymca",2),("Pepper",2),("Tree",2),("Canary",2),
        ("Bay",2),("Sfo",2),("A",2),("Days",2),("South",2),("By",2),("Motel",2),
        ("Crowne",1),("Tuscan",1),("Premier",1),("Fishermans",1),("In'Tl",1),("Citigarden",1),("Eurostars",1),
        ("Oyster",1),("Mateo",1),("Downtown",1),("Highlands",1),("Goleta",1),("Hilton",1),("Embassy",1),("6",1),
        ("Dylan",1),("Harbor",1),("F",1),("Sonora",1),("Los",1),("Long",1),("Seasons",1),("International",1),
        ("Formerly",1),("Oaks",1),("Baymont",1),("Ex.Abigail",1),("Wharf",1),("Ai",1),("Four",1),("Near",1),
        ("Plaza",1),("Anaheim-Orange",1),("Village",1),("Hyatt",1),("Spa",1),("Biltmore",1),("Dolphin",1),
        ("Watsonville",1),("Indigo",1),("Galaxy",1),("Civic",1),("Point",1),("Flushing",1),("Residence",1),
        ("Franc",1),("Grosvenor",1),("Value",1),("Valley",1),("218",1),("Suites.",1),("Greenpoint",1),
        ("Universal",1),("Resort",1),("Arpt",1),("Parker'S",1),("Ritz",1),("Encinita",1),("Lake",1),("Fess",1),
        ("Carlton",1),("Studios",1),("Doubletree",1),("Squaw",1),("Americas",1),("At",1),("Fran",1),
        ("Barb",1),("Seton",1),("Roosevelt",1),("West",1),("Center",1),("Meridian",1),("Tahoe",1),
        ("Courtyard",1),("Lodge",1),("Extended",1),("Stay",1),("Prados",1),("Aida",1),("Garden",1))
        
        processor.mostCommonWords(data).collect().sortBy(x => (x._1,x._2)) should equal (res.sortBy(x => (x._1,x._2)))
    }

    "mostCommonCountry" should "return the most common country name" in {
        processor.mostCommonCountry(data,getClass().getResource("/countrycodes.csv").toString().drop(5)) should equal ("United States")
    }

    "HotelsInArea" should "calculate how many hotels are within 10km from target area" in {
        processor.hotelsInArea(37.80626, -122.41628) should equal (2)
    }

    val edges = Array(Edge(1,10,1), Edge(10,1,1), Edge(11,1,1), Edge(1,11,1), Edge(1,12,1))
    val vertices = Array((12L,12), (10L,10), (11L,11), (1L,1))
    val graph = Graph(spark.sparkContext.parallelize(vertices),spark.sparkContext.parallelize(edges),0)

    "loadSocial" should "produce a graphx graph" in {
        val attempt = processor.loadSocial(getClass().getResource("/graph_sample.txt").toString().drop(5))
        attempt.edges.count() should equal (5)
        attempt.vertices.count() should equal (4)
        attempt.triplets.collect().deep should equal(graph.triplets.collect().deep)
    }

    "mostActiveUser" should "tell which user has the most connections outwards" in {

        processor.mostActiveUser(graph) should equal (1)
    }

    "pageRankHighest" should "calculate the pageRank and return the most \"famous\" user" in {
        processor.pageRankHighest(graph) should equal (1)
  }

}
