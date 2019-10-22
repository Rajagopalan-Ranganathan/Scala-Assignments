package questions

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.classification.{LogisticRegression,LogisticRegressionModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{StructType,StructField,StringType, DoubleType}
import org.apache.spark.sql.SQLContext
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.sql.Row



/**
* GamingProcessor is used to predict if the user is a subscriber.
* You can find the data files from /resources/gaming_data.
* Data schema is https://github.com/cloudwicklabs/generator/wiki/OSGE-Schema
* (first table)
*
* Use Spark's machine learning library mllib's logistic regression algorithm.
* https://spark.apache.org/docs/latest/ml-classification-regression.html#binomial-logistic-regression
*
* Use these features for training your model:
*   - gender 
*   - age
*   - country
*   - friend_count
*   - lifetime
*   - citygame_played
*   - pictionarygame_played
*   - scramblegame_played
*   - snipergame_played
*
*   -paid_subscriber(this is the feature to predict)
*
* The data contains categorical features, so you need to
* change them accordingly.
* https://spark.apache.org/docs/latest/ml-features.html
*
*
*
*/

class GamingProcessor() {


  // these parameters can be changed
  val spark = SparkSession.builder
  .master("local")
  .appName("gaming")
  .config("spark.driver.memory", "5g")
  .config("spark.executor.memory", "2g")
  .getOrCreate()


  import spark.implicits._


  /**
  * convert creates a dataframe, removes unnecessary colums and converts the rest to right format. 
  * Data schema:
  *   - gender: Double (1 if male else 0)
  *   - age: Double
  *   - country: String
  *   - friend_count: Double
  *   - lifetime: Double
  *   - game1: Double (citygame_played)
  *   - game2: Double (pictionarygame_played)
  *   - game3: Double (scramblegame_played)
  *   - game4: Double (snipergame_played)
  *   - paid_customer: Double (1 if yes else 0)
  *
  * @param path to file
  * @return converted DataFrame
  */
  def convert(path: String): DataFrame = {
    val schemaStruct = StructType(Array(
      StructField("gender", DoubleType, nullable = false),
      StructField("age", DoubleType, nullable = false),
      StructField("country", StringType, nullable = true),
      StructField("friend_count", DoubleType, nullable = false),
      StructField("lifetime", DoubleType, nullable = false),
      StructField("game1", DoubleType, nullable = false),
      StructField("game2", DoubleType, nullable = false),
      StructField("game3", DoubleType, nullable = false),
      StructField("game4", DoubleType, nullable = false),
      StructField("paid_customer", DoubleType, nullable = false))
    )

    val players = spark.sparkContext.textFile(path)
    val data = players
      .map(_.split(","))
      .map(player => Row(
        if (player(3).trim.equalsIgnoreCase("male")) 1.0 else 0.0,
        player(4).toDouble,
        player(6),
        player(8).toDouble,
        player(9).toDouble,
        player(10).toDouble,
        player(11).toDouble,
        player(12).toDouble,
        player(13).toDouble,
        if (player(15).trim.equalsIgnoreCase("yes")) 1.0 else 0.0
      ))

      //       .map(player => {
      //   if (player(3).trim.equalsIgnoreCase("male")) 1.0 else 0.0
      //   player(4).toDouble
      //   player(6)
      //   player(8).toDouble
      //   player(9).toDouble
      //   player(10).toDouble
      //   player(11).toDouble
      //   player(12).toDouble
      //   player(13).toDouble 
      //   if (player(15).trim.equalsIgnoreCase("yes")) 1.0 else 0.0
      // }).toDF()
    
     spark.createDataFrame(data, schemaStruct)

 }

  /**
  * indexer converts categorical features into doubles.
  * https://spark.apache.org/docs/latest/ml-features.html
  * 'country' is the only categorical feature.
  * After these modifications schema should be:
  *
  *   - gender: Double (1 if male else 0)
  *   - age: Double
  *   - country: String
  *   - friend_count: Double
  *   - lifetime: Double
  *   - game1: Double (citygame_played)
  *   - game2: Double (pictionarygame_played)
  *   - game3: Double (scramblegame_played)
  *   - game4: Double (snipergame_played)
  *   - paid_customer: Double (1 if yes else 0)
  *   - country_index: Double
  *
  * @param df DataFrame
  * @return Dataframe
  */
  def indexer(df: DataFrame): DataFrame = {
      val ret = new StringIndexer().setInputCol("country").setOutputCol("country_index").fit(df).transform(df)
      return ret
  }

  /**
  * Combine features into one vector. Most mllib algorithms require this step
  * https://spark.apache.org/docs/latest/ml-features.html#vectorassembler
  * Column name should be 'features'
  *
  * @param Dataframe that is transformed using indexer
  * @return Dataframe
  */
  def featureAssembler(df: DataFrame): DataFrame = {
    
    val assembler = new VectorAssembler().setInputCols(Array("gender", "age","friend_count","lifetime","game1","game2","game3","game4","paid_customer")).setOutputCol("features")
    val output = assembler.transform(df)
    return output
}
  /**
  * To improve performance data also need to be standardized, so use
  * https://spark.apache.org/docs/latest/ml-features.html#standardscaler
  *
  * @param Dataframe that is transformed using featureAssembler
  * @param  name of the scaled feature vector (output column name)
  * @return Dataframe
  */
  def scaler(df: DataFrame, outputColName: String): DataFrame = {
    val scaler = new StandardScaler()
  .setInputCol("features")
  .setOutputCol(outputColName)
  .setWithStd(true)
  .setWithMean(false)

    // Compute summary statistics by fitting the StandardScaler.
    val scalerModel = scaler.fit(df)
    val scaledData = scalerModel.transform(df)
    return scaledData
  }

  /**
  * createModel creates a logistic regression model
  * When training, 5 iterations should be enough.
  *
  * @param transformed dataframe
  * @param featuresCol name of the features columns
  * @param labelCol name of the label col (paid_customer)
  * @param predCol name of the prediction col
  * @return trained LogisticRegressionModel
  */
  def createModel(training: DataFrame, featuresCol: String, labelCol: String, predCol: String): LogisticRegressionModel = {
     val logmodel = new LogisticRegression()
       logmodel.setMaxIter(5).setFeaturesCol(featuresCol).setPredictionCol(predCol).setLabelCol(labelCol)
         val lrModel = logmodel.fit(training)
         return lrModel
  } 


  /**
  * Given a transformed and normalized dataset
  * this method predicts if the customer is going to
  * subscribe to the service.
  *
  * @param model trained logistic regression model
  * @param dataToPredict normalized data for prediction
  * @return DataFrame predicted scores (1.0 == yes, 0.0 == no)
  */
  def predict(model: LogisticRegressionModel, dataToPredict: DataFrame): DataFrame = {
          val result =  model.transform(dataToPredict)
         return result

  }

}
/**
*
*  Change the student id
*/
object GamingProcessor {
    val studentId = "601153"
}
