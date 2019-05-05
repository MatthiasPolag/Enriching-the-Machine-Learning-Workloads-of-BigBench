import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.serializer.JavaSerializer

object logRegQ28 {
  def main(args: Array[String]) {
//begin timer
val startTime = System.nanoTime

//set up spark context
val conf = new SparkConf().setAppName("Test Project")
val sc = new SparkContext(conf)
//set up hive context
val hiveObj = new HiveContext(sc)
hiveObj.refreshTable("bigbenchv2.product_reviews")


//query data from hive, unlike q 28 review id is not selected, and only 2 classes exist. Neutral is mapped to negative.
// Negative is 0; Positive is 1
val queryStartTime = System.nanoTime

val sentenceData = hiveObj.sql("SELECT CASE pr_rating WHEN 1 THEN '0' WHEN 2 THEN '0' WHEN 3 THEN '1' WHEN 4 THEN '2' WHEN 5 THEN '2' END AS pr_r_rating, pr_content FROM bigbenchv2.product_reviews WHERE pmod(pr_review_id, 5) IN (1,2,3)").toDF("label", "sentence")

val queryTime = (System.nanoTime - queryStartTime) / 1e9d

//prepare the data
val preparationStartTime = System.nanoTime


//tranform string input into tokenized data
val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
val wordsData = tokenizer.transform(sentenceData)

//create tfidf matrix with feature vectors
val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures")

val featurizedData = hashingTF.transform(wordsData)

val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
val idfModel = idf.fit(featurizedData)

val rescaledData = idfModel.transform(featurizedData)

//transform the tfidf into labeled point
val labeledPoint = rescaledData.map( row => LabeledPoint(
 row.getAs[String]("label").toDouble, 
 row.getAs[org.apache.spark.mllib.linalg.Vector]("features")
))

// Split data into training (60%) and test (40%).
val splits = labeledPoint.randomSplit(Array(0.6, 0.4), seed = 112L)
val trainingData = splits(0).cache()
val testData = splits(1)

val preparationTime = (System.nanoTime - preparationStartTime) / 1e9d

//train the model
val executionStartTime = System.nanoTime

val model = new LogisticRegressionWithLBFGS().setNumClasses(3).run(trainingData)

val executionTime = (System.nanoTime - executionStartTime) / 1e9d

// Compute raw scores on the test set.
val predictionAndLabels = testData.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}

// Get evaluation metrics.
val metrics = new MulticlassMetrics(predictionAndLabels)
val precision = metrics.precision
println("Precision = " + precision)


val totalTime = (System.nanoTime - startTime) / 1e9d

println("Query time: " )
println(queryTime)

println("Preparation time: " )
println(preparationTime)

println("Execution time: " )
println(executionTime)

println("Total run time: " )
println(totalTime)
}
}
