
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.mllib.evaluation.MulticlassMetrics


import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.serializer.JavaSerializer

object nBayesQ31 {
  def main(args: Array[String]) {
//begin timer
val startTime = System.nanoTime

//set up spark context
val conf = new SparkConf().setAppName("Test Project")
val sc = new SparkContext(conf)

//set up hive context
val hiveObj = new HiveContext(sc)
hiveObj.refreshTable("bigbenchv2.product_reviews")

//query the data directly from hive
val queryStartTime = System.nanoTime

val df = hiveObj.sql("SELECT CASE WHEN clicks_in_0 > 3.871510156577754 THEN 1.0 ELSE 0.0 END AS interest,clicks_in_1,clicks_in_2,clicks_in_3,clicks_in_4,clicks_in_5,clicks_in_6,clicks_in_7,clicks_in_8,clicks_in_9,clicks_in_10,clicks_in_11,clicks_in_12,clicks_in_13,clicks_in_14,clicks_in_15,clicks_in_16,clicks_in_17,clicks_in_18,clicks_in_19 FROM bigbenchv2.category_clicks")

val queryTime = (System.nanoTime - queryStartTime) / 1e9d

val preparationStartTime = System.nanoTime

val labeledPoint = df.map( row => LabeledPoint(
 row.getAs[Double]("interest").toDouble, 
Vectors.dense(
 row.getAs[Long]("clicks_in_1").toDouble,
row.getAs[Long]("clicks_in_2").toDouble,
row.getAs[Long]("clicks_in_3").toDouble,
row.getAs[Long]("clicks_in_4").toDouble,
row.getAs[Long]("clicks_in_5").toDouble,
row.getAs[Long]("clicks_in_6").toDouble,
row.getAs[Long]("clicks_in_7").toDouble,
row.getAs[Long]("clicks_in_8").toDouble,
row.getAs[Long]("clicks_in_9").toDouble,
row.getAs[Long]("clicks_in_10").toDouble,
row.getAs[Long]("clicks_in_11").toDouble,
row.getAs[Long]("clicks_in_12").toDouble,
row.getAs[Long]("clicks_in_13").toDouble,
row.getAs[Long]("clicks_in_14").toDouble,
row.getAs[Long]("clicks_in_15").toDouble,
row.getAs[Long]("clicks_in_16").toDouble,
row.getAs[Long]("clicks_in_17").toDouble,
row.getAs[Long]("clicks_in_18").toDouble,
row.getAs[Long]("clicks_in_19").toDouble)
))

// Split the data into training and test sets (30% held out for testing)
val splits = labeledPoint.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))

val preparationTime = (System.nanoTime - preparationStartTime) / 1e9d

//-----------------------------------------Actual model------------------------------------------
val executionStartTime = System.nanoTime

val model = NaiveBayes.train(trainingData, lambda = 1.0, modelType = "multinomial")

val executionTime = (System.nanoTime - executionStartTime) / 1e9d


val predictionAndLabel = testData.map(p => (model.predict(p.features), p.label))
val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / testData.count()


val metrics = new MulticlassMetrics(predictionAndLabel)

println("Confusion matrix:")
println(metrics.confusionMatrix)

// Save and load model
//model.save(sc, "target/tmp/myNaiveBayesModel")
//val sameModel = NaiveBayesModel.load(sc, "target/tmp/myNaiveBayesModel")

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
