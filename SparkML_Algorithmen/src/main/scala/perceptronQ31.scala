
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.ml.feature.VectorAssembler


import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.serializer.JavaSerializer

object mlpQ28 {
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

val df = hiveObj.sql("SELECT CASE WHEN clicks_in_0 > 3.871510156577754 THEN 1.0 ELSE 0.0 END AS label,clicks_in_1,clicks_in_2,clicks_in_3,clicks_in_4,clicks_in_5,clicks_in_6,clicks_in_7,clicks_in_8,clicks_in_9,clicks_in_10,clicks_in_11,clicks_in_12,clicks_in_13,clicks_in_14,clicks_in_15,clicks_in_16,clicks_in_17,clicks_in_18,clicks_in_19 FROM bigbenchv2.category_clicks")

val queryTime = (System.nanoTime - queryStartTime) / 1e9d


//prepare the data
val preparationStartTime = System.nanoTime

val assembler = new VectorAssembler().setInputCols(Array("clicks_in_1", "clicks_in_2", "clicks_in_3", "clicks_in_4", "clicks_in_5", "clicks_in_6" , "clicks_in_7", "clicks_in_8", "clicks_in_9" , "clicks_in_10", "clicks_in_11", "clicks_in_12" , "clicks_in_13", "clicks_in_14", "clicks_in_15" , "clicks_in_16", "clicks_in_17", "clicks_in_18" , "clicks_in_19")).setOutputCol("features") 

val vd = assembler.transform(df)



// Split data into training (60%) and test (40%).
val splits = vd.randomSplit(Array(0.6, 0.4), seed = 112L)
val trainingData = splits(0).cache()
val testData = splits(1)

val preparationTime = (System.nanoTime - preparationStartTime) / 1e9d


val executionStartTime = System.nanoTime
// specify layers for the neural network:
// input layer of size 4 (features), two intermediate of size 5 and 4
// and output of size 2 (classes)
val layers = Array[Int](19, 5, 4, 3)


//----------- create the trainer and set its parameters-------------------------------------------------------
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

// train the model
val model = trainer.fit(trainingData)

val executionTime = (System.nanoTime - executionStartTime) / 1e9d

// compute precision on the test set
val testStartTime = System.nanoTime

val result = model.transform(testData)
val predictionAndLabels = result.select("prediction", "label")
val testErr = predictionAndLabels.filter(result("prediction") !== result("label")).count().toDouble / testData.count()
val evaluator = new MulticlassClassificationEvaluator().setMetricName("precision")
println("Precision:" + evaluator.evaluate(predictionAndLabels))
val testTime = (System.nanoTime - testStartTime) / 1e9d


val totalTime = (System.nanoTime - startTime) / 1e9d
println("Query time: " )
println(queryTime)

println("Preparation time: " )
println(preparationTime)

println("Execution time: " )
println(executionTime)

println("Test time: " )
println(testTime)

println("Total run time: " )
println(totalTime)
}
}
