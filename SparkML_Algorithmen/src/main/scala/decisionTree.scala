

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors


import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.serializer.JavaSerializer

object decisionTreeQ31 {
  def main(args: Array[String]) {
//begin timer
val startTime = System.nanoTime
//set up spark context
val conf = new SparkConf().setAppName("Test Project")
val sc = new SparkContext(conf)

// Loads data
val hiveObj = new HiveContext(sc)

hiveObj.refreshTable("category_clicks")


//query the data directly from hive
val queryStartTime = System.nanoTime

val df = hiveObj.sql("SELECT wl_customer_id,CASE WHEN clicks_in_0 > 3.871510156577754 THEN 1 ELSE 0 END AS interest,clicks_in_1,clicks_in_2,clicks_in_3,clicks_in_4,clicks_in_5,clicks_in_6,clicks_in_7,clicks_in_8,clicks_in_9,clicks_in_10,clicks_in_11,clicks_in_12,clicks_in_13,clicks_in_14,clicks_in_15,clicks_in_16,clicks_in_17,clicks_in_18,clicks_in_19 FROM bigbenchv2.category_clicks")

val queryTime = (System.nanoTime - queryStartTime) / 1e9d

//prepare the data
val preparationStartTime = System.nanoTime

val labeledPoint = df.map( row => LabeledPoint(
 row.getAs[Integer]("interest").toDouble, 
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

// Train a DecisionTree model.
//  Empty categoricalFeaturesInfo indicates all features are continuous.
val executionStartTime = System.nanoTime

val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val impurity = "gini"
val maxDepth = 7
val maxBins = 32

val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
  impurity, maxDepth, maxBins)

val executionTime = (System.nanoTime - executionStartTime) / 1e9d

// Evaluate model on test instances and compute test error
val testStartTime = System.nanoTime

val labelAndPreds = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()
println("Test Error = " + testErr)
println("Learned classification tree model:\n" + model.toDebugString)

val testTime = (System.nanoTime - testStartTime) / 1e9d
// Save and load model
//model.save(sc, "target/tmp/myDecisionTreeClassificationModel")
//val sameModel = DecisionTreeModel.load(sc, "target/tmp/myDecisionTreeClassificationModelReg")


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
