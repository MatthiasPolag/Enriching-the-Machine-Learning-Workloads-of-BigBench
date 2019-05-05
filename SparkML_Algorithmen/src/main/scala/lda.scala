
import org.apache.spark.mllib.clustering.{LDA, DistributedLDAModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.Tokenizer


import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.serializer.JavaSerializer

object lda {
  def main(args: Array[String]) {
//begin timer
val startTime = System.nanoTime

//set up spark context
val conf = new SparkConf().setAppName("Test Project")
val sc = new SparkContext(conf)

//set up hive context
val hiveObj = new HiveContext(sc)
hiveObj.refreshTable("bigbenchv2.product_reviews")



//query data from hive, unlike q 28 review id is not selected, lda most likelz assigns labels itself so selection can be reduced to pr_content
val queryStartTime = System.nanoTime

val sentenceData = hiveObj.sql("SELECT CASE pr_rating WHEN 1 THEN 'NEG' WHEN 2 THEN 'NEG' WHEN 3 THEN 'NEU' WHEN 4 THEN 'POS' WHEN 5 THEN 'POS' END AS pr_r_rating, pr_content FROM bigbenchv2.product_reviews WHERE pmod(pr_review_id, 5) IN (1,2,3)").toDF("label", "sentence")

val queryTime = (System.nanoTime - queryStartTime) / 1e9d

//tranform string input into tokenized data
val preparationStartTime = System.nanoTime

val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
val wordsData = tokenizer.transform(sentenceData)

//create tfidf matrix with feature vectors
val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures")

val featurizedData = hashingTF.transform(wordsData)

//val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
//val idfModel = idf.fit(featurizedData)

//val rescaledData = idfModel.transform(featurizedData)

//get the feature vectors into the right format
//val colData = rescaledData.select("features")
val colData = featurizedData.select("rawFeatures")
val rdData = colData.rdd
val vecData = rdData.map(r => r(0).asInstanceOf[org.apache.spark.mllib.linalg.Vector])
val corpus = vecData.zipWithIndex.map(_.swap).cache()

val preparationTime = (System.nanoTime - preparationStartTime) / 1e9d

//create the lda model with 3 topics
val executionStartTime = System.nanoTime
val ldaModel = new LDA().setK(3).run(corpus)

val executionTime = (System.nanoTime - executionStartTime) / 1e9d

// Output topics. Each is a distribution over words (matching word count vectors)
println("Learned topics (as distributions over vocab of " + ldaModel.vocabSize + " words):")
val topics = ldaModel.topicsMatrix
for (topic <- Range(0, 3)) {
  print("Topic " + topic + ":")
  for (word <- Range(0, ldaModel.vocabSize)) { print(" " + topics(word, topic)); }
  println()
}

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
