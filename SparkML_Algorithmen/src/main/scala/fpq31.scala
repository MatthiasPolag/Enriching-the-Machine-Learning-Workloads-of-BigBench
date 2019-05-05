
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.rdd.RDD

import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.WrappedArray
import scala.collection.mutable.ListBuffer
import org.apache.spark.serializer.JavaSerializer
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.Row

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.serializer.JavaSerializer

object fpQ31 {
  def main(args: Array[String]) {
//begin timer
val startTime = System.nanoTime

//set up spark context
val conf = new SparkConf().setAppName("Test Project")
val sc = new SparkContext(conf)

val hiveObj = new HiveContext(sc)

hiveObj.refreshTable("bigbenchv2.store_sales")

val queryStartTime = System.nanoTime

val df = hiveObj.sql("Select collect_set(ss_item_id) FROM bigbenchv2.store_sales GROUP BY ss_transaction_id")

val queryTime = (System.nanoTime - queryStartTime) / 1e9d

//prepare the data
val preparationStartTime = System.nanoTime

val transactions = df.map(r => r.get(0).asInstanceOf[WrappedArray[Long]].toArray.map(_.toString))

val preparationTime = (System.nanoTime - preparationStartTime) / 1e9d

val executionStartTime = System.nanoTime

val fpg = new FPGrowth()
  fpg.setMinSupport(0.004)  //seriouslz doesn|t find sets with support of 0.02
  fpg.setNumPartitions(2)
val model = fpg.run(transactions)

val executionTime = (System.nanoTime - executionStartTime) / 1e9d

model.freqItemsets.collect().foreach { itemset =>
  println(s"Set found: ${itemset.items.mkString("[", ",", "]")},${itemset.freq}")
}


/*
val minConfidence = 0.01
model.generateAssociationRules(minConfidence).collect().foreach { rule =>
  println(
    rule.antecedent.mkString("[", ",", "]")
      + " => " + rule.consequent .mkString("[", ",", "]")
      + ", " + rule.confidence)
}
*/
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

