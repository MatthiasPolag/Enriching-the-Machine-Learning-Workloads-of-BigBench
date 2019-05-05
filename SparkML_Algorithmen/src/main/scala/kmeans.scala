import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.ml.clustering.KMeans


import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.serializer.JavaSerializer

object kmeans {
  def main(args: Array[String]) {
//begin timer
val startTime = System.nanoTime
//set up spark context
val conf = new SparkConf().setAppName("Test Project")
val sc = new SparkContext(conf)

// Loads data
val hiveObj = new HiveContext(sc)

hiveObj.refreshTable("bigbenchv2.store_sales")
hiveObj.refreshTable("bigbenchv2.items")

//Start query timer
val queryStartTime = System.nanoTime

//query the data directly from hive
val df = hiveObj.sql("SELECT ss.ss_customer_id AS cid, count(CASE WHEN i.i_class_id=1  THEN 1 ELSE NULL END) AS id1,count(CASE WHEN i.i_class_id=3  THEN 1 ELSE NULL END) AS id3,count(CASE WHEN i.i_class_id=5  THEN 1 ELSE NULL END) AS id5,count(CASE WHEN i.i_class_id=7  THEN 1 ELSE NULL END) AS id7, count(CASE WHEN i.i_class_id=9  THEN 1 ELSE NULL END) AS id9,count(CASE WHEN i.i_class_id=11 THEN 1 ELSE NULL END) AS id11,count(CASE WHEN i.i_class_id=13 THEN 1 ELSE NULL END) AS id13,count(CASE WHEN i.i_class_id=15 THEN 1 ELSE NULL END) AS id15,count(CASE WHEN i.i_class_id=2  THEN 1 ELSE NULL END) AS id2,count(CASE WHEN i.i_class_id=4  THEN 1 ELSE NULL END) AS id4,count(CASE WHEN i.i_class_id=6  THEN 1 ELSE NULL END) AS id6,count(CASE WHEN i.i_class_id=12 THEN 1 ELSE NULL END) AS id12, count(CASE WHEN i.i_class_id=8  THEN 1 ELSE NULL END) AS id8,count(CASE WHEN i.i_class_id=10 THEN 1 ELSE NULL END) AS id10,count(CASE WHEN i.i_class_id=14 THEN 1 ELSE NULL END) AS id14,count(CASE WHEN i.i_class_id=16 THEN 1 ELSE NULL END) AS id16 FROM bigbenchv2.store_sales ss INNER JOIN bigbenchv2.items i ON ss.ss_item_id = i.i_item_id WHERE i.i_category_name IN ('cat#01','cat#02','cat#03','cat#04','cat#05','cat#06','cat#07','cat#08','cat#09','cat#10','cat#11','cat#12','cat#013','cat#14','cat#15') AND ss.ss_customer_id IS NOT NULL GROUP BY ss.ss_customer_id HAVING count(ss.ss_item_id) > 3")

val queryTime = (System.nanoTime - queryStartTime) / 1e9d

//Transform the data to the right format
val preparationStartTime = System.nanoTime

val assembler = new VectorAssembler().setInputCols(Array("cid", "id1", "id2", "id3", "id4", "id5", "id6", "id7", "id8", "id9", "id10" )).setOutputCol("FEATURES") 

val vd = assembler.transform(df)


val preparationTime = (System.nanoTime - preparationStartTime) / 1e9d

// Trains a k-means model
val executionStartTime = System.nanoTime

val kmeans = new KMeans().setK(8).setFeaturesCol("features").setPredictionCol("prediction")
val model = kmeans.fit(vd)

val executionTime = (System.nanoTime - executionStartTime) / 1e9d

// Shows the result
println("Final Centers: ")
model.clusterCenters.foreach(println)

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
