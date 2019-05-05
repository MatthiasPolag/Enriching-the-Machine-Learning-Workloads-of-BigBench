#!/bin/bash

echo "executing query 26"
# Measure time for query execution time
	# Start timer to measure data loading for the file formats
	STARTDATE="`date +%Y/%m/%d:%H:%M:%S`"
	STARTDATE_EPOCH="`date +%s`" # seconds since epochstart


#step 1.    NEED TO BE RECOMMENTED, Q renamed!
TEMP_DATA_DIR="/home/user1/MP/tempData"
TEMP_RESULT_DIR="/user/user1/bigbenchv2/recommendation"
	
#create local file to save the result of the query

mkdir /home/user1/MP/tempData

# create the temp_result dir
hadoop fs -mkdir /user/user1/bigbenchv2/recommendation


#save the table data into the file
hive -e 'SELECT * FROM bigbenchv2.recommendation_data' | sed 's/\t/,/g'> "$TEMP_DATA_DIR/rawData"

PREPSTARTDATE="`date +%Y/%m/%d:%H:%M:%S`"
PREPSTARTDATE_EPOCH="`date +%s`" # seconds since epochstart

#split it into the feature vector part 
sed 's/^[^,]*,//g' "$TEMP_DATA_DIR/rawData" > "$TEMP_DATA_DIR/featureVectors"
#and the label part
sed 's/,.*//g' "$TEMP_DATA_DIR/rawData" > "$TEMP_DATA_DIR/labels"


#create metadatafiles with the correct number of columns AND delete the unnecessary part of the file path
rows=$(wc -l "$TEMP_DATA_DIR/featureVectors" | sed 's/[^0-9]*//g')

#count number of words
words=$(sed 's/,/ /g' "$TEMP_DATA_DIR/featureVectors" | wc -w | sed 's/[^0-9]*//g')

#calculate number of rows
columns=$(expr $words / $rows)

#calculate 60% split
split=$(expr $rows \* 60 / 100)

#store rows variable for testing file
testrows=$(expr $rows - $split)

secondpart=$(expr $split + 1)

#split into training and testing set
head -n  88671 "$TEMP_DATA_DIR/featureVectors" > "$TEMP_DATA_DIR/featureVectoraa"
tail -n 59114 "$TEMP_DATA_DIR/featureVectors" > "$TEMP_DATA_DIR/featureVectorab"
head -n  88671 "$TEMP_DATA_DIR/labels" > "$TEMP_DATA_DIR/labelaa" 
tail -n 59114 "$TEMP_DATA_DIR/labels" > "$TEMP_DATA_DIR/labelab"

#rename files
mv "$TEMP_DATA_DIR/featureVectoraa" "$TEMP_DATA_DIR/trainingData"
mv "$TEMP_DATA_DIR/featureVectorab" "$TEMP_DATA_DIR/testData"

mv "$TEMP_DATA_DIR/labelaa" "$TEMP_DATA_DIR/trainingLabels"
mv "$TEMP_DATA_DIR/labelab" "$TEMP_DATA_DIR/testLabels"

#create metadata file
echo  {\"rows\": 88671, \"cols\": 19, \"format\": \"csv\"} > "$TEMP_DATA_DIR/trainingData.mtd"
echo  {\"rows\": 88671, \"cols\": 1, \"format\": \"csv\"} > "$TEMP_DATA_DIR/trainingLabels.mtd"

echo  {\"rows\": 59114, \"cols\": 19, \"format\": \"csv\"} > "$TEMP_DATA_DIR/testData.mtd"
echo  {\"rows\": 59114, \"cols\": 1, \"format\": \"csv\"} > "$TEMP_DATA_DIR/testLabels.mtd"

#upload the files into hdfs
hdfs dfs -put "$TEMP_DATA_DIR/trainingData" "$TEMP_RESULT_DIR"
hdfs dfs -put "$TEMP_DATA_DIR/trainingLabels" "$TEMP_RESULT_DIR"

hdfs dfs -put "$TEMP_DATA_DIR/trainingData.mtd" "$TEMP_RESULT_DIR"
hdfs dfs -put "$TEMP_DATA_DIR/trainingLabels.mtd" "$TEMP_RESULT_DIR"

hdfs dfs -put "$TEMP_DATA_DIR/testData" "$TEMP_RESULT_DIR"
hdfs dfs -put "$TEMP_DATA_DIR/testLabels" "$TEMP_RESULT_DIR"	

hdfs dfs -put "$TEMP_DATA_DIR/testData.mtd" "$TEMP_RESULT_DIR"
hdfs dfs -put "$TEMP_DATA_DIR/testLabels.mtd" "$TEMP_RESULT_DIR"


PREPDATE="`date +%Y/%m/%d:%H:%M:%S`"
	PREPDATE_EPOCH="`date +%s`" # seconds since epoch
	PREPDIFF_s="$(($PREPDATE_EPOCH - $PREPSTARTDATE_EPOCH))"
	PREPDIFF_ms="$(($PREPDIFF_s * 1000))"
	PREPDURATION="$(($PREPDIFF_s / 3600 ))h $((($PREPDIFF_s % 3600) / 60))m $(($PREPDIFF_s % 60))s"

#step 3  set java home
#JAVA_HOME=/usr/java/jdk1.8.0_60/

# clean the result before starting
hadoop fs -rm -r -f /user/cloudera/bigbenchv2/naiveBayes

#step 3.     run the algorithm
hadoop fs -mkdir /user/user1/bigbenchv2/naiveBayes

RESULT_DIR="/user/user1/bigbenchv2/naiveBayes"


	EXSTARTDATE="`date +%Y/%m/%d:%H:%M:%S`"
	EXSTARTDATE_EPOCH="`date +%s`" # seconds since epochstart
hadoop jar /home/user1/MP/systemml-1.1.0-bin/SystemML.jar -f "/home/user1/MP/systemml-1.1.0-bin/scripts/algorithms/naive-bayes.dml" -nvargs X="$TEMP_RESULT_DIR/trainingData" Y="$TEMP_RESULT_DIR/trainingLabels" Log="$RESULT_DIR/log" prior="$RESULT_DIR/prior" accuracy="$RESULT_DIR/accuracy" conditionals="$RESULT_DIR/conditionals"

EXDATE="`date +%Y/%m/%d:%H:%M:%S`"
	EXDATE_EPOCH="`date +%s`" # seconds since epoch
	EXDIFF_s="$(($EXDATE_EPOCH - $EXSTARTDATE_EPOCH))"
	EXDIFF_ms="$(($EXDIFF_s * 1000))"
	EXDURATION="$(($EXDIFF_s / 3600 ))h $((($PREPDIFF_s % 3600) / 60))m $(($PREPDIFF_s % 60))s"


#test the algorithm
hadoop jar /home/user1/MP/systemml-1.1.0-bin/SystemML.jar -f "/home/user1/MP/systemml-1.1.0-bin/scripts/algorithms/naive-bayes-predict.dml" -nvargs X="$TEMP_RESULT_DIR/testData" Y="$TEMP_RESULT_DIR/testLabels" Log=/"RESULT_DIR/log" prior="$RESULT_DIR/prior" accuracy="$RESULT_DIR/accuracy" conditionals="$RESULT_DIR/conditionals" probabilities="$RESULT_DIR/probabilities"


#step 4 cleanup 
#rm -r "$TEMP_DATA_DIR"

hadoop fs -rm -r -f "$TEMP_RESULT_DIR"

# Calculate the time
	STOPDATE="`date +%Y/%m/%d:%H:%M:%S`"
	STOPDATE_EPOCH="`date +%s`" # seconds since epoch
	DIFF_s="$(($STOPDATE_EPOCH - $STARTDATE_EPOCH))"
	DIFF_ms="$(($DIFF_s * 1000))"
	DURATION="$(($DIFF_s / 3600 ))h $((($DIFF_s % 3600) / 60))m $(($DIFF_s % 60))s"
# print times
 echo "query preparation time: ${PREPDIFF_s} (sec)| ${PREPDURATION}"
 echo "query execution time: ${DIFF_s} (sec)| ${DURATION}"
