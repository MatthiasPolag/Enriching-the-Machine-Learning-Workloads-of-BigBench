#!/bin/bash

echo "executing query 26"
# Measure time for query execution time
	# Start timer to measure data loading for the file formats
	STARTDATE="`date +%Y/%m/%d:%H:%M:%S`"
	STARTDATE_EPOCH="`date +%s`" # seconds since epochstart

#step 1. 
# run Hive and create the tables
# Write input for k-means into temp table
#hive -f q26.hql
TEMP_DATA_DIR="/home/user1/MP/tempData"
TEMP_RESULT_DIR="/user/user1/bigbenchv2/recommendation"

#not a prettz process, but since overwrite directory doesn't use the correct separator ' ' it is necessary until there is a way to rectify this problem and make it possible to rename the file
	
#create local file to save the result of the query

mkdir /home/user1/MP/tempData
# create the temp_result dir
hadoop fs -mkdir /user/user1/bigbenchv2/recommendation
#save the table data into the file
hive -e 'SELECT * FROM bigbenchv2.q26_results' | sed 's/\t/,/g'> "$TEMP_DATA_DIR/rawData"

PREPSTARTDATE="`date +%Y/%m/%d:%H:%M:%S`"
PREPSTARTDATE_EPOCH="`date +%s`" # seconds since epochstart

#upload the file into hdfs
hdfs dfs -put "$TEMP_DATA_DIR/rawData" "$TEMP_RESULT_DIR/rawData"

#create metadatafile with the correct number of columns AND delete the unnecessary part of the file path
rows=$(wc -l "$TEMP_DATA_DIR/rawData" | sed 's/[^0-9]*//g')

#count number of words
words=$(sed 's/,/ /g' "$TEMP_DATA_DIR/rawData" | wc -w | sed 's/[^0-9]*//g')

#calculate number of rows
columns=$(expr $words / $rows)

#create metadata file
echo  {\"rows\": 92298, \"cols\": 16, \"format\": \"csv\"} > "$TEMP_DATA_DIR/rawData.mtd"



#upload the metadatafile into hdfs
hdfs dfs -put "$TEMP_DATA_DIR/rawData.mtd" "$TEMP_RESULT_DIR"	
	
# clean the result before starting

#hadoop fs -rm -r -f "$RESULT_DIR" 
# create the result dir
hadoop fs -mkdir /user/user1/bigbenchv2/kmeans
RESULT_DIR="/user/user1/bigbenchv2/kmeans"

	

#step 3  set java home
#JAVA_HOME=/usr/java/jdk1.8.0_60/

PREPDATE="`date +%Y/%m/%d:%H:%M:%S`"
	PREPDATE_EPOCH="`date +%s`" # seconds since epoch
	PREPDIFF_s="$(($PREPDATE_EPOCH - $PREPSTARTDATE_EPOCH))"
	PREPDIFF_ms="$(($PREPDIFF_s * 1000))"
	PREPDURATION="$(($PREPDIFF_s / 3600 ))h $((($PREPDIFF_s % 3600) / 60))m $(($PREPDIFF_s % 60))s"

#step 3.     Possibly need to include full path to Kmeans.dml ????
hadoop jar /home/user1/MP/systemml-1.1.0-bin/SystemML.jar -f /home/user1/MP/systemml-1.1.0-bin/scripts/algorithms/Kmeans.dml -nvargs X="$TEMP_RESULT_DIR/rawData" k=8 runs=10 Y="$RESULT_DIR/cluster.txt" isY=TRUE

#step 4 cleanup
rm -r "$TEMP_DATA_DIR"

hadoop fs -rm -r -f "$TEMP_RESULT_DIR"


# Calculate the time
	STOPDATE="`date +%Y/%m/%d:%H:%M:%S`"
	STOPDATE_EPOCH="`date +%s`" # seconds since epoch
	DIFF_s="$(($STOPDATE_EPOCH - $STARTDATE_EPOCH))"
	DIFF_ms="$(($DIFF_s * 1000))"
	DURATION="$(($DIFF_s / 3600 ))h $((($DIFF_s % 3600) / 60))m $(($DIFF_s % 60))s"
# print times
 echo "query execution time: ${DIFF_s} (sec)| ${DURATION}"
