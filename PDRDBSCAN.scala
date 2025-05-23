spark-submit \
  --class PDRDBSCAN \
  --master spark://192.168.192.163:7077 \
  --driver-memory 1g \
  --executor-memory 1g \
  --num-executors 1 \
  /root/pdrdbscan_2.12-0.1.jar


spark-submit \
  --class PDRDBSCAN \
  --master spark://192.168.192.163:7077 \
  --driver-memory 2g \
  --executor-memory 2g \
  --executor-cores 2 \
  --num-executors 3 \
  /root/pdr_2.11-0.1.0-SNAPSHOT.jar

spark-submit \
  --class PDR_DBSCAN_Spark \
  --master spark://192.168.192.163:7077 \
  --driver-memory 1g \
  --executor-memory 1g \
  --executor-cores 1 \
  --num-executors 4 \
  /root/pdr_2.11-0.1.0-SNAPSHOT.jar

spark-submit \
  --class PDR_DBSCAN_Spark \
  --master spark://192.168.192.163:7077 \
  --driver-memory 1g \
  --executor-memory 1g \
  --executor-cores 1 \
  --num-executors 4 \
  --jars /root/lib/rtree-0.11.jar,/root/lib/rxjava-1.3.8.jar \
  /root/pdr_2.11-0.1.0-SNAPSHOT.jar

spark-submit \
  --class PDR_DBSCAN_Spark \
  --master spark://192.168.192.163:7077 \
  --driver-memory 1g \
  --executor-memory 1g \
  --executor-cores 1 \
  --num-executors 4 \
  --jars /root/lib/rtree-0.11.jar,/root/lib/rxjava-1.3.8.jar,/root/lib/guavamini-1.0.jar \
  /root/pdr_2.11-0.1.0-SNAPSHOT.jar


spark-submit \
  --class PDRDBSCAN \
  --master spark://192.168.192.163:7077 \
  --driver-memory 1g \
  --executor-memory 1g \
  --executor-cores 1 \
  --num-executors 4 \
  --conf spark.hadoop.fs.defaultFS=file:/// \
  /root/pdr_2.11-0.1.0-SNAPSHOT.jar

