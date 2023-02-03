LOG_DIR=$1
LOG_FILES=$(ls $LOG_DIR)
for log_file in ${LOG_FILES}
do
  throughput=`cat $LOG_DIR/${log_file} | grep Throughput`
  echo "$log_file ${throughput}"
done