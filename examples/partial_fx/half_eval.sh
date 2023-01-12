ONNX_DIR=$1
LOG_DIR=$2
ENGINE_DIR=$3
if [ ! -d ${LOG_DIR} ]; then
  echo "mkdir ${LOG_DIR}"
  mkdir ${LOG_DIR}
fi
if [ ! -d ${ENGINE_DIR} ]; then
  echo "mkdir ${ENGINE_DIR}"
  mkdir ${ENGINE_DIR}
fi
ONNX_FILES=$(ls $ONNX_DIR)
for onnx_file in ${ONNX_FILES}
do
  # check suffix is onnx
  if [ "${onnx_file##*.}"x = "onnx"x ]; then
    echo "build ${ONNX_DIR}/${onnx_file}"
    /usr/local/TensorRT-release/bin/trtexec --streams=1 --workspace=1024 --fp16 --inputIOFormats=fp16:chw  --outputIOFormats=fp16:chw --onnx=${ONNX_DIR}/${onnx_file} --saveEngine=${ENGINE_DIR}/${onnx_file%.*}_fp16.trt 2>&1 > ${LOG_DIR}/${onnx_file%.*}_trt_build_log.txt
  fi
done