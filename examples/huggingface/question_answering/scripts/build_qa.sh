/usr/local/TensorRT-8.4.3.1/targets/x86_64-linux-gnu/bin/trtexec --streams=1 \
                                                                 --explicitBatch \
                                                                 --workspace=16384 \
                                                                 --int8 \
                                                                 --onnx=bert_qa_calib_128_w8_a8_naive_qat.onnx \
                                                                 --minShapes=input_ids:1x128,attention_mask:1x128,token_type_ids:1x128 \
                                                                 --optShapes=input_ids:16x128,attention_mask:16x128,token_type_ids:16x128 \
                                                                 --maxShapes=input_ids:32x128,attention_mask:32x128,token_type_ids:32x128 \
                                                                 --saveEngine=bert_qa_calib_128_w8_a8_naive_qat.trt