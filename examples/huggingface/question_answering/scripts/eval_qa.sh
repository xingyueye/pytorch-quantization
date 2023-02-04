CUDA_VISIBLE_DEVICES=0 python3 eval_trt_qa.py \
  --tokenizer_name bert-base-uncased \
  --dataset_name squad.py \
  --trt_engine_path bert_qa_calib_128_w8_a8_naive_qat.trt \
  --max_seq_length 128 \
  --per_device_eval_batch_size 32 \
  --doc_stride 32 \
  --output_dir ./