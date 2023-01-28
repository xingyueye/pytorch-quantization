python3 run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad.py \
  --max_seq_length 128 \
  --doc_stride 32 \
  --output_dir calib/bert-base-uncased \
  --do_calib \
  --do_eval