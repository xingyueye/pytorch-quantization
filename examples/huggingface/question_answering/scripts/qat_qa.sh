CUDA_VISIBLE_DEVICES=0 python3 run_qa.py \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --tokenizer_name bert-base-uncased \
  --dataset_name squad.py \
  --per_device_train_batch_size 12 \
  --learning_rate 4e-5 \
  --num_train_epochs 2 \
  --max_seq_length 128 \
  --doc_stride 32 \
  --output_dir finetuned_int8/bert-base-uncased-lr4e-5-bs12-epoch2-GPU1 \
  --save_steps 0 \
  --do_train \
  --do_eval  \
  --pretrained_calib calib/bert-base-uncased/pytorch_model.bin \
  --remove_unused_columns False
