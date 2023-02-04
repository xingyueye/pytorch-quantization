# Huggingface BERT Quantization Example



## Quantization Aware Training (QAT)

Calibrate the pretrained model and finetune with quantization awared:

```
CUDA_VISIBLE_DEVICES=0 python3 run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad.py \
  --max_seq_length 128 \
  --doc_stride 32 \
  --output_dir calib/bert-base-uncased \
  --do_calib \
  --do_eval
```

```
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
```

## Export QAT model to ONNX

To export the QAT model finetuned above:

```
CUDA_VISIBLE_DEVICES=0 python3 run_qa.py \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --tokenizer_name bert-base-uncased \
  --dataset_name squad.py \
  --max_seq_length 128 \
  --doc_stride 32 \
  --do_eval \
  --save_onnx \
  --pretrained_calib finetuned_int8/bert-base-uncased-lr4e-5-bs12-epoch2-GPU1/pytorch_model.bin \
  --output_dir ./
```


## Build the int8 TensorRT engine

```
trtexec --streams=1 \
        --explicitBatch \
        --workspace=16384 \
        --int8 \
        --onnx=bert_qa_calib_128_w8_a8_naive_qat.onnx \
        --minShapes=input_ids:1x128,attention_mask:1x128,token_type_ids:1x128 \
        --optShapes=input_ids:16x128,attention_mask:16x128,token_type_ids:16x128 \
        --maxShapes=input_ids:32x128,attention_mask:32x128,token_type_ids:32x128 \
        --saveEngine=bert_qa_calib_128_w8_a8_naive_qat.trt
```

## Evaluate the int8 TensorRT engine

```
CUDA_VISIBLE_DEVICES=0 python3 eval_trt_qa.py \
  --tokenizer_name bert-base-uncased \
  --dataset_name squad.py \
  --trt_engine_path bert_qa_calib_128_w8_a8_naive_qat.trt \
  --max_seq_length 128 \
  --per_device_eval_batch_size 32 \
  --doc_stride 32 \
  --output_dir ./
```

