# calib
IMAGENET='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hdpmlpserving/liqingyuan02/dataset/ILSVRC2012'
MODEL_LIST='resnet18d resnet26 resnet26d resnet26t resnet34 resnet34d'
for MODEL in $MODEL_LIST;
do
  CUDA_VISIBLE_DEVICES=0  python3 -m torch.distributed.launch --nproc_per_node 1 \
                                      --master_port 12346 train.py  \
                                      $IMAGENET \
                                      -b 4 \
                                      --model ${MODEL} \
                                      --quantizer Timm \
                                      --quant \
                                      --calib \
                                      --quant_config partial/configs/mpq_config.yaml \
                                      --pretrained \
                                      --val-split val \
                                      --validation-batch-size 100

  CUDA_VISIBLE_DEVICES=0  python3 -m torch.distributed.launch --nproc_per_node 1 \
                                      --master_port 12346 train.py  \
                                      $IMAGENET \
                                      -b 4 \
                                      --model ${MODEL} \
                                      --quantizer Timm \
                                      --quant \
                                      --partial \
                                      --partial_dump \
                                      --quant_config partial/configs/mpq_config.yaml \
                                      --pretrained_calib ${MODEL}_calib_128_w8a8_naive.pt \
                                      --val-split val \
                                      --validation-batch-size 100 \
                                      --export \
                                      --export-batch-size 1

done
MODEL_LIST='resnet50 resnet50d resnetrs50 resnext50_32x4d resnext50d_32x4d tv_resnet50'
for MODEL in $MODEL_LIST;
do
  CUDA_VISIBLE_DEVICES=0  python3 -m torch.distributed.launch --nproc_per_node 1 \
                                      --master_port 12346 train.py  \
                                      $IMAGENET \
                                      -b 4 \
                                      --model ${MODEL} \
                                      --quantizer Timm \
                                      --quant \
                                      --calib \
                                      --quant_config partial/configs/mpq_config.yaml \
                                      --pretrained \
                                      --val-split val \
                                      --validation-batch-size 50

  CUDA_VISIBLE_DEVICES=0  python3 -m torch.distributed.launch --nproc_per_node 1 \
                                      --master_port 12346 train.py  \
                                      $IMAGENET \
                                      -b 4 \
                                      --model ${MODEL} \
                                      --quantizer Timm \
                                      --quant \
                                      --partial \
                                      --partial_dump \
                                      --quant_config partial/configs/mpq_config.yaml \
                                      --pretrained_calib ${MODEL}_calib_128_w8a8_naive.pt \
                                      --val-split val \
                                      --validation-batch-size 50 \
                                      --export \
                                      --export-batch-size 1 \
                                      --export-dynamic-axes "{'input': {0: 'batch'}, 'output': {0: 'batch'}}"

done

MODEL_LIST='resnet101 resnet101d resnetrs101'
for MODEL in $MODEL_LIST;
do
  CUDA_VISIBLE_DEVICES=0  python3 -m torch.distributed.launch --nproc_per_node 1 \
                                      --master_port 12346 train.py  \
                                      $IMAGENET \
                                      -b 4 \
                                      --model ${MODEL} \
                                      --quantizer Timm \
                                      --quant \
                                      --calib \
                                      --quant_config partial/configs/mpq_config.yaml \
                                      --pretrained \
                                      --val-split val \
                                      --validation-batch-size 25

  CUDA_VISIBLE_DEVICES=0  python3 -m torch.distributed.launch --nproc_per_node 1 \
                                      --master_port 12346 train.py  \
                                      $IMAGENET \
                                      -b 4 \
                                      --model ${MODEL} \
                                      --quantizer Timm \
                                      --quant \
                                      --partial \
                                      --partial_dump \
                                      --quant_config partial/configs/mpq_config.yaml \
                                      --pretrained_calib ${MODEL}_calib_128_w8a8_naive.pt \
                                      --val-split val \
                                      --validation-batch-size 25 \
                                      --export \
                                      --export-batch-size 1 \
                                      --export-dynamic-axes "{'input': {0: 'batch'}, 'output': {0: 'batch'}}"

done

MODEL_LIST='resnetrs152 resnetrs270'
for MODEL in $MODEL_LIST;
do
  CUDA_VISIBLE_DEVICES=0  python3 -m torch.distributed.launch --nproc_per_node 1 \
                                      --master_port 12346 train.py  \
                                      $IMAGENET \
                                      -b 4 \
                                      --model ${MODEL} \
                                      --quantizer Timm \
                                      --quant \
                                      --calib \
                                      --quant_config partial/configs/mpq_config.yaml \
                                      --pretrained \
                                      --val-split val \
                                      --validation-batch-size 10

  CUDA_VISIBLE_DEVICES=0  python3 -m torch.distributed.launch --nproc_per_node 1 \
                                      --master_port 12346 train.py  \
                                      $IMAGENET \
                                      -b 4 \
                                      --model ${MODEL} \
                                      --quantizer Timm \
                                      --quant \
                                      --partial \
                                      --partial_dump \
                                      --quant_config partial/configs/mpq_config.yaml \
                                      --pretrained_calib ${MODEL}_calib_128_w8a8_naive.pt \
                                      --val-split val \
                                      --validation-batch-size 10 \
                                      --export \
                                      --export-batch-size 1 \
                                      --export-dynamic-axes "{'input': {0: 'batch'}, 'output': {0: 'batch'}}"

done