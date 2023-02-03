# calib
IMAGENET='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hdpmlpserving/liqingyuan02/dataset/ILSVRC2012'
MODEL_LIST='tv_resnet50'
for MODEL in $MODEL_LIST;
do
  CUDA_VISIBLE_DEVICES=0  python3 -m torch.distributed.launch --nproc_per_node 1 \
                                      --master_port 12346 train.py  \
                                      $IMAGENET \
                                      -b 32 \
                                      --model ${MODEL} \
                                      --quantizer Timm \
                                      --quant \
                                      --calib \
                                      --quant_config partial/configs/mpq_config.yaml \
                                      --pretrained \
                                      --val-split val

  CUDA_VISIBLE_DEVICES=0  python3 -m torch.distributed.launch --nproc_per_node 1 \
                                      --master_port 12346 train.py  \
                                      $IMAGENET \
                                      -b 32 \
                                      --model ${MODEL} \
                                      --quantizer Timm \
                                      --quant \
                                      --partial \
                                      --partial_dump \
                                      --quant_config partial/configs/mpq_config.yaml \
                                      --pretrained_calib ${MODEL}_calib_128_w8a8_naive.pt \
                                      --val-split val \
                                      --export \
                                      --export-batch-size 4

done