# calib
IMAGENET='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hdpmlpserving/liqingyuan02/dataset/ILSVRC2012'
MODEL_LIST='swinv2_tiny_window8_256 swinv2_small_window8_256 swinv2_base_window8_256 swinv2_large_window12to16_192to256_22kft1k'
for MODEL in $MODEL_LIST;
do
  CUDA_VISIBLE_DEVICES=0  python3 -m torch.distributed.launch --nproc_per_node 1 \
                                      --master_port 12346 train.py  \
                                      $IMAGENET \
                                      -b 4 \
                                      --model ${MODEL} \
                                      --quantizer FTSWIN \
                                      --quant \
                                      --calib \
                                      --quant_config ptq/configs/mtpq_config_swin_v2.yaml \
                                      --pretrained \
                                      --val-split val
done
