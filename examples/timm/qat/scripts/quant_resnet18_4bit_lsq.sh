# IMAGENET='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hdpmlpserving/liqingyuan02/dataset/ILSVRC2012'
IMAGENET='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/ILSVRC2012/'

# resnet18
# calib
CUDA_VISIBLE_DEVICES=0  python3 -m torch.distributed.launch --nproc_per_node 1 \
                                    --master_port 12346 train.py  \
                                    $IMAGENET \
                                    -b 32 \
                                    --model resnet18 \
                                    --quant \
                                    --calib \
                                    --quant_config qat/configs/mtpq_config_r18_4bit_lsq.yaml \
                                    --initial-checkpoint r18_w32a32.pth \
                                    --val-split val
# qat
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node 8 \
                                    --master_port 12346 train.py  \
                                    $IMAGENET \
                                    --model resnet18 \
                                    --quant \
                                    --quant_config qat/configs/mtpq_config_r18_4bit_lsq.yaml \
                                    --pretrained_calib resnet18_calib_128_w4a4_lsq.pt \
                                    -b 64 \
                                    --sched cosine \
                                    --epochs 100 \
                                    --warmup-epochs 1 \
                                    --momentum 0.9 \
                                    --opt sgd \
                                    --opt-eps .001 \
                                    --weight-decay 1e-4 \
                                    --color-jitter 0.0 \
                                    --lr .003 \
                                    --val-split val \
                                    --output resnet18_cosine_lr0.003_epoch100_no_jitter_lsq