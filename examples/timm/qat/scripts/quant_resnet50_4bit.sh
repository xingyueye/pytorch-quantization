# tv_renset50
# calib
# IMAGENET='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hdpmlpserving/liqingyuan02/dataset/ILSVRC2012'
IMAGENET='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/common/ILSVRC2012'
CUDA_VISIBLE_DEVICES=2  python3 -m torch.distributed.launch --nproc_per_node 1 \
                                    --master_port 12346 train.py  \
                                    $IMAGENET \
                                    -b 32 \
                                    --model tv_resnet50 \
                                    --quant \
                                    --calib \
                                    --quant_config qat/configs/mtpq_config_r50_8bit_stable_lsq.yaml \
                                    --pretrained \
                                    --val-split val
# qat
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node 8 \
                                    --master_port 12346 train.py  \
                                    $IMAGENET \
                                    --model tv_resnet50 \
                                    --quant \
                                    --quant_config qat/configs/mtpq_config_r50_4bit_stable_lsq.yaml \
                                    --pretrained_calib tv_resnet50_calib_128_w4a4_stable_lsq.pt \
                                    -b 64 \
                                    --sched cosine \
                                    --epochs 30 \
                                    --warmup-epochs 1 \
                                    --momentum 0.9 \
                                    --opt sgd \
                                    --opt-eps .001 \
                                    --weight-decay 1e-4 \
                                    --color-jitter 0.0 \
                                    --lr .006 \
                                    --val-split val \
                                    --output tv_resnet50_cosine_lr0.006_epoch30_no_jitter_stable_lsq