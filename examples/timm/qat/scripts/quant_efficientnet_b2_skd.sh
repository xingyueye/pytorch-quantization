IMAGENET='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/ILSVRC2012/'
# efficientnet_b2
# calib
CUDA_VISIBLE_DEVICES=0  python3 -m torch.distributed.launch --nproc_per_node 1 \
                                    --master_port 12347 train.py  \
                                    ${IMAGENET} \
                                    -b 32 \
                                    --model efficientnet_b2 \
                                    --quant \
                                    --calib \
                                    --quant_config qat/configs/mtpq_config_efficientnet.yaml \
                                    --pretrained \
                                    --val-split val \
                                    --input-size 3 260 260
# qat
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node 8 \
                                    --master_port 12346 train.py  \
                                    ${IMAGENET} \
                                    --model efficientnet_b2 \
                                    --quant \
                                    --quant_config qat/configs/mtpq_config_efficientnet.yaml \
                                    --pretrained_calib efficientnet_b2_calib_128_w8a8_naive.pt \
                                    -b 128 \
                                    --sched step \
                                    --epochs 18 \
                                    --decay-epochs 2.4 \
                                    --decay-rate .97 \
                                    --momentum 0.9 \
                                    --opt rmsproptf \
                                    --opt-eps .001 -j 8 \
                                    --weight-decay 1e-5 \
                                    --color-jitter 0.0 \
                                    --lr .00048 \
                                    --drop 0.2 \
                                    --aa rand-m9-mstd0.5 \
                                    --remode pixel \
                                    --reprob 0.2 \
                                    --amp \
                                    --val-split val \
                                    --input-size 3 260 260 \
                                    --output efficientnet_b2_no_jitter_naive \
                                    --distillation-type soft \
                                    --distillation-scale 2000 \
                                    --teacher-model efficientnet_b2 \
                                    --teacher-path  https://s3plus.sankuai.com/automl-model-zoo/huyiming03/efficientnet/efficientnet_b2_ra-bcdf34b7.pth