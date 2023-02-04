# efficientnet_b0
# calib
CUDA_VISIBLE_DEVICES=0  python3 -m torch.distributed.launch --nproc_per_node 1 \
                                    --master_port 12346 train.py  \
                                    /mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/ILSVRC2012/ \
                                    -b 32 \
                                    --model mobilenetv3_large_100 \
                                    --quant \
                                    --calib \
                                    --quant_config configs/mpq_config_mobilenet.yaml \
                                    --pretrained \
                                    --val-split val \
# qat
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node 8 \
                                    --master_port 12346 train.py  \
                                    /mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/ILSVRC2012/ \
                                    --model mobilenetv3_large_100 \
                                    --quant \
                                    --quant_config configs/mpq_config_mobilenet.yaml \
                                    --pretrained_calib mobilenetv3_large_100_calib_128_w8a8_naive.pt \
                                    -b 512 \
                                    --sched step \
                                    --epochs 50 \
                                    --decay-epochs 2.4 \
                                    --decay-rate .973 \
                                    --momentum 0.9 \
                                    --opt rmsproptf \
                                    --opt-eps .001 -j 7 \
                                    --weight-decay 1e-5 \
                                    --color-jitter 0.0 \
                                    --lr .00064 \
                                    --lr-noise 0.42 0.9 \
                                    --drop 0.2 \
                                    --aa rand-m9-mstd0.5 \
                                    --remode pixel \
                                    --reprob 0.2 \
                                    --amp \
                                    --val-split val \
                                    --output mobilenetv3_large_100_no_jitter_naive \
                                    --distillation-type soft \
                                    --distillation-scale 2000 \
                                    --teacher-model mobilenetv3_large_100 \
                                    --teacher-path https://s3plus.sankuai.com/automl-model-zoo/huyiming03/mbv3/mobilenetv3_large_100_ra-f55367f5.pth