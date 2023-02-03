# tv_renset50
# calib
IMAGENET='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hdpmlpserving/liqingyuan02/dataset/ILSVRC2012'
CUDA_VISIBLE_DEVICES=0  python3 -m torch.distributed.launch --nproc_per_node 1 \
                                    --master_port 12346 train.py  \
                                    $IMAGENET \
                                    -b 32 \
                                    --model tv_resnet50 \
                                    --quantizer Timm \
                                    --quant \
                                    --calib \
                                    --quant_config ptq/configs/mpq_config.yaml \
                                    --pretrained \
                                    --val-split val