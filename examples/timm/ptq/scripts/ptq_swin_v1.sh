# calib
IMAGENET='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hdpmlpserving/liqingyuan02/dataset/ILSVRC2012'
# Swin Tiny
CUDA_VISIBLE_DEVICES=0  python3 -m torch.distributed.launch --nproc_per_node 1 \
                                    --master_port 12346 train.py  \
                                    $IMAGENET \
                                    -b 4 \
                                    --model swin_tiny_patch4_window7_224 \
                                    --quantizer FTSWIN \
                                    --quant \
                                    --calib \
                                    --quant_config ptq/configs/mpq_config_swin_v1.yaml \
                                    --pretrained \
                                    --val-split val

# Swin Small
CUDA_VISIBLE_DEVICES=0  python3 -m torch.distributed.launch --nproc_per_node 1 \
                                    --master_port 12346 train.py  \
                                    $IMAGENET \
                                    -b 4 \
                                    --model swin_small_patch4_window7_224 \
                                    --quantizer FTSWIN \
                                    --quant \
                                    --calib \
                                    --quant_config ptq/configs/mpq_config_swin_v1.yaml \
                                    --pretrained \
                                    --val-split val

# Swin Base
CUDA_VISIBLE_DEVICES=0  python3 -m torch.distributed.launch --nproc_per_node 1 \
                                    --master_port 12346 train.py  \
                                    $IMAGENET \
                                    -b 4 \
                                    --model swin_base_patch4_window7_224 \
                                    --quantizer FTSWIN \
                                    --quant \
                                    --calib \
                                    --quant_config ptq/configs/mpq_config_swin_v1.yaml \
                                    --pretrained \
                                    --val-split val

# Swin Large
CUDA_VISIBLE_DEVICES=0  python3 -m torch.distributed.launch --nproc_per_node 1 \
                                    --master_port 12346 train.py  \
                                    $IMAGENET \
                                    -b 4 \
                                    --model swin_large_patch4_window7_224 \
                                    --quantizer FTSWIN \
                                    --quant \
                                    --calib \
                                    --quant_config ptq/configs/mpq_config_swin_v1.yaml \
                                    --pretrained \
                                    --val-split val