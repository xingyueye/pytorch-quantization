# Timm Quantization Example
Timm, which is the state of art pretrained pytorch-image-models repository. We will show how to quantize the Timm's models by using Meituan Pytorch Quantization Tookit.  
## Install
```
pip3 install timm == 0.6.11
```

## Post Training Quantization
Take tv_resnet50 as an example, run following command to perform PTQ.
```shell
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
                                    --val-split val \
                                    --export \
                                    --export-batch-size 4
```

## Quantization Aware Training
The accuracy of modern CNN models will heavily degrade after PTQ, such as EfficientNet. For this situation, we can use QAT to restore accuracy.  
At first, calibrate the model to get quantization hyper-parameters.
```shell
CUDA_VISIBLE_DEVICES=0  python3 -m torch.distributed.launch --nproc_per_node 1 \
                                    --master_port 12346 train.py  \
                                    /mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/ILSVRC2012/ \
                                    -b 32 \
                                    --model efficientnet_b0 \
                                    --quant \
                                    --calib \
                                    --quant_config qat/configs/mpq_config_efficientnet.yaml \
                                    --pretrained \
                                    --val-split val \
                                    --input-size 3 224 224
```
Then, finetune the quantized model to recover the accuracy. 
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node 8 \
                                    --master_port 12346 train.py  \
                                    /mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/ILSVRC2012/ \
                                    --model efficientnet_b0 \
                                    --quant \
                                    --quant_config qat/configs/mpq_config_efficientnet.yaml \
                                    --pretrained_calib efficientnet_b0_calib_128_w8a8_naive.pt \
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
                                    --input-size 3 224 224 \
                                    --output efficientnet_b0_no_jitter_naive \
                                    --distillation-type soft \
                                    --distillation-scale 2000 \
                                    --teacher-model efficientnet_b0 \
                                    --teacher-path  https://s3plus.sankuai.com/automl-model-zoo/huyiming03/efficientnet/efficientnet_b0_ra-3dd342df.pth
```

## Partial Quantization
QAT will introduce training procedure, sometimes it's not trival to implement. Usually networks only have a few layers which are sensitive to quantization, so just make those layers fallback to float computation to avoid accuracy drop. This is called Partial Quantization.  
At first, calibrate the model to get quantization hyper-parameters.
```shell
  CUDA_VISIBLE_DEVICES=0  python3 -m torch.distributed.launch --nproc_per_node 1 \
                                      --master_port 12346 train.py  \
                                      $IMAGENET \
                                      -b 4 \
                                      --model resnext50d_32x4d \
                                      --quantizer Timm \
                                      --quant \
                                      --calib \
                                      --quant_config partial/configs/mpq_config.yaml \
                                      --pretrained \
                                      --val-split val \
                                      --validation-batch-size 50
```
Do partial quantization. It automatically analyse the quantization sensitivity of each layer and skip most sensitivity layer util the accuracy drop meets the threshold setted by users. We provide 4 methods to analyse sensitivity, they are 'mse', 'cosine', 'top1', 'snr'. Among them, 'mse，'snr' and 'cosine' are fast sensitivity analysis methods, which only require about 128 images
```shell
  CUDA_VISIBLE_DEVICES=0  python3 -m torch.distributed.launch --nproc_per_node 1 \
                                      --master_port 12346 train.py  \
                                      $IMAGENET \
                                      -b 4 \
                                      --model resnext50d_32x4d \
                                      --quantizer Timm \
                                      --quant \
                                      --partial \
                                      --partial_dump \
                                      --quant_config partial/configs/mpq_config.yaml \
                                      --pretrained_calib resnext50d_32x4d_calib_128_w8a8_naive.pt \
                                      --val-split val \
                                      --validation-batch-size 50 \
                                      --export \
                                      --export-batch-size 1
```