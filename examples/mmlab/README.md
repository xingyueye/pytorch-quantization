# MMLab Quantization Example
OpenMMLab is an open source algorithm system suitable for academic research and industrial applications, covering many research topics in computer vision, including MMClassficaiton, MMDetection, MMSegmentation, etc.. Here, we will show how to quantize a MMClassfication model by using Meituan Pytorch Quantization Tookit.  
## Install
```
Install open source mmcls following https://github.com/open-mmlab/mmclassification#installation. 
Or MT-CVZoo version: ssh://git@git.sankuai.com/vision/infra-mt-cvzoo-classification.git.

# It is recommended to place the corresponding **_quant.py script in the tools/ folder to use.
```

## Post Training Quantization
Take tv_resnet50 as an example, run following command to perform PTQ.
```shell
CHECKPOINT='https://s3plus.sankuai.com/automl-model-zoo/resnet50_8xb32_in1k_20210831-ea4938fc.pth'
cd ./mmclassification/
CUDA_VISIBLE_DEVICES=0  python mmcls_quant.py \
                          ./examples/mmlab/configs/resnet50_b32x8_imagenet_quant.py \
                          CHECKPOINT \
                          --ptq \
                          --metrics accuracy \
                          --quant_config ./examples/mmlab/configs/mpq_config.yaml
```

## Quantization Aware Training
The accuracy of modern CNN models will heavily degrade after PTQ, such as EfficientNet. For this situation, we can use QAT to restore accuracy.  
At first, calibrate the model to get quantization hyper-parameters.
```shell
CUDA_VISIBLE_DEVICES=0  python mmcls_quant.py \
                          ./examples/mmlab/configs/resnet50_b32x8_imagenet_quant.py \
                          CHECKPOINT \
                          --ptq \
                          --metrics accuracy \
                          --quant_config ./examples/mmlab/configs/mpq_config.yaml
```
Then, finetune the quantized model to recover the accuracy. 
```shell
CUDA_VISIBLE_DEVICES=0  python mmcls_quant.py \
                          ./examples/mmlab/configs/resnet50_b32x8_imagenet_quant.py \
                          CHECKPOINT \
                          --qat \
                          --metrics accuracy \
                          --quant_config ./examples/mmlab/configs/mpq_config.yaml
```

## Partial Quantization
QAT will introduce training procedure, sometimes it's not trival to implement. Usually networks only have a few layers which are sensitive to quantization, so just make those layers fallback to float computation to avoid accuracy drop. This is called Partial Quantization.  
```shell
CUDA_VISIBLE_DEVICES=0  python mmcls_quant.py \
                          ./examples/mmlab/configs/resnet50_b32x8_imagenet_quant.py \
                          CHECKPOINT \
                          --ptq \
                          --partial \
                          --metrics accuracy \
                          --quant_config ./examples/mmlab/configs/mpq_config.yaml
```