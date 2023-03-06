# Meituan-Pytorch-Quantization Release Changelog
## 0.1.0-rc 2023-03-06
****
### Bug fix
+ Fix backward of LSQ/Stable-LSQ
+ Add dilation to QuantConv2d/QuantConvTranspose2d

### Feature
+ Support unsigned asym quantization
+ Support skip module


## 0.1.0-beta 2023-02-07
****
### Quantization algorithm  
+ Naive QAT
+ PTQ
+ Partial (top1/mse/snr/cosine)
+ LSQ
+ Stable-LSQ

### Framework support
+ Timm
+ Huggingface
+ MMCls

### Verified model list  
**Timm**
+ resnet18d resnet26 resnet26d resnet26t resnet34 resnet34d resnet50 resnet50d resnetrs50 resnext50_32x4d resnext50d_32x4d tv_resnet50 resnet101 resnet101d resnetrs101 resnetrs152 resnetrs270  
+ mobilenetv2_050 mobilenetv2_100 mobilenetv3_small_075 mobilenetv3_large_100
+ swin_tiny_patch4_window7_224 swin_small_patch4_window7_224 swin_base_patch4_window7_224 swin_large_patch4_window7_224
+ fanet  

**Huggingface**
+ bert-base-uncased  

**MMCls**
+ renset50_b32x8
****


