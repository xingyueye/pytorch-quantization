# Meituan PyTorch Quantization

<img src='assets/mtpq_small.png' width='350' height='75' />

## Overview

Meituan PyTorch Quantization (MTPQ) is an Meituan initiative for accelerating industrial application for quantization in vision, NLP, and audio etc. MTPQ significantly refactors the software architecture of `pytorch-quantization`, where it takes a top-down approach to automatically parse user-defined models and inserts quantization nodes. MTPQ ships with PTQ, Partial PTQ, QAT and a myriad of up-to-date quantization algorithms. The final quantized model is made exportable to onnx in TensorRT's appetite to leverage NVIDIAs high performance GPUs.

## Benchmark

### Quantization Performance on Classical Models 

MTPQ Quantization performance is tested with a list of classical models that are mostly used in practice, such as ResNet, MobileNet, EfficientNet, Swin, Bert and YOLOv6. The average accuracy loss is 0.41% compared to their FP16 counterparts.

![Classical Models Accuracy](assets/sop_acc.png)

The following chart uses a batch size of 4, tested on NVIDIA Tesla T4 with TensorRT 8.4.

![QPS Classical Models](assets/sop_perf.png)

All models enjoy a substantial QPS boost (58% on average) after quantization.

![Classical Models Relative Boost](assets/sop_rel_boost.png)

### Quantization Performance on Timm Models

MTPQ well supports timm models where the majority of them has a tolerable accuracy loss if quantized only with PTQ or partial PTQ.

![Timm Quantization Performance](assets/Timm_PTQ_perf.png)

## Install

### Prerequisites

- gcc 5.4+
- python >=3.6
- torch>=1.9

#### From Binaries (Recommended)

```bash
pip3 install mtpq
```

#### From Source (For Develeopment)

```bash
git clone ssh://git@git.sankuai.com/mtmlp/meituan-pytorch-quantization.git
cd meituan-pytorch-quantization
```

Install PyTorch and prerequisites
```bash
pip3 install -r requirements.txt
# for CUDA 10.2 users
pip3 install torch>=1.9.1
# for CUDA 11.1 users
pip3 install torch>=1.9.1+cu111
```

Build and install
```bash
# Python version >= 3.6, GCC version >= 5.4 required
python3 setup.py install
```

## Quickstart

### PTQ on ResNet50 
```bash
cd examples/timm
sh examples/timm/ptq/scripts/ptq_resnet50.sh
```

### Partial PTQ on ResNet series

```bash
cd examples/timm/
sh partial/scripts/partial_resnet.sh
```
### QAT on EfficientNet
```bash
cd examples/timm/
sh qat/scripts/quant_efficientnet_b0_skd.sh
```

More examples see [examples/timm/README.md](examples/timm/README.md) and [examples/huggingface/question_answering/README.md](examples/huggingface/question_answering/README.md).

### MTPQ Architecture

For beginners, one needs to play with Model Quantizer only. For developers, it is made easy enough to inherit the base classes for new features, e.g., Module Converter to support more operations, Pattern Matcher to track new graph patterns, and Tensor Quantizer to adopt novel quantization algorithms etc.

![MTPQ Architecture](assets/architecture.png)

### Auto-Parsing and Q/DQ Node Insertion

The Module Converter and Pattern Matcher are designed to parse the model and convert it into its quantization couterapart.

![Auto-Parse Demo](assets/auto_parse_demo.gif)


## Support Matrix

### Training framework
- OpenMMLab
- Timm
- Huggineface

### Bitwidth
- 8bit (exportable)
- 4bit

### Deployment
- TensorRT 7.2+
- TensorRT 8+

## Resources

* Pytorch Quantization Toolkit [userguide](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html)
* Quantization Basics [whitepaper](https://arxiv.org/abs/2004.09602)

