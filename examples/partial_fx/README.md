# Partial Quantization Batch
We use torch.fx tool to apply batch partial quantization to timm model zoo.

```python
python3 partial_quant_fx_batch.py --timm_zoo timm_zoo_urls.txt
                                  --data  ILSVRC2012_PATH
                                  --split val
                                  --batch-size 4
                                  --calib_num 32
                                  --method entropy
                                  --sensitivity_method mse
                                  --drop 0.5
                                  --save_partial
                                  --export_onnx
                                  --output timm_pptq_fx
```

```python
python3 partial_quant_fx_demo.py --data ILSVRC2012_PATH
                                 --split val
                                 --model tv_resnet50
                                 --pretrained
                                 --save_onnx
                                 --sensitivity_method mse
                                 --drop 0.5
                                 --batch-size 4 
                                 --calib_num 32
                                 --output tv_resnet50
```