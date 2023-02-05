# TensorRT Engine Evaluation


## Timm
```python
python3 eval_timm_trt_model.py --engine_path ./timm_trt_engines \
                               --eval-dir /workshop/liqingyuan02/dataset/ILSVRC2012/ \
                               --batch-size 1 \
                               --io-type fp16
```