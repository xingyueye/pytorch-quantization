DATE=`date "+%Y%m%d"`

python3 partial_quant_fx_demo.py --data ../../../../dataset/ILSVRC2012/  --split val --quant_config ./partial_config.yaml \
                                --model resnet18d --pretrained --batch-size 4 --eval-batch-size 100  \
                                --save_onnx  --output ./timm_pptq_resnet_${DATE}

python3 partial_quant_fx_demo.py --data ../../../../dataset/ILSVRC2012/  --split val --quant_config ./partial_config.yaml \
                                --model resnet26 --pretrained --batch-size 4 --eval-batch-size 100  \
                                --save_onnx  --output ./timm_pptq_resnet_${DATE}

python3 partial_quant_fx_demo.py --data ../../../../dataset/ILSVRC2012/  --split val --quant_config ./partial_config.yaml \
                                --model resnet26d --pretrained --batch-size 4 --eval-batch-size 100  \
                                --save_onnx  --output ./timm_pptq_resnet_${DATE}

python3 partial_quant_fx_demo.py --data ../../../../dataset/ILSVRC2012/  --split val --quant_config ./partial_config.yaml \
                                --model resnet26t --pretrained --batch-size 4 --eval-batch-size 100  \
                                --save_onnx  --output ./timm_pptq_resnet_${DATE}

python3 partial_quant_fx_demo.py --data ../../../../dataset/ILSVRC2012/  --split val --quant_config ./partial_config.yaml \
                                --model resnet34 --pretrained --batch-size 4 --eval-batch-size 100  \
                                --save_onnx  --output ./timm_pptq_resnet_${DATE}

python3 partial_quant_fx_demo.py --data ../../../../dataset/ILSVRC2012/  --split val --quant_config ./partial_config.yaml \
                                --model resnet34d --pretrained --batch-size 4 --eval-batch-size 100  \
                                --save_onnx  --output ./timm_pptq_resnet_${DATE}

python3 partial_quant_fx_demo.py --data ../../../../dataset/ILSVRC2012/  --split val --quant_config ./partial_config.yaml \
                                --model resnet50 --pretrained --batch-size 4 --eval-batch-size 50  \
                                --save_onnx  --output ./timm_pptq_resnet_${DATE}

python3 partial_quant_fx_demo.py --data ../../../../dataset/ILSVRC2012/  --split val --quant_config ./partial_config.yaml \
                                --model resnet50d --pretrained --batch-size 4 --eval-batch-size 50  \
                                --save_onnx  --output ./timm_pptq_resnet_${DATE}

python3 partial_quant_fx_demo.py --data ../../../../dataset/ILSVRC2012/  --split val --quant_config ./partial_config.yaml \
                                --model resnet101 --pretrained --batch-size 4 --eval-batch-size 25  \
                                --save_onnx  --output ./timm_pptq_resnet_${DATE}

python3 partial_quant_fx_demo.py --data ../../../../dataset/ILSVRC2012/  --split val --quant_config ./partial_config.yaml \
                                --model resnet101d --pretrained --batch-size 4 --eval-batch-size 25  \
                                --save_onnx  --output ./timm_pptq_resnet_${DATE}

python3 partial_quant_fx_demo.py --data ../../../../dataset/ILSVRC2012/  --split val --quant_config ./partial_config.yaml \
                                --model resnetrs50 --pretrained --batch-size 4 --eval-batch-size 50  \
                                --save_onnx  --output ./timm_pptq_resnet_${DATE}

python3 partial_quant_fx_demo.py --data ../../../../dataset/ILSVRC2012/  --split val --quant_config ./partial_config.yaml \
                                --model resnetrs101 --pretrained --batch-size 4 --eval-batch-size 25  \
                                --save_onnx  --output ./timm_pptq_resnet_${DATE}

python3 partial_quant_fx_demo.py --data ../../../../dataset/ILSVRC2012/  --split val --quant_config ./partial_config.yaml \
                                --model resnetrs152 --pretrained --batch-size 4 --eval-batch-size 10  \
                                --save_onnx  --output ./timm_pptq_resnet_${DATE}

python3 partial_quant_fx_demo.py --data ../../../../dataset/ILSVRC2012/  --split val --quant_config ./partial_config.yaml \
                                --model resnetrs270 --pretrained --batch-size 4 --eval-batch-size 10  \
                                --save_onnx  --output ./timm_pptq_resnet_${DATE}

python3 partial_quant_fx_demo.py --data ../../../../dataset/ILSVRC2012/  --split val --quant_config ./partial_config.yaml \
                                --model resnext50_32x4d --pretrained --batch-size 4 --eval-batch-size 50  \
                                --save_onnx  --output ./timm_pptq_resnet_${DATE}

python3 partial_quant_fx_demo.py --data ../../../../dataset/ILSVRC2012/  --split val --quant_config ./partial_config.yaml \
                                --model resnext50d_32x4d --pretrained --batch-size 4 --eval-batch-size 50  \
                                --save_onnx  --output ./timm_pptq_resnet_${DATE}