from pytorch_quantization import nn as quant_nn

def set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

def get_module(model, submodule_key):
    sub_tokens = submodule_key.split('.')
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod

def module_quant_disable(model, k):
    cur_module = get_module(model, k)
    if hasattr(cur_module, '_input_quantizer'):
        cur_module._input_quantizer.disable()
    if hasattr(cur_module, '_weight_quantizer'):
        cur_module._weight_quantizer.disable()

def module_quant_enable(model, k):
    cur_module = get_module(model, k)
    if hasattr(cur_module, '_input_quantizer'):
        cur_module._input_quantizer.enable()
    if hasattr(cur_module, '_weight_quantizer'):
        cur_module._weight_quantizer.enable()

def model_quant_disable(model):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.disable()

def model_quant_enable(model):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.enable()

