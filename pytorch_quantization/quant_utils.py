import yaml
from easydict import EasyDict
from tqdm import tqdm

import torch
import torch.nn as nn

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules


_DEFAULT_QUANT_MAP = {"Conv1d": quant_nn.QuantConv1d,
                      "Conv2d": quant_nn.QuantConv2d,
                      "Conv3d": quant_nn.QuantConv3d,
                      "ConvTranspose1d": quant_nn.QuantConvTranspose1d,
                      "ConvTranspose2d": quant_nn.QuantConvTranspose2d,
                      "ConvTranspose3d": quant_nn.QuantConvTranspose3d,
                      "Linear": quant_nn.QuantLinear,
                      "LSTM": quant_nn.QuantLSTM,
                      "LSTMCell": quant_nn.QuantLSTMCell,
                      "AvgPool1d": quant_nn.QuantAvgPool1d,
                      "AvgPool2d": quant_nn.QuantAvgPool2d,
                      "AvgPool3d": quant_nn.QuantAvgPool3d,
                      "AdaptiveAvgPool1d": quant_nn.QuantAdaptiveAvgPool1d,
                      "AdaptiveAvgPool2d": quant_nn.QuantAdaptiveAvgPool2d,
                      "AdaptiveAvgPool3d": quant_nn.QuantAdaptiveAvgPool3d}

_DEFAULT_DE_QUANT_MAP = {
                      "QuantConv1d": nn.Conv1d,
                      "QuantConv2d": nn.Conv2d,
                      "QuantConv3d": nn.Conv3d,
                      "QuantConvTranspose1d": nn.ConvTranspose1d,
                      "QuantConvTranspose2d": nn.ConvTranspose2d,
                      "QuantConvTranspose3d": nn.ConvTranspose3d,
                      "QuantLinear": nn.Linear,
                      "QuantLSTM": nn.LSTM,
                      "QuantLSTMCell": nn.LSTMCell,
                      "QuantAvgPool1d": nn.AvgPool1d,
                      "QuantAvgPool2d": nn.AvgPool2d,
                      "QuantAvgPool3d": nn.AvgPool3d,
                      "QuantAdaptiveAvgPool1d": nn.AdaptiveAvgPool1d,
                      "QuantAdaptiveAvgPool2d": nn.AdaptiveAvgPool2d,
                      "QuantAdaptiveAvgPool3d": nn.AdaptiveAvgPool3d}

def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return EasyDict(config)

def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        # image = image.float()/255.0
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    # Load Calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                #MinMaxCalib
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                #HistogramCalib
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    model.cuda()

def lsq_init(model, quant_module_map=[]):
    for name, module in model.named_modules():
        if module.__class__.__name__ in _DEFAULT_DE_QUANT_MAP:
            if hasattr(module, "_input_quantizer"):
                module._input_quantizer.init_learn_scale()
            if hasattr(module, "_weight_quantizer"):
                module._weight_quantizer.init_learn_scale(inputs=module.weight.data)

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

def get_quant_desc(config):
    quant_desc = {
        "input_desc": QuantDescriptor(num_bits=config.a_qscheme.bit, calib_method=config.a_qscheme.calib_method,
                                                 _learn_scale_type=config.a_qscheme.quantizer_type),
        "conv_weight_desc": QuantDescriptor(num_bits=config.w_qscheme.bit, axis=(0), calib_method=config.w_qscheme.calib_method,
                                                  _learn_scale_type=config.w_qscheme.quantizer_type),
        "deconv_weight_desc": QuantDescriptor(num_bits=config.w_qscheme.bit, axis=(1), calib_method=config.w_qscheme.calib_method,
                                                  _learn_scale_type=config.w_qscheme.quantizer_type),                                                                     
    }
    return EasyDict(quant_desc)

def quant_ops_replace(model, config, quant_module_map=_DEFAULT_QUANT_MAP):
    quant_desc = get_quant_desc(config)

    for k, m in model.named_modules():
        if not m.__class__.__name__ in quant_module_map:
            continue    
        if isinstance(m, nn.Conv2d):
            in_channels = m.in_channels
            out_channels = m.out_channels
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            groups = m.groups
            quant_conv = quant_nn.QuantConv2d(in_channels,
                                              out_channels,
                                              kernel_size,
                                              stride,
                                              padding,
                                              groups=groups,
                                              quant_desc_input = quant_desc.input_desc,
                                              quant_desc_weight = quant_desc.conv_weight_desc)
            quant_conv.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                quant_conv.bias.data.copy_(m.bias.detach())
            else:
                quant_conv.bias = None
            set_module(model, k, quant_conv)
        elif isinstance(m, nn.ConvTranspose2d):
            in_channels = m.in_channels
            out_channels = m.out_channels
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            groups = m.groups
            quant_convtrans = quant_nn.QuantConvTranspose2d(in_channels,
                                                       out_channels,
                                                       kernel_size,
                                                       stride,
                                                       padding,
                                                       groups=groups,
                                                       quant_desc_input = quant_desc.input_desc,
                                                       quant_desc_weight = quant_desc.deconv_weight_desc)
            quant_convtrans.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                quant_convtrans.bias.data.copy_(m.bias.detach())
            else:
                quant_convtrans.bias = None
            set_module(model, k, quant_convtrans)
        elif isinstance(m, nn.MaxPool2d):
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            dilation = m.dilation
            ceil_mode = m.ceil_mode
            quant_maxpool2d = quant_nn.QuantMaxPool2d(kernel_size,
                                                      stride,
                                                      padding,
                                                      dilation,
                                                      ceil_mode,
                                                      quant_desc_input = quant_desc.input_desc)
            set_module(model, k, quant_maxpool2d)
        elif isinstance(m, nn.Linear):
            quant_linear = quant_nn.QuantLinear(
                                            m.in_features,
                                            m.out_features,
                                            quant_desc_input = quant_desc.input_desc,
                                            quant_desc_weight = quant_desc.deconv_weight_desc)
            quant_linear.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                quant_linear.bias.data.copy_(m.bias.detach())
            else:
                quant_linear.bias = None
            # quant_linear.to(m.weight.device)
            set_module(model, k, quant_linear)
        else:
            # module can not be quantized, continue
            continue

def get_quant_module_map(quant_layers_type=[]):
    if len(quant_layers_type) == 0:
        return _DEFAULT_QUANT_MAP
    else:
        quant_map = {k:_DEFAULT_QUANT_MAP[k] for k in quant_layers_type if k in _DEFAULT_QUANT_MAP}
        return quant_map

def quant_model_init(model, config_file, calib_weights=''):
    config = parse_config(config_file)
    quant_module_map = get_quant_module_map(config.quant_layers_type)

    quant_ops_replace(model, config, quant_module_map)
    if calib_weights:
        state_dict = torch.load(calib_weights, map_location='cpu')
        model.load_state_dict(state_dict['model'].state_dict())
        # model.load_state_dict(state_dict)
        lsq_init(model, quant_module_map)  
    return model, config

def save_calib_model(model, config):
    # Save calibrated checkpoint
    # import os
    # output_model_path = os.path.join(config.calib_output_path, '{}_lsq_calib.pt'.
    #                                     format(args.model))
    output_model_path = "calib_{}.pt".format(config.calib_data_nums)                                    
    print('Saving calibrated model to {}... '.format(output_model_path))
    # if not os.path.exists(args.calib_output_path):
    #     os.mkdir(args.calib_output_path)
    torch.save({'model': model}, output_model_path)

def quant_model_calib(model, data_loader, config):
    model.eval()
    model.cuda()
    # It is a bit slow since we collect histograms on CPU

    calib_num = min(config.calib_data_nums, len(data_loader.dataset))
    calib_batch = calib_num // data_loader.batch_size
    with torch.no_grad():
        collect_stats(model, data_loader, calib_batch)
        compute_amax(model, method=config.a_qscheme.hist_method, percentile=config.a_qscheme.percentile)
    save_calib_model(model, config)

def quant_model_export(model, onnx_path, data_shape):
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    # data_shape = (args.batch_size,) + data_config['input_size']
    imgs = torch.randn(data_shape).cuda()
    torch.onnx.export(model, imgs, onnx_path,
                        input_names=['input'],
                        output_names=['output'],
                        verbose=False,
                        opset_version=13,
                        operator_export_type=torch.onnx.OperatorExportTypes.ONNX)

    get_remove_qdq_onnx_and_cache(onnx_path)
