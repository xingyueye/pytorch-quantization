import yaml
from easydict import EasyDict
from tqdm import tqdm

# from mtpq import calib
from mtpq.tensor_quant import QuantDescriptor
from mtpq.quant_utils import set_module
from mtpq.quant_fx import insert_qdq_nodes_via_subgraph_match
from mtpq.nn.modules.converter import *


_DEFAULT_QUANT_MAP = {"Conv1d": quant_nn.QuantConv1d,
                      "Conv2d": quant_nn.QuantConv2d,
                      "Conv3d": quant_nn.QuantConv3d,
                      "ConvTranspose1d": quant_nn.QuantConvTranspose1d,
                      "ConvTranspose2d": quant_nn.QuantConvTranspose2d,
                      "ConvTranspose3d": quant_nn.QuantConvTranspose3d,
                      "Linear": quant_nn.QuantLinear,
                      "LSTM": quant_nn.QuantLSTM,
                      "LSTMCell": quant_nn.QuantLSTMCell,
                      "MaxPool1d": quant_nn.QuantMaxPool1d,
                      "MaxPool2d": quant_nn.QuantMaxPool2d,
                      "MaxPool3d": quant_nn.QuantMaxPool3d,
                      "AvgPool1d": quant_nn.QuantAvgPool1d,
                      "AvgPool2d": quant_nn.QuantAvgPool2d,
                      "AvgPool3d": quant_nn.QuantAvgPool3d,
                      "AdaptiveAvgPool1d": quant_nn.QuantAdaptiveAvgPool1d,
                      "AdaptiveAvgPool2d": quant_nn.QuantAdaptiveAvgPool2d,
                      "AdaptiveAvgPool3d": quant_nn.QuantAdaptiveAvgPool3d}

_DEFAULT_CNN_CUSTOM_MAP = {"Hardswish": quant_nn.HardswishReplace}
_DEFAULT_BERT_CUSTOM_MAP = {}
_DEFAULT_FTSWIN_CUSTOM_MAP = {"Linear": quant_nn.QuantLinearFT}

_CUSTOM_MAP = {"CNN": _DEFAULT_CNN_CUSTOM_MAP,
               "BERT": _DEFAULT_BERT_CUSTOM_MAP,
               "FTSWIN": _DEFAULT_FTSWIN_CUSTOM_MAP}

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
                      "QuantMaxPool1d": nn.MaxPool1d,
                      "QuantMaxPool2d": nn.MaxPool2d,
                      "QuantMaxPool3d": nn.MaxPool3d,
                      "QuantAvgPool1d": nn.AvgPool1d,
                      "QuantAvgPool2d": nn.AvgPool2d,
                      "QuantAvgPool3d": nn.AvgPool3d,
                      "QuantAdaptiveAvgPool1d": nn.AdaptiveAvgPool1d,
                      "QuantAdaptiveAvgPool2d": nn.AdaptiveAvgPool2d,
                      "QuantAdaptiveAvgPool3d": nn.AdaptiveAvgPool3d}

_DEFAULT_FUSE_PATTERN_MAP = {
                      "Conv2d": "Conv2dBNFuse",}

def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return EasyDict(config)


def enable_calibration(model):
    """Enable calibration of all *_input_quantizer modules in model."""
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()


def disable_calibration(model):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def _internal_predict(model, data_loader, num_batches):
    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        # image = image.float()/255.0
        model(image.cuda())
        if i >= num_batches:
            break

def collect_stats(model, data_loader, num_batches, predict):
    """Feed data to the network and collect statistic"""
    enable_calibration(model)

    if predict is None:
        _internal_predict(model, data_loader, num_batches)
    else:
        predict(model, data_loader, num_batches)

    disable_calibration(model)


def compute_amax(model, **kwargs):
    # Load Calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                # #MinMaxCalib
                # if isinstance(module._calibrator, calib.MaxCalibrator):
                #     module.load_calib_amax()
                # else:
                # #HistogramCalib
                module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    model.cuda()


def get_quant_desc(config):
    quant_desc = {
        "input_desc": QuantDescriptor(num_bits=config.a_qscheme.bit, calib_method=config.a_qscheme.calib_method,
                                                 quantizer_type=config.a_qscheme.quantizer_type,
                                                 unsigned=config.a_qscheme.unsigned if hasattr(config.a_qscheme, 'unsigned') else False),
        "conv_weight_desc": QuantDescriptor(num_bits=config.w_qscheme.bit, axis=(0) if config.w_qscheme.per_channel is True else None,
                                            calib_method=config.w_qscheme.calib_method,
                                            quantizer_type=config.w_qscheme.quantizer_type),
        "deconv_weight_desc": QuantDescriptor(num_bits=config.w_qscheme.bit, axis=(1) if config.w_qscheme.per_channel is True else None,
                                              calib_method=config.w_qscheme.calib_method,
                                              quantizer_type=config.w_qscheme.quantizer_type),
        "output_desc": QuantDescriptor(num_bits=config.a_qscheme.bit, calib_method=config.a_qscheme.calib_method,
                                      quantizer_type=config.a_qscheme.quantizer_type),
    }
    return EasyDict(quant_desc)

def skip_layers_check(k, config):
    for skip_module in config.skip_modules:
        if skip_module in k:
            return True
    return True if k in config.skip_layers else False

def quant_ops_replace(model, config, quant_module_map=_DEFAULT_QUANT_MAP):
    quant_desc = get_quant_desc(config)

    for k, m in model.named_modules():
        if skip_layers_check(k, config):
            print("Skip Layer {}".format(k))
            continue
        module_type = m.__class__.__name__
        if module_type in quant_module_map:
            converter = globals()['{}Converter'.format(module_type)](quant_desc)
            set_module(model, k, converter.convert(m))

    return model

def custom_ops_replace(model, config, custom_module_map=_CUSTOM_MAP['CNN']):
    quant_desc = get_quant_desc(config)

    for k, m in model.named_modules():
        if skip_layers_check(k, config):
            print("Skip Layer {}".format(k))
            continue
        module_type = m.__class__.__name__
        if module_type in custom_module_map.keys():
            converter = globals()['{}CustomConverter'.format(module_type)](quant_desc)
            set_module(model, k, converter.convert(m))


    return model

def fuse_pattern_replace(model, config, custom_module_map=_DEFAULT_FUSE_PATTERN_MAP):
    quant_desc = get_quant_desc(config)

    for k, m in model.named_modules():
        if skip_layers_check(k, config):
            print("Skip Layer {}".format(k))
            continue
        module_type = m.__class__.__name__
        if module_type in custom_module_map.keys():
            converter = globals()['{}Converter'.format(custom_module_map[module_type])](quant_desc)
            set_module(model, k, converter.convert(m))

    for k, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and hasattr(m, 'following_bn') and m.following_bn:
            set_module(model, k, nn.Identity())

    return model

def get_quant_module_map(quant_layers_type=[]):
    if len(quant_layers_type) == 0:
        return _DEFAULT_QUANT_MAP
    else:
        quant_map = {k:_DEFAULT_QUANT_MAP[k] for k in quant_layers_type if k in _DEFAULT_QUANT_MAP}
        return quant_map

def get_custom_module_map(type_str):
    return _CUSTOM_MAP[type_str]

def find_conv_bn_patterns(module):
    named_children = module.named_children()
    for name, child in named_children:
        if isinstance(child, nn.Conv2d):
            try:
                next_name, next_child = next(named_children)
            except:
                next_child = None
            if isinstance(next_child, nn.BatchNorm2d):
                setattr(child, 'is_bn_following', True)
                setattr(next_child, 'following_bn', True)
                follow_bn = {'bn_name': next_name,
                             'bn_module': next_child}
                setattr(child, 'follow_bn', follow_bn)
            else:
                setattr(child, 'is_bn_following', False)
        else:
            find_conv_bn_patterns(child)

def quant_insert_qdq(model, config, type_str='CNN', do_trace=True):
    quant_desc = get_quant_desc(config)
    return insert_qdq_nodes_via_subgraph_match(model, quant_desc.input_desc, type_str, do_trace)

def quant_model_init(model, config, calib_weights='', type_str='CNN', do_trace=True):
    # config = parse_config(config_file)
    custom_module_map = get_custom_module_map(type_str)
    quant_module_map = get_quant_module_map(config.quant_layers_type)

    # Custom Ops replace to avoid some unsupported ops
    model = custom_ops_replace(model, config, custom_module_map)

    # Pattern modules Fuse
    find_conv_bn_patterns(model)
    model = fuse_pattern_replace(model, config)
    # Quantization Modules replace
    model = quant_ops_replace(model, config, quant_module_map)

    # Pattern match and insert quantization node
    if not hasattr(config, 'use_fx') or config.use_fx:
        model = quant_insert_qdq(model, config, type_str, do_trace)
    if calib_weights:
        state_dict = torch.load(calib_weights, map_location='cpu')
        if 'model' in state_dict.keys():
            model.load_state_dict(state_dict['model'].state_dict())
        else:
            model.load_state_dict(state_dict)
    return model

def quant_model_init_mmlab(model, config, calib_weights='', type_str='CNN', do_trace=True):
    # config = parse_config(config_file)
    custom_module_map = get_custom_module_map(type_str)
    quant_module_map = get_quant_module_map(config.quant_layers_type)
    model = custom_ops_replace(model, config, custom_module_map)
    model = quant_ops_replace(model, config, quant_module_map)
    model_cp = model.module if hasattr(model, 'module') else model 
    model_cp.backbone = quant_insert_qdq(model_cp.backbone, config, type_str, do_trace)
    # model_cp.neck = quant_insert_qdq(model_cp.neck, config, type_str, do_trace)
    return model

def save_calib_model(model_name, model):
    # Save calibrated checkpoint
    print('Saving calibrated model to {}... '.format(model_name))
    torch.save({'model': model}, model_name)

def quant_model_calib(model, data_loader, config, batch_size, predict):
    model.eval()
    model.cuda()
    # It is a bit slow since we collect histograms on CPU

    calib_num = min(config.calib_data_nums, len(data_loader.dataset))
    calib_batch = calib_num // batch_size
    with torch.no_grad():
        collect_stats(model, data_loader, calib_batch, predict)
        compute_amax(model, method=config.a_qscheme.hist_method, percentile=config.a_qscheme.percentile)

def quant_model_calib_timm(model, data_loader, config, batch_size, predict):
    model.eval()
    model.cuda()
    # It is a bit slow since we collect histograms on CPU

    calib_num = min(config.calib_data_nums, len(data_loader.dataset))
    calib_batch = calib_num // batch_size
    with torch.no_grad():
        collect_stats(model, data_loader, calib_batch, predict)
        compute_amax(model, method=config.a_qscheme.hist_method, percentile=config.a_qscheme.percentile)

def quant_model_calib_bert(model, data_loader, config, batch_size, predict):
    model.eval()
    model.cuda()
    # It is a bit slow since we collect histograms on CPU

    calib_num = min(config.calib_data_nums, len(data_loader.dataset))
    calib_batch = calib_num // batch_size
    with torch.no_grad():
        collect_stats(model, data_loader, calib_batch, predict)
        compute_amax(model, method=config.a_qscheme.hist_method, percentile=config.a_qscheme.percentile)

def quant_model_export(model, onnx_path, data_shape, dynamic_axes=None):
    model.eval()
    model.cuda()
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    imgs = torch.randn(data_shape).cuda()
    # model_traced = torch.jit.trace(model, imgs)
    torch.onnx.export(model, imgs, onnx_path,
                      input_names=['input'],
                      output_names=['output'],
                      verbose=False,
                      opset_version=13,
                      do_constant_folding=True,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                      dynamic_axes=dynamic_axes)
