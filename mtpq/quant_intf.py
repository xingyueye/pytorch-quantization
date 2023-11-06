from tqdm import tqdm
from easydict import EasyDict

from mtpq import nn as quant_nn
# from mtpq import calib
from mtpq.utils.config_utils import get_quant_desc

from mtpq.quant_utils import set_module
from mtpq.quant_fx import insert_qdq_nodes_via_subgraph_match
from mtpq.nn.modules.converter import *
# from mtpq.utils.layer_reconstruction import layer_rebuild, save_inp_oup_data
from mtpq.utils.layer_reconstruction import layer_rebuild, cache_layer_blobs
from mtpq.utils.calib_utils import *

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
_ADAROUND_FUSE_PATTERN_MAP ={
                    "Conv2d" : "Conv2dBNFuseInPlace"
}


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
    if config.w_qscheme.quantizer_type == "adaround":
        custom_module_map = _ADAROUND_FUSE_PATTERN_MAP
        
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
            if hasattr(m,"act"):## FIXME This is just for quick application in SNPE, just suit for EfficientNet 
                act_layer = m.act
                set_module(model, k, act_layer)
            else:
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

    model = custom_ops_replace(model, config, custom_module_map)
    model = quant_ops_replace(model, config, quant_module_map)
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


# def bert_quant_insert_qdq(model, config):
#     quant_desc = get_quant_desc(config)
#     return bert_insert_qdq_nodes_via_subgraph_match(model, quant_desc.input_desc)
#
# def bert_quant_model_init(model, config, calib_weights=''):
#     # config = parse_config(config_file)
#     quant_module_map = get_quant_module_map(config.quant_layers_type)
#
#     model = quant_ops_replace(model, config, quant_module_map)
#     model = bert_quant_insert_qdq(model, config)
#     if calib_weights:
#         state_dict = torch.load(calib_weights, map_location='cpu')
#         model.load_state_dict(state_dict['model'].state_dict())
#     return model

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

    calib_num = min(config.calib_data_nums, len(data_loader.dataset)) if hasattr(data_loader, 'dataset') else config.calib_data_nums
    calib_batch = calib_num // batch_size
    with torch.no_grad():
        collect_stats(model, data_loader, calib_batch, predict)
        compute_amax(model, method=config.a_qscheme.hist_method, percentile=config.a_qscheme.percentile,strict=False)

def quant_model_calib_bert(model, data_loader, config, batch_size, predict):
    model.eval()
    model.cuda()
    # It is a bit slow since we collect histograms on CPU

    calib_num = min(config.calib_data_nums, len(data_loader.dataset))
    calib_batch = calib_num // batch_size
    with torch.no_grad():
        collect_stats(model, data_loader, calib_batch, predict)
        compute_amax(model, method=config.a_qscheme.hist_method, percentile=config.a_qscheme.percentile)

def quant_model_calib_adaround(model, data_loader, config, batch_size, predict, batch_first=True, qdrop_prob=0.5, use_bc=False):
    model.eval()
    model.cuda()
    calib_num = min(config.calib_data_nums, len(data_loader.dataset)) if hasattr(data_loader, 'dataset') else config.calib_data_nums
    calib_batch = calib_num // batch_size
    print(f'adaround: before layer reconstruction: batch first: {batch_first}, \
          calib num: {calib_num}({config.calib_data_nums}, {len(data_loader.dataset)}), \
              batches num: {calib_batch}, qdrop prob: {qdrop_prob}, usebc: {use_bc}')
    #activation caibration
    # with torch.no_grad():
    #     collect_stats(model, data_loader, calib_batch, predict)
    #     compute_amax(model, method=config.a_qscheme.hist_method, percentile=config.a_qscheme.percentile)
    
    print('adaround: layer reconstruction begin')
    #weight caibration: layer reconstruction
    # cali_data, _ = get_train_samples(data_loader, num_samples=1024)
    # recon_model(model,cali_data, batch_size,model)
    reconstruct_model(model, data_loader, batch_size, model, predict, calib_batch, batch_first=batch_first, qdrop_prob=qdrop_prob, use_bc=use_bc)
    print('adaround: layer reconstruction done')

    #activation caibration
    with torch.no_grad():
        collect_stats(model, data_loader, calib_batch, predict)
        compute_amax(model, method=config.a_qscheme.hist_method, percentile=config.a_qscheme.percentile)

def reconstruct_model(model, data_loader, batch_size, ori_model, predict_func, calib_batch, 
                      batch_first=True, name_prefix='', qdrop_prob=0.5, use_bc=False):
    """
    Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
    """
    for name, module in model.named_children():
        if isinstance(module,(quant_nn.Conv2d, quant_nn.Conv2dBNFuseInPlace, quant_nn.Linear, quant_nn.LinearFT)):
            # compute_amax(model)
            print(f'adaround: Reconstruction for layer {name_prefix}.{name}')
            # cached_inps, cached_outs = save_inp_oup_data(ori_model, module, cali_data, batch_size, keep_gpu=True)
            # layer_rebuild(module, cached_inps, cached_outs, batch_size=batch_size, warmup=0.1)
            
            cached_inputs, cached_outputs = cache_layer_blobs(ori_model, module, predict_func, data_loader, batch_size, calib_batch, 
                                                              batch_first=batch_first, qdrop_prob=qdrop_prob)
            layer_rebuild(module, cached_inputs, cached_outputs, 
                          batch_size=batch_size, warmup=0.1, iters=30000, qdrop_prob=qdrop_prob, use_bc=use_bc)
        else:
            reconstruct_model(module, data_loader, batch_size, ori_model, predict_func, calib_batch, 
                              batch_first=batch_first, name_prefix=f'{name_prefix}.{name}',qdrop_prob=qdrop_prob, use_bc=use_bc)

# def recon_model(model,cali_data, batch_size,ori_model):
#     """
#     Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
#     """
#     for name, module in model.named_children():
#         if isinstance(module,(quant_nn.Conv2d, quant_nn.Conv2dBNFuseInPlace, quant_nn.Linear, quant_nn.LinearFT)):
#             # compute_amax(model)
#             print('Reconstruction for layer {}'.format(name))
#             cached_inps, cached_outs = save_inp_oup_data(ori_model, module, cali_data, batch_size, keep_gpu=True)
#             layer_rebuild(module, cached_inps, cached_outs,batch_size=batch_size,warmup=0.1)
#         else:
#             recon_model(module, cali_data, batch_size,ori_model)

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


def get_train_samples(train_loader, num_samples):
    train_data, target = [], []
    for batch in train_loader:
        train_data.append(batch[0])
        target.append(batch[1])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples], torch.cat(target, dim=0)[:num_samples]
