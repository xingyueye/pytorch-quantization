import torch
from tqdm import tqdm
from mtpq import nn as quant_nn

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
    
def save_hist(model, save_to=None):
    hist_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                hist_dict[name] = {
                    'hist': module._calibrator._calib_hist if hasattr(module._calibrator, '_calib_hist') else None,
                    'bin_edges': module._calibrator._calib_bin_edges if hasattr(module._calibrator, '_calib_bin_edges') else None,
                    'amax': module.amax
                }
    if save_to is not None and type(save_to) is str:
        torch.save(hist_dict, save_to)
    return hist_dict

def print_quant_summary(model):
    """Print summary of all quantizer modules in the model."""

    counters = {'quantizers': 0, 'enabled_quantizers': 0,
                'weights': 0, 'quant_weights': 0, 'sparse_weights': 0,
                'params': 0, 'sparse_params': 0}
    for name, mod in model.named_modules():
        if isinstance(mod, quantization.nn.TensorQuantizer):
            print(f'{name:80} {mod}')
            counters['quantizers'] += 1
            if not mod._disabled:
                counters['enabled_quantizers'] += 1

        for pname, param in mod.named_parameters():
            if '.' in pname:
                continue
            counters['params'] += param.numel()
            # fullname = f'{name}.{pname}'
            # print(f'{fullname:80} {param.numel():12}')
            weight_quantizer = getattr(mod, '_weight_quantizer', None)
            if pname == 'weight':
                counters['weights'] += param.numel()
                if weight_quantizer is not None and not weight_quantizer._disabled:
                    counters['quant_weights'] += param.numel()
                counters['sparse_weights'] += param.eq(0).sum().item()
            counters['sparse_params'] += param.eq(0).sum().item()

    def print_fraction(a, b, counters, desc):
        va = counters[a]
        vb = counters[b]
        pct = va/vb * 100 if vb != 0 else float('NaN')
        print(f'{counters[a]:12}/{vb:12} ({pct:6.2f}%) {desc}')
    print_fraction('enabled_quantizers', 'quantizers', counters, 'TensorQuantizers enabled')
    print_fraction('quant_weights', 'weights', counters, 'Quantized weights')
    print_fraction('sparse_weights', 'weights', counters, 'Zero weights')
    print_fraction('weights', 'params', counters, 'Weight parameters')
    print('\n\n')