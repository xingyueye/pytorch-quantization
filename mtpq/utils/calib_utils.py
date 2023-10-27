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