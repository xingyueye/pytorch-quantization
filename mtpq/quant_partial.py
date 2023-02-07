import torch
from mtpq.utils.sensitivity import torch_cosine_error, torch_mse_error, torch_snr_error
from mtpq.quant_utils import module_quant_disable, module_quant_enable, model_quant_disable, \
    model_quant_enable, get_module
from mtpq import nn as quant_nn

class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """

    def __init__(self, store_input=False, store_output=False):
        self.store_input = store_input
        self.store_output = store_output

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch


class GetLayerSensitivity:
    def __init__(self, model, layer, device: torch.device):
        self.model = model
        self.layer = layer
        self.device = device
        self.data_saver = DataSaverHook(store_input=True, store_output=True)

    def __call__(self, forward_func, k):
        self.model.eval()
        model_quant_disable(self.model)

        handle = self.layer.register_forward_hook(self.data_saver)
        forward_func()
        module_ori_output = self.data_saver.output_store.detach()

        module_quant_enable(self.model, k)
        forward_func()
        module_quant_output = self.data_saver.output_store.detach()

        handle.remove()
        module_quant_disable(self.model, k)

        return module_ori_output, module_quant_output


def quantable_layers_gather(model):
    quantable_layers = []
    for name, module in model.named_modules():
        if name.endswith("_quantizer"):
            layer_name = name.replace("._input_quantizer", "").replace("._weight_quantizer", "")
            if layer_name not in quantable_layers:
                quantable_layers.append(layer_name)
    return quantable_layers


def fast_sensitivity(model, loader, method, device=None, forward_func=None):
    if device is None:
        device = next(model.parameters()).device

    sensitivity_list = list()
    model_quant_disable(model)
    quantable_layers = quantable_layers_gather(model)

    for k in quantable_layers:
        module_quant_enable(model, k)
        m = get_module(model, k)
        # If it's nn.Module, we use its child module
        children_modules = list(m.children())
        if len(children_modules) == 1 and isinstance(children_modules[0], quant_nn.TensorQuantizer):
            m = get_module(model, k + '._input_quantizer')
        sensitivity = GetLayerSensitivity(model, m, device)
        module_ori_output, module_quant_output = sensitivity(forward_func, k)

        if method == 'mse':
            mse = torch_mse_error(module_ori_output, module_quant_output)
            sensitivity_list.append((k, mse))
        elif method == 'cosine':
            cosine = torch_cosine_error(module_ori_output, module_quant_output)
            sensitivity_list.append((k, cosine))
        elif method == 'snr':
            snr = torch_snr_error(module_ori_output, module_quant_output)
            sensitivity_list.append((k, snr))
        else:
            raise ValueError(f'Unsupported sensitivity method.')

        module_quant_disable(model, k)

    model_quant_enable(model)
    return sensitivity_list


def top1_sensitivity(model, loader, eval_func, device=None):
    if device is None:
        device = next(model.parameters()).device

    model_quant_disable(model)
    top1_list = list()
    quantable_layers = quantable_layers_gather(model)
    for k in quantable_layers:
        module_quant_enable(model, k)
        top1_acc = eval_func(loader, model)
        top1_list.append((k, top1_acc))
        module_quant_disable(model, k)

    model_quant_enable(model)

    return top1_list


def do_partial_quant(sensitivity_list, model, loader, eval_func, org_acc1, ptq_acc1, drop):
    disable_layer_list = list()
    partial_acc1 = ptq_acc1
    model_quant_enable(model)

    total_layers = len(sensitivity_list)
    count = 0
    for layer_name, sensitivity in sensitivity_list:
        if org_acc1 - partial_acc1 < drop or count >= int(0.1 * total_layers):
            break
        print("Disable quantization of layer {}".format(layer_name))
        module_quant_disable(model, layer_name)
        partial = eval_func(loader, model)
        if partial < partial_acc1:
            print("! Acc drop after quantization disable, fallback")
            module_quant_enable(model, layer_name)
            continue
        partial_acc1 = partial
        disable_layer_list.append(layer_name)
        count += 1

    return disable_layer_list, partial_acc1