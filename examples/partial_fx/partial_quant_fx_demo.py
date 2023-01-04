import argparse
import os
import csv
import glob
import json
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict
from contextlib import suppress
from thop import profile
from tqdm import tqdm
from multiprocessing import Process

from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging

import sys

sys.path.append("../../")
from pytorch_quantization.utils.sensitivity import torch_cosine_error, torch_mse_error, torch_snr_error
from pytorch_quantization.quant_utils import module_quant_disable, module_quant_enable, model_quant_disable, \
    model_quant_enable, get_module
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.quant_intf import quant_model_init

parser = argparse.ArgumentParser(description='PyTorch Partial Quantiztion Demo')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--model', '-m', metavar='NAME', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--eval-batch-size', default=20, type=int,
                    metavar='N', help='eval-batch size (default: 20)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--output', type=str, default='./',
                    help='output directory of results')

parser.add_argument('--quant_config', type=str, default='./partial_config.yaml',
                    help='quantzaiton configuration')
parser.add_argument('--num_bits', type=int, default=8,
                    help='quantization bit width')
parser.add_argument('--method', type=str, default='entropy', choices=['max', 'entropy', 'percentile', 'mse'],
                    help='calibration method')
parser.add_argument('--sensitivity_method', type=str, default='mse', choices=['mse', 'cosine', 'top1', 'snr'],
                    help='sensitivity method')
parser.add_argument('--percentile', type=float, default=99.99,
                    help='percentile need to be set when calibration method is percentile')
parser.add_argument('--drop', type=float, default=0.5,
                    help='allowed accuracy drop')
parser.add_argument('--per_layer_drop', type=float, default=0.2,
                    help='threshold of sensitive layers')
parser.add_argument('--calib_num', type=int, default=32,
                    help='calibration batch number')
parser.add_argument('--calib_weight', type=str, default=None,
                    help='calibration weight')
parser.add_argument('--calib_workers', type=int, default=10,
                    help='multi-threads number to collect amax of quantizers')
parser.add_argument('--skip_layers_num', type=int, default=0,
                    help='number of layers to be skipped')
parser.add_argument('--skip_layers', nargs='+', default=[],
                    help='layers to be skipped')
parser.add_argument('--save_partial', action='store_true',
                    help='save partial model pth')
parser.add_argument('--save_onnx', action='store_true',
                    help='export to onnx')


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

    def __call__(self, model_input, k):
        self.model.eval()
        model_quant_disable(self.model)

        handle = self.layer.register_forward_hook(self.data_saver)
        with torch.no_grad():
            _ = self.model(model_input.to(self.device))
        module_ori_output = self.data_saver.output_store.detach()

        module_quant_enable(self.model, k)
        with torch.no_grad():
            _ = self.model(model_input.to(self.device))
        module_quant_output = self.data_saver.output_store.detach()

        handle.remove()
        module_quant_disable(self.model, k)

        return module_ori_output, module_quant_output


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
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    model.cuda()

def _load_calib_amax_mp(quantizer_list, **kwargs):
    for name, quantizer in quantizer_list:
        if quantizer._calibrator is not None:
            if isinstance(quantizer._calibrator, calib.MaxCalibrator):
                quantizer.load_calib_amax()
            else:
                quantizer.load_calib_amax(**kwargs)

def compute_amax_mp(model, **kwargs):
    # collect Quantizer
    quantizer_list = list()
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            quantizer_list.append((name, module))
    print("Total Quantizers {}".format(len(quantizer_list)))
    co_workers = kwargs.pop('calib_workers', 10)
    worker_list = list()
    interval = len(quantizer_list) // co_workers
    for i in range(co_workers):
        start = i
        end = co_workers * interval
        worker = Process(target=_load_calib_amax_mp, args=(quantizer_list[start : end : co_workers],), kwargs=kwargs)
        worker_list.append(worker)
    remain = len(quantizer_list) % co_workers
    if remain > 0:
        start = co_workers * interval
        end = len(quantizer_list)
        worker = Process(target=_load_calib_amax_mp, args=(quantizer_list[start: end],), kwargs=kwargs)
        worker_list.append(worker)

    for worker in worker_list:
        worker.start()
    for worker in worker_list:
        worker.join()
    model.cuda()

def print_amax(model):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            print(F"{name:40}: {module}")

def quant_config(args):
    # initialize input quant policy
    if args.method == 'max':
        method = 'max'
    else:
        method = 'histogram'
    quant_desc_input = QuantDescriptor(num_bits=args.num_bits, calib_method=method)
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantAvgPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantAdaptiveAvgPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
    # weight default quant model is perchannel


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def validate_model(val_loader, model, ptq=False, device=None, print_freq=100):
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg


def quantable_layers_gather(model):
    quantable_layers = []
    for name, module in model.named_modules():
        if name.endswith("_quantizer"):
            layer_name = name.replace("._input_quantizer", "").replace("._weight_quantizer", "")
            if layer_name not in quantable_layers:
                quantable_layers.append(layer_name)
    return quantable_layers

def sensitivity_analyse(model, loader, method, device=None):
    if device is None:
        device = next(model.parameters()).device

    for i, (images, target) in enumerate(loader):
        images = images.to(device)
        break

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
        module_ori_output, module_quant_output = sensitivity(images, k)

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


def partial_quant(sensitivity_list, model, loader, acc1, ptq_acc1, drop, per_layer_drop, device=None):
    disable_layer_list = list()
    partial_acc1 = ptq_acc1
    model_quant_enable(model)

    total_layers = len(sensitivity_list)
    count = 0
    for layer_name, sensitivity in sensitivity_list:
        print("Disable quantization of layer {}".format(layer_name))
        if acc1 - partial_acc1 < drop or count >= int(0.1 * total_layers):
            break
        module_quant_disable(model, layer_name)
        partial = validate_model(loader, model)
        if partial < partial_acc1:
            print("! Acc drop after quantization disable, fallback")
            module_quant_enable(model, layer_name)
            continue
        '''
        if partial - partial_acc1 < per_layer_drop:
            # tiny effect, skip
            module_quant_enable(model, layer_name)
            continue
        else:
            partial_acc1 = partial
            disable_layer_list.append(layer_name)
        '''
        partial_acc1 = partial
        disable_layer_list.append(layer_name)
        count += 1

    return disable_layer_list, partial_acc1


def partial_quant_skip_layers(model, loader, skip_layers_list):
    disable_layer_list = list()
    model_quant_enable(model)
    for layer_name, sensitivity in skip_layers_list:
        module_quant_disable(model, layer_name)
        disable_layer_list.append(layer_name)
    partial_acc1 = validate_model(loader, model)
    return disable_layer_list, partial_acc1


def top1_sensitivity(model, loader, device=None):
    if device is None:
        device = next(model.parameters()).device

    model_quant_disable(model)
    top1_list = list()
    quantable_layers = quantable_layers_gather(model)
    for k in quantable_layers:
        module_quant_enable(model, k)
        top1_acc = validate_model(loader, model)
        top1_list.append((k, top1_acc))
        module_quant_disable(model, k)

    model_quant_enable(model)

    return top1_list


def save_preproc_config(model_name, data_config, path):
    prep_dict = dict()
    prep_dict[model_name] = data_config
    prep_path = os.path.join(path, model_name + "_preproc.json")
    json.dump(prep_dict, open(prep_path, 'w'))


def write_results(filename, arch, acc1, quant_acc1, skip_layers=None, partial_acc1=None):
    with open(filename, mode='w') as cf:
        cf.write("mse" + '\n')
        cf.write(arch + '\n')
        cf.write(str(acc1) + '\n')
        cf.write(str(quant_acc1) + '\n')
        if partial_acc1 is not None:
            cf.write(str(partial_acc1) + '\n')
        if skip_layers is not None:
            for layer in skip_layers:
                cf.write(layer + '\n')


def export_onnx(model, onnx_path, args, data_config):
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    data_shape = (args.batch_size,) + data_config['input_size']
    imgs = torch.randn(data_shape).cuda()
    # ONNX export will fail without jit.trace
    model_traced = torch.jit.trace(model, imgs)
    torch.onnx.export(model_traced, imgs, onnx_path,
                      input_names=['input'],
                      output_names=['output'],
                      verbose=False,
                      opset_version=13,
                      do_constant_folding=True,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX)


def main(args):
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=3,
        exportable=True,
        # fix_stem=True
    )

    data_config = resolve_data_config(vars(args), model=model, use_test_size=True, verbose=True)
    print(data_config)
    model, qconf = quant_model_init(model, args.quant_config)
    model = model.cuda()
    model.eval()

    train_dataset = create_dataset(
        root=args.data,
        name=args.dataset,
        split="train",
        download=args.dataset_download)

    train_loader = create_loader(
        train_dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=True,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=data_config['crop_pct'],
        pin_memory=False,
        tf_preprocessing=False)

    val_dataset = create_dataset(
        root=args.data,
        name=args.dataset,
        split="validation",
        download=args.dataset_download)

    val_loader = create_loader(
        val_dataset,
        input_size=data_config['input_size'],
        batch_size=args.eval_batch_size,
        use_prefetcher=True,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=data_config['crop_pct'],
        pin_memory=False,
        tf_preprocessing=False)

    mini_dataset = create_dataset(
        root=args.data,
        name=args.dataset,
        split="val_mini",
        download=args.dataset_download)

    mini_loader = create_loader(
        mini_dataset,
        input_size=data_config['input_size'],
        batch_size=args.eval_batch_size,
        use_prefetcher=True,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=data_config['crop_pct'],
        pin_memory=False,
        tf_preprocessing=False)

    if not os.path.exists(os.path.join(args.output, 'prep')):
        os.makedirs(os.path.join(args.output, 'prep'))
    if not os.path.exists(os.path.join(args.output, 'results')):
        os.makedirs(os.path.join(args.output, 'results'))
    if args.calib_weight is None:
        if not os.path.exists(os.path.join(args.output, 'calib')):
            os.makedirs(os.path.join(args.output, 'calib'))
    if args.save_partial is True:
        if not os.path.exists(os.path.join(args.output, 'partial')):
            os.makedirs(os.path.join(args.output, 'partial'))
    if args.save_onnx is True:
        if not os.path.exists(os.path.join(args.output, 'onnx')):
            os.makedirs(os.path.join(args.output, 'onnx'))

    save_preproc_config(args.model, data_config, os.path.join(args.output, 'prep'))
    if args.calib_weight is None:
        calib_num = qconf.calib_data_nums // args.batch_size
        with torch.no_grad():
            collect_stats(model, train_loader, calib_num)
            compute_amax(model, method=qconf.a_qscheme.hist_method, percentile=qconf.a_qscheme.percentile)
        torch.save(model.state_dict(), os.path.join(os.path.join(args.output, 'calib'), args.model + '_calib.pth'))
    else:
        model.load_state_dict(torch.load(args.calib_weight))

    model_quant_disable(model)
    ori_acc1 = validate_model(val_loader, model)

    model_quant_enable(model)
    quant_acc1 = validate_model(val_loader, model)

    if ori_acc1 - quant_acc1 > args.drop:
        if qconf.partial_ptq.sensitivity_method == 'top1':
            suffix = "top1_pptq"
            sensitivity_list = top1_sensitivity(model, mini_loader)
            sensitivity_list.sort(key=lambda tup: tup[1], reverse=False)
        else:
            suffix = "{}_pptq".format(qconf.partial_ptq.sensitivity_method)
            sensitivity_list = sensitivity_analyse(model, val_loader, qconf.partial_ptq.sensitivity_method)
            sensitivity_list.sort(key=lambda tup: tup[1], reverse=True)

        print(sensitivity_list)
        if args.skip_layers_num > 0:
            skip_layers_list = sensitivity_list[:args.skip_layers_num]
            skip_layers, partial_acc1 = partial_quant_skip_layers(model, val_loader, skip_layers_list)
        else:
            skip_layers, partial_acc1 = partial_quant(sensitivity_list, model, val_loader, ori_acc1, quant_acc1,
                                                      qconf.partial_ptq.drop,
                                                      qconf.partial_ptq.per_layer_drop)
        write_results(os.path.join(os.path.join(args.output, 'results'), args.model + "_{}.txt".format(suffix)),
                      args.model,
                      ori_acc1,
                      quant_acc1,
                      skip_layers,
                      partial_acc1)
        if args.save_partial:
            torch.save(model.state_dict(),
                       os.path.join(os.path.join(args.output, 'partial'), args.model + '_partial.pth'))
    else:
        suffix = "ptq"
        write_results(os.path.join(os.path.join(args.output, 'results'), args.model + "_{}.txt".format(suffix)),
                      args.model,
                      ori_acc1,
                      quant_acc1)

    if args.save_onnx:
        onnx_path = os.path.join(os.path.join(args.output, 'onnx'), "{}_{}.onnx".format(args.model, suffix))
        export_onnx(model, onnx_path, args, data_config)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
