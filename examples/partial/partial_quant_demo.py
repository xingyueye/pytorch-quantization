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

from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging

import sys
sys.path.append("../../")
from pytorch_quantization.utils.sensitivity import torch_cosine_error, torch_mse_error, torch_snr_error
from pytorch_quantization.quant_utils import module_quant_disable, module_quant_enable, model_quant_disable, model_quant_enable
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules
from pytorch_quantization import tensor_quant

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
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
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--export', action='store_true', default=False,
                    help='export onnx model.')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')

parser.add_argument('--per_channel', dest='per_channel', action='store_true')
parser.add_argument('--num_bits', type=int, default=8)
parser.add_argument('--method', type=str, default='entropy', choices=['max', 'entropy', 'percentile', 'mse'])
parser.add_argument('--sensitivity_method', type=str, default='mse', choices=['mse', 'cosine', 'top1', 'snr'])
parser.add_argument('--percentile', type=float, default=99.99)
parser.add_argument('--drop', type=float, default=0.5)
parser.add_argument('--per_layer_drop', type=float, default=0.2)
parser.add_argument('--calib_num', type=int, default=4)
parser.add_argument('--calib_weight', type=str, default=None)

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
    def __init__(self, model, layer, device:torch.device):
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

def sensitivity_analyse(model, loader, method, device=None):
    if device is None:
        device = next(model.parameters()).device

    for i, (images, target) in enumerate(loader):
        images = images.to(device)
        break
    
    sensitivity_list = list()
    model_quant_disable(model)

    for k, m in model.named_modules():
        if isinstance(m, quant_nn.QuantConv2d) or \
        isinstance(m, quant_nn.QuantConvTranspose2d) or \
        isinstance(m, quant_nn.QuantLinear):
            module_quant_enable(model, k)

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
        if acc1 - partial_acc1 < drop or count >= int(0.1 * total_layers):
            break
        module_quant_disable(model, layer_name)
        partial = validate_model(loader, model)
        if partial - partial_acc1 < per_layer_drop:
            # tiny effect, skip
            module_quant_enable(model, layer_name)
            continue
        else:
            partial_acc1 = partial
            disable_layer_list.append(layer_name)
        count += 1

    return disable_layer_list, partial_acc1

def top1_sensitivity(model, loader, device=None):
    if device is None:
        device = next(model.parameters()).device

    model_quant_disable(model)
    top1_list = list()

    for k, m in model.named_modules():
        if isinstance(m, quant_nn.QuantConv2d) or \
        isinstance(m, quant_nn.QuantConvTranspose2d):
            module_quant_enable(model, k)
            top1_acc = validate_model(loader, model)
            top1_list.append((k, top1_acc))
            module_quant_disable(model, k)

    model_quant_enable(model)
    
    return top1_list
            
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

def main():
    args = parser.parse_args()
    quant_modules.initialize()
    quant_config(args)

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
        batch_size=args.batch_size,
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
        batch_size=args.batch_size,
        use_prefetcher=True,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=data_config['crop_pct'],
        pin_memory=False,
        tf_preprocessing=False)

    if args.calib_weight is None:
        with torch.no_grad():
            collect_stats(model, train_loader, args.calib_num)
            compute_amax(model, method=args.method, percentile=args.percentile)
        torch.save(model.state_dict(), args.model + '_calib.pth')
    else:
        model.load_state_dict(torch.load(args.calib_weight))

    model_quant_disable(model)
    ori_acc1 = validate_model(val_loader, model)

    model_quant_enable(model)
    quant_acc1 = validate_model(val_loader, model)

    if ori_acc1 - quant_acc1 > args.drop:
        if args.sensitivity_method == 'top1':
            top1_list = top1_sensitivity(model, mini_loader)
            top1_list.sort(key=lambda tup: tup[1], reverse=False)
            print(top1_list)
            skip_layers, partial_acc1 = partial_quant(top1_list, model, val_loader, ori_acc1, quant_acc1, args.drop, args.per_layer_drop)
            write_results(args.model + "_top1_quant.txt", args.model, ori_acc1, quant_acc1, skip_layers, partial_acc1)
        else:
            mse_list = sensitivity_analyse(model, val_loader, args.sensitivity_method)
            mse_list.sort(key=lambda tup: tup[1], reverse=True)
            print(mse_list)
            skip_layers, partial_acc1 = partial_quant(mse_list, model, val_loader, ori_acc1, quant_acc1, args.drop, args.per_layer_drop)
            write_results(args.model + "_mse_quant.txt", args.model, ori_acc1, quant_acc1, skip_layers, partial_acc1)
    else:
        write_results(args.model + "_quant.txt", args.model, ori_acc1, quant_acc1)

if __name__ == '__main__':
    main()