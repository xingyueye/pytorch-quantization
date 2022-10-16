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
import numpy as np

sys.path.append("../../")
from pytorch_quantization.utils.sensitivity import torch_cosine_error, torch_mse_error, torch_snr_error
from pytorch_quantization.quant_utils import module_quant_disable, module_quant_enable, model_quant_disable, \
    model_quant_enable, get_module
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules
from pytorch_quantization import tensor_quant

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
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--output', type=str, default='./',
                    help='output directory of results')

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


def get_layer_in_out(model, input, device: torch.device):
    model.eval()
    model_quant_disable(model)

    handle_list = []
    layer_in_out = dict()
    for name, module in model.named_modules():
        if isinstance(module) == quant_nn.QuantConv2d or \
            isinstance(module) == quant_nn.QuantLinear or \
            isinstance(module) == quant_nn.QuantConvTranspose2d:
            layer_in_out[name] = DataSaverHook(store_input=True, store_output=True)
            handle_list.append(module.register_forward_hook(layer_in_out[name]))

    with torch.no_grad():
        _ = model(input.to(device))

    for handle in handle_list:
        handle.remove()

    return layer_in_out


def scale_search(model, layer_in_out, device=None):
    model.eval()
    model.to(device)
    model_quant_disable(model)

    for name, in_out in layer_in_out.items():
        module_quant_enable(model, name)
        layer = get_module(model, name)
        amax_orig = layer._input_quantizer._amax.detach().item()
        scale_orig =  amax_orig / 127.0
        scale_start, scale_end = 0.5 * scale_orig, 2.0 * scale_orig
        scale_interval = (scale_end - scale_start) / 100
        similarity_list = []
        # search best scale
        for i in range(100):
            amax_tmp = (scale_start + i*scale_interval) * 127.0
            layer._input_quantizer._amax.fill_(amax_tmp)
            with torch.no_grad():
                output = layer(in_out[name].input_store.detach().to(device))
            similarity = torch_cosine_error(output.detach().cpu(), in_out[name].output_store.detach().cpu())
            similarity_list.append(similarity)

        similarity_np = np.array(similarity_list)
        scale_best = similarity_np.argsort()[::-1][0]
        amax_best = scale_best * 127.0
        print("Modify {} amax from {:.6f} to {:.6f}".format(name, amax_orig, amax_best))
        layer._input_quantizer._amax.fill_(amax_best)
        module_quant_disable(model, name)


def easy_quant(model, images, device):
    layer_in_out = get_layer_in_out(model, images, device)
    scale_search(model, layer_in_out, device)


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


def main(args):
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

    images_list = []
    for i, (images, target) in enumerate(train_loader):
        images_list.append(images)
        if i >= args.calib_num:
            break
    images = torch.concat(images_list, dim=0)
    easy_quant(model, images, device=torch.device('cuda'))

    model_quant_enable(model)
    easy_acc1 = validate_model(val_loader, model)

    print("ori acc1 ={:.4f} quant_acc1 = {:.4f} easy_acc1 = {:.4f}".format(ori_acc1, quant_acc1, easy_acc1))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
