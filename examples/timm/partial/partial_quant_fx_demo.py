import argparse
import os
import glob
import json
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel

from timm.models import create_model
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.utils import accuracy, AverageMeter

import sys

sys.path.append("../../../")
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.model_quantizer import TimmModelQuantizer

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

parser.add_argument('--quant_config', type=str, default='./mpq_config.yaml',
                    help='quantzaiton configuration')
parser.add_argument('--calib_weight', type=str, default=None,
                    help='calib weight')
parser.add_argument('--save_onnx', action='store_true',
                    help='export to onnx')


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


def validate_func(val_loader, model, device=None, print_freq=100):
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
    model.eval()
    model.cuda()
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
    # model, qconf = quant_model_init(model, args.quant_config)
    quantizer = TimmModelQuantizer(args.model, model, args.quant_config, calib_weights=args.calib_weight)

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
    if args.save_onnx is True:
        if not os.path.exists(os.path.join(args.output, 'onnx')):
            os.makedirs(os.path.join(args.output, 'onnx'))

    save_preproc_config(args.model, data_config, os.path.join(args.output, 'prep'))
    if args.calib_weight is None:
        quantizer.calibration(train_loader, args.batch_size, save_calib_model=True)

    skip_layers, ori_acc1, partial_acc1, sensitivity = quantizer.partial_quant(val_loader, validate_func, mini_eval_loader=mini_loader)

    if len(skip_layers) > 0:
        suffix = "{}_pptq".format(sensitivity)
        write_results(os.path.join(os.path.join(args.output, 'results'), args.model + "_{}.txt".format(suffix)),
                      args.model,
                      ori_acc1,
                      partial_acc1,
                      skip_layers,
                      partial_acc1)
    else:
        suffix = "ptq"
        write_results(os.path.join(os.path.join(args.output, 'results'), args.model + "_{}.txt".format(suffix)),
                      args.model,
                      ori_acc1,
                      partial_acc1)

    if args.save_onnx:
        onnx_path = os.path.join(os.path.join(args.output, 'onnx'), "{}_{}.onnx".format(args.model, suffix))
        export_onnx(quantizer.model, onnx_path, args, data_config)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
