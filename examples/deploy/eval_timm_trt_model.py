import argparse
import os
import glob

from timm.models import create_model
from timm.data import resolve_data_config
from TensorRT.eval import Evaluator
from TensorRT.data import get_val_loader

import sys

sys.path.append("../../")


parser = argparse.ArgumentParser(description='PyTorch Partial Quantiztion Demo')
parser.add_argument('--engine_path', metavar='DIR',
                    help='path to engines')
parser.add_argument('--model', '-m', metavar='NAME', default='tv_resnet50',
                    help='model architecture (default: tv_resnet50)')
parser.add_argument('--eval-dir', default='None', type=str,
                    help='model architecture (default: tv_resnet50)')
parser.add_argument('--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 6)')
parser.add_argument('--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset (default: 1000)')
parser.add_argument('--output', type=str, default='./timm_trt_eval_results.txt',
                    help='output directory of results')


timm_model_list = [
    'resnet18d',
    'resnet26',
    'resnet26d',
    'resnet26t',
    'resnet34',
    'resnet34d',
    'resnet50',
    'resnet50d',
    'resnet101',
    'resnet101d',
    'resnetrs50',
    'resnetrs101',
    'resnetrs152',
    'resnetrs270',
    'resnext50_32x4d',
    'resnext50d_32x4d',
    'tv_resnet50'
    'efficientnet_b0',
    'efficientnet_b1',
    'efficientnet_b2',
    'efficientnet_b3'
]

def main(args):
    results_txt = ''
    trt_engines = glob.glob(os.path.join(args.engine_path, "*.trt"))
    for timm_model in timm_model_list:
        engine_path = None
        for engine in trt_engines:
            if timm_model + "_" in engine:
                engine_path = engine
                break
        if engine_path is None:
            print("Model {} TensorRT build is not found".format(timm_model))
            continue
        model = create_model(
            timm_model,
            num_classes=args.num_classes,
            in_chans=3,
            exportable=True,
        )
        data_config = resolve_data_config(vars(args), model=model, use_test_size=True, verbose=True)
        print(data_config)
        # build eval loader
        val_loader = get_val_loader(data_root=args.eval_dir,
                                    n_worker=args.workers,
                                    batch_size=args.batch_size,
                                    val_size=data_config['input_size'][1],
                                    interpolation=data_config['interpolation'],
                                    mean=data_config['mean'],
                                    std=data_config['std'],
                                    cropt_pct=data_config['crop_pct'])

        evaluator = Evaluator(engine_path, val_loader, dtype=args.io_datatype)
        top1, top5, latency, qps = evaluator.evaluate()
        results_txt += "{}:TOP1={:.2f},TOP5={:.2f},LAT={:.2f},QPS={:.2f}\n".format(timm_model, top1, top5, latency, qps)

    with open(args.output, 'w') as file:
        file.write(results_txt)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)