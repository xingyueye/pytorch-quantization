import time
import logging
import argparse
import torch
import torch.nn as nn

from .data import get_val_loader
from .utils import AverageMeter, progress_bar, accuracy
from .infer import TensorRTInfer

logging.basicConfig(level=logging.INFO)
logging.getLogger("Evaluator").setLevel(logging.INFO)
log = logging.getLogger("Evaluator")


class Evaluator(object):
    def __init__(self, engine_path, val_loader, dtype='fp32'):
        # we hope user hardcode to simplify the evalution
        self.engine_path = engine_path
        self.val_loader = val_loader
        self.trt_infer = TensorRTInfer(self.engine_path)
        self.dtype = dtype
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = val_loader.batch_size

    def evaluate(self):
        # for trt build, model already on GPU
        # model.cuda()
        # model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        latency = AverageMeter()
        warmup = 5
        cnt = 0
        log.info("Evaluate {}...".format(self.engine_path))
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                if self.dtype == 'fp16':
                    inputs = inputs.half()
                s_t = time.time()
                outputs = self.trt_infer.infer(inputs)
                e_t = time.time()
                loss = self.criterion(outputs, targets)
                cnt+=1
                if cnt > warmup:
                    latency.update(1000*(e_t - s_t), 1)
                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                progress_bar(batch_idx, len(self.val_loader), 'Loss: {:.3f} | Acc1: {:.3f}% | Acc5: {:.3f}%'
                            .format(losses.avg, top1.avg, top5.avg))

        return top1.avg, top5.avg, latency.avg, 1000.0/latency.avg * self.batch_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, default=None,
                        help="The TensorRT engine to infer with")
    parser.add_argument("--eval_dir", type=str, default=None,
                        help="The input to infer, either a single image path, or a directory of images")
    parser.add_argument("--workers", type=int, default=4,
                        help="worker number")
    parser.add_argument("--input_shape", nargs='+', type=int, default=[1, 3, 224, 224],
                        help="input shape")
    parser.add_argument("--mean", default=[0.485, 0.456, 0.406],
                        help="mean of image dataset")
    parser.add_argument("--std", default=[0.229, 0.224, 0.225],
                        help="std of image dataset")
    parser.add_argument("--cropt_pct", type=float, default=0.875,
                        help="ratio of picture crop")
    parser.add_argument("--interpolation", type=str, default='bilinear',
                        help="interpolation method")
    parser.add_argument("--io_dtype", type=str, default='fp16',
                        help="input/output data precision")

    args = parser.parse_args()
    print(args)

    val_loader = get_val_loader(data_root=args.eval_dir,
                                n_worker=args.workers,
                                batch_size=args.input_shape[0],
                                val_size=args.input_shape[2],
                                interpolation=args.interpolation,
                                mean=args.mean,
                                std=args.std,
                                cropt_pct=args.cropt_pct)

    engine_eval = Evaluator(args.engine, val_loader, dtype = args.io_dtype)
    engine_eval.evaluate()


