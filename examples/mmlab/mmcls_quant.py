# Copyright (c) OpenMMLab. All rights reserved.
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from numbers import Number
import numpy as np
import logging
import logging.handlers

from mmcls import __version__
from mmcls.apis import set_random_seed, train_model
from mmcls.datasets import build_dataset, build_dataloader
from mmcls.models import build_classifier
from mmcls.utils import collect_env, get_root_logger
from mmcls.apis import set_random_seed, train_model, multi_gpu_test, single_gpu_test

# Differences from mmclassification
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from pytorch_quantization.model_quantizer import ModelQuantizerFactory
import functools

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument('--device', choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
             '"accuracy", "precision", "recall", "f1_score", "support" for single '
             'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
             'multi-label dataset')
    parser.add_argument(
        '--metric-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be parsed as a dict metric_options for dataset.evaluate()'
             ' function.')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--test-dataset',
        choices=['test', 'val'],
        default='test',
        help='dataset used for testing')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--ptq', action='store_true', default=False, help='Enable adaptive partail ptq')
    parser.add_argument('--qat', action='store_true', default=False, help='Enable adaptive partail qat')
    parser.add_argument('--partial', action='store_true', default=False, help='Enable adaptive partail partial-ptq')
    parser.add_argument('--save_quant_model', action='store_true', default=False, help='Enable to save quantized models')
    parser.add_argument('--pretrained_calib', type=str, default='', help='Pretrained model')
    parser.add_argument('--quant_config', type=str, default='', help='Pretrained model')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def test(data_loader, model, args, dataset, distributed, logger):
    if not distributed:
        outputs = single_gpu_test(model, data_loader)
    else:
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        results = {}
        if args.metrics:
            eval_results = dataset.evaluate(outputs, args.metrics,
                                            args.metric_options)
            results.update(eval_results)
            logger.info(f'Results ===================> {results}')
            for k, v in eval_results.items():
                if isinstance(v, np.ndarray):
                    v = [round(out, 2) for out in v.tolist()]
                elif isinstance(v, Number):
                    v = round(v, 2)
                elif isinstance(v, list):
                    v = [[round(o, 2) for o in out.tolist()] if isinstance(out, np.ndarray) 
                        else round(out, 2) for out in v]
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        print(k, kk, vv)
                else:
                    raise ValueError(f'Unsupport metric type: {type(v)}')
                print(f'\n{k} : {v}')
    # print(eval_results['accuracy_top-1'])
    if 'accuracy_top-1' in eval_results:
        output = eval_results['accuracy_top-1']
    else:
        # multi-task , eval_results <class 'dict'> ; eval_results.items() <class 'dict'> 
        output = {}
        for k, v in eval_results.items():
            for kk, vv in v.items():
                if kk not in output or vv > output[kk]:
                    output[kk] = vv
    return output

def main():
    args = parse_args()
    # print(args.cfg_options)
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    model = build_classifier(cfg.model)
    model.init_weights()

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if not distributed:
        if args.device == 'cpu':
            model = model.cpu()
        else:
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)
            if not model.device_ids:
                assert mmcv.digit_version(mmcv.__version__) >= (1, 4, 4), \
                    'To test with CPU, please confirm your mmcv version ' \
                    'is not lower than v1.4.4'
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmcls version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmcls_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES)
    
    if args.test_dataset == 'test':
        dataset_cfg = cfg.data.test
    else:
        dataset_cfg = cfg.data.val
    test_dataset = build_dataset(dataset_cfg)
    test_loader = build_dataloader(
        test_dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=True)

    calib_file = os.path.join(cfg.work_dir, args.checkpoint.split('/')[-1].replace(".pth", "_calib.pth"))
    quantizer = ModelQuantizerFactory.get_model_quantizer('MMCls',
                                                            cfg.model.type,
                                                            model,
                                                            args.quant_config,
                                                            calib_weights=calib_file)
    print(model)
    model = quantizer.model      
    
    _test = functools.partial(test, args=args, dataset=test_dataset, distributed=distributed, logger=logger)
    if args.ptq:
        quantizer.calibration(test_loader, test_loader.batch_size, save_calib_model=True)
        if args.partial:
            quantizer.load_calib_weights()
            skip_layers, ori_acc1, partial_acc1, sensitivity = quantizer.partial_quant(test_loader, _test)
        _test(test_loader, model)

    if args.qat:
        quantizer.load_calib_weights()
        train_model(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=True,
            timestamp=timestamp,
            device='cpu' if args.device == 'cpu' else 'cuda',
            meta=meta)
    
    if args.save_quant_model:
        quantizer.export_onnx(data_shape=(1, 3, 224, 224))

if __name__ == '__main__':
    main()
