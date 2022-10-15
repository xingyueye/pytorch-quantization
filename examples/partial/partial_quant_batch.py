import os
import sys
import argparse
import subprocess

import timm_urls

parser = argparse.ArgumentParser(description='PyTorch Partial Quantiztion Batch')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--pretrained_path', default=None, type=str,
                    help='path to pretrained weights')

parser.add_argument('--method', type=str, default='entropy', choices=['max', 'entropy', 'percentile', 'mse'])
parser.add_argument('--sensitivity_method', type=str, default='mse', choices=['mse', 'cosine', 'top1', 'snr'])
parser.add_argument('--percentile', type=float, default=99.99)
parser.add_argument('--drop', type=float, default=0.5)
parser.add_argument('--per_layer_drop', type=float, default=0.2)
parser.add_argument('--calib_num', type=int, default=4)
parser.add_argument('--calib_weight', type=str, default=None)

def partial_quant(args):
    if not os.path.exists(args.pretrained_path):
        print("Create dir {}".format(args.pretrained_path))
        os.makedirs(args.pretrained_path)

    for k, v in timm_urls.timm_urls.items():
        model_name = k
        weight_url = v[0]
        print("Step1 Download model {} from url {}".format(model_name, weight_url))
        # Download pretrained weights
        download_cmd = 'wget {} -P {}'.format(weight_url, args.pretrained_path)
        print(download_cmd)
        os.system(download_cmd)

        print("Step2 Do partial quantization...")
        PARTIAL_QUANT_FILENAME = 'partial_quant_demo.py'
        log_file = "{}_partial_quantization.txt".format(model_name)
        command_list = [sys.executable, PARTIAL_QUANT_FILENAME,
                        '--data', args.data, '--pretrained',
                        '--split', args.split, '--model', model_name,
                        '--batch_size', args.batch_size, '--calib_num', args.calib_num,
                        '--method', args.method, '--sensitivity_method', args.senstivity_methd,
                        '--drop', args.drop, '--per_layer_drop', args.per_layer_drop,
                        '--workers', args.workers]

        print(command_list)
        with open(log_file, 'w') as file_id:
            subp = subprocess.Popen(command_list, stdout=file_id, stderr=file_id)
            subp.communicate()

if __name__ == '__main__':
    args = parser.parse_args()


