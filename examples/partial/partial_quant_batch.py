import os
import sys
import argparse
import subprocess
import glob

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
parser.add_argument('--output_path', default='./', type=str,
                    help='path to save results')

parser.add_argument('--method', type=str, default='entropy', choices=['max', 'entropy', 'percentile', 'mse'])
parser.add_argument('--sensitivity_method', type=str, default='mse', choices=['mse', 'cosine', 'top1', 'snr'])
parser.add_argument('--percentile', type=float, default=99.99)
parser.add_argument('--drop', type=float, default=0.5)
parser.add_argument('--per_layer_drop', type=float, default=0.2)
parser.add_argument('--calib_num', type=int, default=4)
parser.add_argument('--calib_weight', type=str, default=None)

parser.add_argument('--analyse_only', default=False, action='store_true',
                    help='analyse partial quantization results')

def partial_quant(args):
    if not os.path.exists(args.pretrained_path):
        print("Create dir {}".format(args.pretrained_path))
        os.makedirs(args.pretrained_path)

    for k, v in timm_urls.timm_urls.items():
        model_name = k
        weight_url = v[0]
        weight_name = os.path.basename(weight_url)
        if not os.path.exists(os.path.join(args.pretrained_path, weight_name)):
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
                        '--batch-size', str(args.batch_size), '--calib_num', str(args.calib_num),
                        '--method', args.method, '--sensitivity_method', args.sensitivity_method,
                        '--drop', str(args.drop), '--per_layer_drop', str(args.per_layer_drop),
                        '--workers', str(args.workers), '--output', args.output_path]

        print(command_list)
        with open(log_file, 'w') as file_id:
            subp = subprocess.Popen(command_list, stdout=file_id, stderr=file_id)
            subp.communicate()

def partial_analyse(args):
    collection_file = 'partial_quant_collection_{}.txt'.format(args.sensitivity_method)
    cfid = open(collection_file, 'w')

    for k, v in timm_urls.timm_urls.items():
        quant_file = os.path.join(os.path.output, k + '_quant.txt')
        partial_file = os.path.join(os.path.output, k + '_' + args.sensitivity_method + '_quant.txt')
        if os.path.exists(quant_file):
            with open(quant_file, 'r') as qfid:
                lines = qfid.readlines()
                _, name, fp32_acc, ptq_acc = lines[0].strip('\n'), lines[1].strip('\n'), \
                float(lines[2].strip('\n')), float(lines[3].strip('\n'))
                diff_acc = fp32_acc - ptq_acc
                quant_str = name + " " + str(fp32_acc) + " " + str(ptq_acc) + " " + str(diff_acc) + '\n'
                cfid.write(quant_str)
        elif os.path.exists(partial_file):
            with open(partial_file, 'r') as pfid:
                lines = pfid.readlines()
                lines_num = len(lines)
                _, name, fp32_acc, ptq_acc, part_acc = lines[0].strip('\n'), lines[1].strip('\n'), \
                float(lines[2].strip('\n')), float(lines[3].strip('\n')), float(lines[4].strip('\n'))
                diff_acc = fp32_acc - part_acc
                part_str = name + " " + str(fp32_acc) + " " + str(ptq_acc) + " " + \
                           str(part_acc) + " " + str(diff_acc)
                for idx in range(5, lines_num):
                    part_str = part_str + " " + lines[idx].strip('\n')
                part_str = part_str + "\n"
                cfid.write(part_str)
    cfid.close()


if __name__ == '__main__':
    args = parser.parse_args()
    if args.analyse_only is False:
        partial_quant(args)
    partial_analyse(args)


