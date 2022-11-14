import os
import sys
import argparse
import subprocess
import torch

parser = argparse.ArgumentParser(description='PyTorch Partial Quantiztion Batch')
parser.add_argument('--timm_zoo', default=None, type=str,
                    help='timm zoo file')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--output_path', default='./', type=str,
                    help='path to save results')
parser.add_argument('--log_path', default='./', type=str,
                    help='path to save logs')

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
parser.add_argument('--calib_weight', action='store_true',
                    help='calib weight used')
parser.add_argument('--skip_layers_num', type=int, default=0,
                    help='number of layers to be skipped')
parser.add_argument('--skip_layers', nargs='+', default=[],
                    help='layers to be skipped')
parser.add_argument('--save_partial', action='store_true',
                    help='save partial model pth')
parser.add_argument('--export_onnx', action='store_true',
                    help='export to onnx')

parser.add_argument('--analyse_only', default=False, action='store_true',
                    help='analyse partial quantization results')

def partial_quant(args):
    pretrained_path = os.path.join(torch.hub.get_dir(), 'checkpoints')
    if not os.path.exists(pretrained_path):
        print("Create dir {}".format(pretrained_path))
        os.makedirs(pretrained_path)

    if not os.path.exists(args.log_path):
        print("Create dir {}".format(args.log_path))
        os.makedirs(args.log_path)

    timm_zoo_file = open(args.timm_zoo, 'r')
    timm_zoo_lines = timm_zoo_file.readlines()
    for idx, line in enumerate(timm_zoo_lines):
        model_name, weight_url, skip_layers = line.strip('\n').split(',')
        weight_name = os.path.basename(weight_url)
        if not os.path.exists(os.path.join(pretrained_path, weight_name)):
            print("Step1 Download model {} from url {}".format(model_name, weight_url))
            # Download pretrained weights
            download_cmd = 'wget {} -P {}'.format(weight_url, pretrained_path)
            print(download_cmd)
            os.system(download_cmd)

        print("Step2 Do partial quantization...")
        PARTIAL_QUANT_FILENAME = 'partial_quant_demo.py'
        log_file = os.path.join(args.log_path, "{}_partial_quantization.txt".format(model_name))
        command_list = [sys.executable, PARTIAL_QUANT_FILENAME,
                        '--data', args.data,
                        '--split', args.split,
                        '--model', model_name,
                        '--batch-size', str(args.batch_size),
                        '--calib_num', str(args.calib_num),
                        '--method', args.method,
                        '--sensitivity_method', args.sensitivity_method,
                        '--drop', str(args.drop),
                        '--per_layer_drop', str(args.per_layer_drop),
                        '--save_partial',
                        '--save_onnx',
                        '--workers', str(args.workers),
                        '--output', args.output_path]

        command_list += ['--pretrained'] if args.calib_weight is False \
            else ['--calib_weight', os.path.join(pretrained_path, weight_name), '--skip_layers_num', skip_layers]

        print(command_list)
        with open(log_file, 'w') as file_id:
            subp = subprocess.Popen(command_list, stdout=file_id, stderr=file_id)
            subp.communicate()

def partial_analyse(args):
    collection_file = 'partial_quant_collection_{}.txt'.format(args.sensitivity_method)
    cfid = open(collection_file, 'w')

    timm_zoo_file = open(args.timm_zoo, 'r')
    timm_zoo_lines = timm_zoo_file.readlines()
    for idx, line in enumerate(timm_zoo_lines):
        model_name, _, _ = line.strip('\n').split(',')
        quant_file = os.path.join(args.output_path, model_name + '_ptq.txt')
        partial_file = os.path.join(args.output_path, model_name + '_' + args.sensitivity_method + '_pptq.txt')
        if os.path.exists(quant_file):
            with open(quant_file, 'r') as qfid:
                lines = qfid.readlines()
                _, name, fp32_acc, ptq_acc = lines[0].strip('\n'), lines[1].strip('\n'), \
                float(lines[2].strip('\n')), float(lines[3].strip('\n'))
                diff_acc = round(fp32_acc - ptq_acc, 4)
                # quant_str = name + " " + str(fp32_acc) + " " + str(ptq_acc) + " " + str(diff_acc) + '\n'
                quant_str = "{}\t\t{:.4f} {:.4f} {:.4f}\n".format(name, fp32_acc, ptq_acc, diff_acc)
                cfid.write(quant_str)
        elif os.path.exists(partial_file):
            with open(partial_file, 'r') as pfid:
                lines = pfid.readlines()
                lines_num = len(lines)
                skip_num = lines_num - 5
                _, name, fp32_acc, ptq_acc, part_acc = lines[0].strip('\n'), lines[1].strip('\n'), \
                float(lines[2].strip('\n')), float(lines[3].strip('\n')), float(lines[4].strip('\n'))
                diff_acc = round(fp32_acc - part_acc, 4)
                # part_str = name + " " + str(fp32_acc) + " " + str(ptq_acc) + " " + \
                #            str(part_acc) + " " + str(diff_acc) + " " + str(skip_num)
                part_str = "{}\t\t{:.4f} {:.4f} {:.4f} {:.4f} {}\t".format(name, fp32_acc, ptq_acc, part_acc, diff_acc,
                                                                           skip_num)
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


