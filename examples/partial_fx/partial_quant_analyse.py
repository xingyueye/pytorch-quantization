import os
import glob
import argparse

parser = argparse.ArgumentParser(description='PyTorch Partial Quantiztion Batch')
parser.add_argument('--partial_path', default=None, type=str,
                    help='path of partial quantization')

def partial_analyse(args):
    results_file = 'partial_quant_results.txt'
    cfid = open(os.path.join(args.partial_path, results_file), 'w')
    partial_result_dir = os.path.join(args.partial_path, 'results')
    partial_result_files = sorted(glob.glob(partial_result_dir + '/*.txt'))
    for partial_file in partial_result_files:
        if '_ptq.txt' in partial_file:
            with open(partial_file, 'r') as qfid:
                lines = qfid.readlines()
                _, name, fp32_acc, ptq_acc = lines[0].strip('\n'), lines[1].strip('\n'), \
                float(lines[2].strip('\n')), float(lines[3].strip('\n'))
                diff_acc = round(fp32_acc - ptq_acc, 4)
                # quant_str = name + " " + str(fp32_acc) + " " + str(ptq_acc) + " " + str(diff_acc) + '\n'
                quant_str = "{},{:.4f},{:.4f},{:.4f}\n".format(name, fp32_acc, ptq_acc, diff_acc)
                cfid.write(quant_str)
        elif '_pptq.txt' in partial_file:
            with open(partial_file, 'r') as pfid:
                lines = pfid.readlines()
                lines_num = len(lines)
                skip_num = lines_num - 5
                _, name, fp32_acc, ptq_acc, part_acc = lines[0].strip('\n'), lines[1].strip('\n'), \
                float(lines[2].strip('\n')), float(lines[3].strip('\n')), float(lines[4].strip('\n'))
                diff_acc = round(fp32_acc - part_acc, 4)
                # part_str = name + " " + str(fp32_acc) + " " + str(ptq_acc) + " " + \
                #            str(part_acc) + " " + str(diff_acc) + " " + str(skip_num)
                part_str = "{},{:.4f},{:.4f},{:.4f},{:.4f},{}".format(name, fp32_acc, ptq_acc, part_acc, diff_acc,
                                                                           skip_num)
                for idx in range(5, lines_num):
                    part_str = part_str + "," + lines[idx].strip('\n')
                part_str = part_str + "\n"
                cfid.write(part_str)
    cfid.close()


if __name__ == '__main__':
    args = parser.parse_args()
    partial_analyse(args)