import json
import os
import onnx
import argparse

from TensorRT.build import trt_engine_build
from TensorRT.data import ImageNetCalibDataset, get_val_loader
from TensorRT.eval import Evaluator
from onnx_utils import onnx_remove_qdqnode, save_calib_cache_file


def prepare_env(model_zoo_path):
    onnx_path = os.path.join(model_zoo_path, 'onnx')
    onnx_rmqdq_path = os.path.join(model_zoo_path, 'onnx_rmqdq')
    cache_rmqdq_path = os.path.join(model_zoo_path, 'cache_rmqdq')
    prep_path = os.path.join(model_zoo_path, 'prep')
    model_file = os.path.join(model_zoo_path, 'partial_quant_collection_mse.txt')
    profile_file = os.path.join(model_zoo_path, 'partial_quant_profile.txt')

    if not os.path.exists(onnx_rmqdq_path):
        print("Create {}".format(onnx_rmqdq_path))
        os.makedirs(onnx_rmqdq_path)
    if not os.path.exists(cache_rmqdq_path):
        print("Create {}".format(cache_rmqdq_path))
        os.makedirs(cache_rmqdq_path)
    trt_fp16_engine_path = os.path.join(model_zoo_path, 'trt_fp16')
    if not os.path.exists(trt_fp16_engine_path):
        print("Create {}".format(trt_fp16_engine_path))
        os.makedirs(trt_fp16_engine_path)
    trt_int8_engine_path = os.path.join(model_zoo_path, 'trt_int8')
    if not os.path.exists(trt_int8_engine_path):
        print("Create {}".format(trt_int8_engine_path))
        os.makedirs(trt_int8_engine_path)
    trt_int8_rmqdq_engine_path = os.path.join(model_zoo_path, 'trt_int8_rmqdq')
    if not os.path.exists(trt_int8_rmqdq_engine_path):
        print("Create {}".format(trt_int8_rmqdq_engine_path))
        os.makedirs(trt_int8_rmqdq_engine_path)

    return model_file, onnx_path, onnx_rmqdq_path, cache_rmqdq_path, \
           prep_path, trt_fp16_engine_path, trt_int8_engine_path, trt_int8_rmqdq_engine_path, profile_file




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_zoo_path", help="path to onnx")
    parser.add_argument("--verbose", action="store_true", help="Enable more verbose log output")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"],
                        help="The precision mode to build in, either 'fp32', 'fp16' or 'int8', default: 'fp16'")
    parser.add_argument("--eval_dir", type=str, default=None,
                        help="The input to eval, a directory of images")
    parser.add_argument("--workers", type=int, default=4,
                        help="worker number")
    parser.add_argument("--calib_input", help="The directory holding images to use for calibration")
    parser.add_argument("--calib_num_images", default=32, type=int,
                        help="The maximum number of images to use for calibration, default: 128")
    parser.add_argument("--calib_batch_size", default=4, type=int,
                        help="The batch size for the calibration process, default: 1")
    parser.add_argument("--calib_algo", default="entropy2", choices=["entropy2", "minmax"],
                        help='calibration algorithm')
    parser.add_argument("--sparsity", action="store_true",
                        help="enable sparsity")
    parser.add_argument('--io_format', default='kLINEAR', type=str, choices=['kLINEAR'])
    parser.add_argument('--io_datatype', default='fp32', type=str, choices=['fp16', 'fp32'])
    parser.add_argument('--graph_dump', default=False, action='store_true',
                        help='dump engine graph to json file')
    args = parser.parse_args()
    print(args)

    assert os.path.exists(args.model_zoo_path), "model_zoo_path {} does not exist"

    model_file, onnx_path, onnx_rmqdq_path, cache_rmqdq_path, prep_path, \
    trt_fp16_epath, trt_int8_epath, trt_int8_rmqdq_epath, profile_file = prepare_env(args.model_zoo_path)

    fprof = open(profile_file, 'w')

    with open(model_file, 'r') as mfile:
        lines = mfile.readlines()
        for line in lines:
            quant_info = line.strip('\n').split(',')
            model_name = quant_info[0]
            if len(quant_info) == 4:
                suffix = '_ptq'
                fp32_top1, ptq_int8_top1, pptq_int8_top1 = float(quant_info[1]), float(quant_info[2]), 0.0
            else:
                suffix = '_mse_pptq'
                fp32_top1, ptq_int8_top1, pptq_int8_top1 = float(quant_info[1]), float(quant_info[2]), float(quant_info[3])
            onnx_file = os.path.join(onnx_path, model_name + suffix + '.onnx')
            model_prep = os.path.join(prep_path, model_name + '_preproc.json')
            prep_config = json.load(open(model_prep))[model_name]
            print(prep_config)
            input_size = prep_config['input_size']
            interpolation = prep_config['interpolation']
            mean = prep_config['mean']
            std = prep_config['std']
            crop_pct = prep_config['crop_pct']
            input_shapes = [[[args.calib_batch_size] + input_size]]

            # remove qdq node of onnx and save calib cache
            onnx_model = onnx.load(onnx_file)
            onnx_rmqdq, activation_map = onnx_remove_qdqnode(onnx_model)
            onnx_rmqdq_file = os.path.join(onnx_rmqdq_path, model_name + suffix + '_rm_qdq.onnx')
            onnx.save(onnx_rmqdq, onnx_rmqdq_file)
            cache_rmqdq_file = os.path.join(cache_rmqdq_path, model_name + suffix + '_rm_qdq_calibration.cache')
            save_calib_cache_file(cache_rmqdq_file, activation_map)

            # build eval loader
            val_loader = get_val_loader(data_root=args.eval_dir,
                                        n_worker=args.workers,
                                        batch_size=args.calib_batch_size,
                                        val_size=input_size[1],
                                        interpolation=interpolation,
                                        mean=mean,
                                        std=std,
                                        cropt_pct=crop_pct)

            # build fp16 trt engine
            kwargs = dict()
            kwargs['precision'] = 'fp16'
            kwargs['calib_num_images'] = args.calib_num_images
            kwargs['calib_batch_size'] = args.calib_batch_size
            kwargs['calib_algo'] = args.calib_algo
            kwargs['io_format'] = args.io_format
            kwargs['io_datatype'] = args.io_datatype
            trt_fp16_engine = os.path.join(trt_fp16_epath, model_name + '_fp16.trt')
            kwargs['engine'] = trt_fp16_engine
            kwargs['graph_dump'] = args.graph_dump
            trt_engine_build(onnx_rmqdq_file, input_shapes, args.verbose, **kwargs)

            evaluator = Evaluator(trt_fp16_engine, val_loader, dtype=args.io_datatype)
            fp16_top1, fp16_top5, fp16_latency, fp16_qps = evaluator.evaluate()
            print("FP16 TOP1 = {:.2f}, FP16 TOP5 = {:.2f}, FP16 LAT={:.2f}, FP16 QPS={:.2f}".format(fp16_top1,
                                                                                                    fp16_top5,
                                                                                                    fp16_latency,
                                                                                                    fp16_qps))

            # build int8 trt engine via using calibration
            calibset = ImageNetCalibDataset(args.calib_input,
                                            input_size[1],
                                            interpolation=interpolation,
                                            mean=mean,
                                            std=std,
                                            cropt_pct=crop_pct)

            kwargs = dict()
            kwargs['precision'] = 'int8'
            kwargs['calib_num_images'] = args.calib_num_images
            kwargs['calib_batch_size'] = args.calib_batch_size
            kwargs['calib_algo'] = args.calib_algo
            kwargs['calib_dataset'] = calibset
            kwargs['calib_file'] = os.path.join(trt_int8_epath, model_name + '_calibration.cache')
            kwargs['io_format'] = args.io_format
            kwargs['io_datatype'] = args.io_datatype
            trt_int8_engine = os.path.join(trt_int8_epath, model_name + '_int8.trt')
            kwargs['engine'] = trt_int8_engine
            kwargs['graph_dump'] = args.graph_dump
            trt_engine_build(onnx_rmqdq_file, input_shapes, args.verbose, **kwargs)

            evaluator = Evaluator(trt_int8_engine, val_loader, dtype=args.io_datatype)
            int8_top1, int8_top5, int8_latency, int8_qps = evaluator.evaluate()
            print("INT8 TOP1 = {:.2f}, INT8 TOP5 = {:.2f}, INT8 LAT={:.2f}, INT8 QPS={:.2f}".format(int8_top1,
                                                                                                    int8_top5,
                                                                                                    int8_latency,
                                                                                                    int8_qps))

            # build int8 trt engine via using calibration cache file
            calibset = ImageNetCalibDataset(args.calib_input,
                                            input_size[1],
                                            interpolation=interpolation,
                                            mean=mean,
                                            std=std,
                                            cropt_pct=crop_pct)
            kwargs = dict()
            kwargs['precision'] = 'int8'
            kwargs['calib_num_images'] = args.calib_num_images
            kwargs['calib_batch_size'] = args.calib_batch_size
            kwargs['calib_algo'] = args.calib_algo
            kwargs['calib_dataset'] = calibset
            kwargs['calib_file'] = cache_rmqdq_file
            kwargs['io_format'] = args.io_format
            kwargs['io_datatype'] = args.io_datatype
            trt_int8_rmqdq_engine = os.path.join(trt_int8_rmqdq_epath, model_name + suffix + '_rmqdq_int8.trt')
            kwargs['engine'] = trt_int8_rmqdq_engine
            kwargs['graph_dump'] = args.graph_dump
            trt_engine_build(onnx_rmqdq_file, input_shapes, args.verbose, **kwargs)

            evaluator = Evaluator(trt_int8_rmqdq_engine, val_loader, dtype=args.io_datatype)
            int8_rmqdq_top1, int8_rmqdq_top5, int8_rmqdq_latency, int8_rmqdq_qps = evaluator.evaluate()
            print("INT8 RMQDQ TOP1 = {:.2f}, INT8 RMQDQ TOP5 = {:.2f}, INT8 RMQDQ LAT={:.2f}, INT8 RMQDQ QPS={:.2f}".\
                                                                                                    format(
                                                                                                    int8_rmqdq_top1,
                                                                                                    int8_rmqdq_top5,
                                                                                                    int8_rmqdq_latency,
                                                                                                    int8_rmqdq_qps))

            prof_str = "{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n".\
                                                                                                    format(
                                                                                                    model_name,
                                                                                                    fp32_top1,
                                                                                                    ptq_int8_top1,
                                                                                                    pptq_int8_top1,
                                                                                                    fp16_top1,
                                                                                                    fp16_latency,
                                                                                                    fp16_qps,
                                                                                                    int8_top1,
                                                                                                    int8_latency,
                                                                                                    int8_qps,
                                                                                                    int8_rmqdq_top1,
                                                                                                    int8_rmqdq_latency,
                                                                                                    int8_rmqdq_qps
                                                                                                    )
            fprof.write(prof_str)
    fprof.close()
