import os
import sys
import logging
import argparse
import tensorrt as trt

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")

class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, onnx_path, input_shapes, verbose, **kwargs):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = 24 * (2 ** 30)  # 8 GB

        assert os.path.exists(onnx_path), "ONNX {} does not exist".format(onnx_path)
        self.onnx_path = onnx_path
        self.batch_size = None
        self.network = None
        self.parser = None
        self.input_shapes = input_shapes
        self.create_param(**kwargs)

    def create_param(self, **kwargs):
        log.info(kwargs)
        self.engine_path = kwargs.pop('engine', './')
        self.precision = kwargs.pop('precision', 'fp16')
        self.sparsity = kwargs.pop('sparsity', False)
        self.calib_num_images = kwargs.pop('calib_num_images', 128)
        self.calib_dataset = kwargs.pop('calib_dataset', None)
        self.calib_algo = kwargs.pop('calib_algo', 'entropy2')
        self.calib_batch_size = kwargs.pop('calib_batch_size', 1)
        self.calib_file = kwargs.pop('calib_file', "")
        self.quantized = kwargs.pop('quantized', False)
        self.tactic = kwargs.pop('tactic', 7)
        self.io_format = kwargs.pop('io_format', 'kLINEAR')
        self.io_datatype = kwargs.pop('io_datatype', 'fp16')

    def create_network(self):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(self.onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                sys.exit(1)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        log.info("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            log.info("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
        for output in outputs:
            log.info("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))
        # assert self.batch_size > 0
        # self.builder.max_batch_size = self.batch_size

    def set_io_format(self):
        if self.io_format == 'kLINEAR':
            formats = 1 << int(trt.TensorFormat.LINEAR)
        else:
            log.error("IO format {} is not supported".format(self.io_format))
            exit(0)

        for i in range(self.network.num_inputs):
            self.network.get_input(i).allowed_formats = formats
            if self.io_datatype == 'fp32':
                self.network.get_input(i).dtype = trt.float32
            elif self.io_datatype == 'fp16':
                self.network.get_input(i).dtype = trt.float16

        for i in range(self.network.num_outputs):
            self.network.get_output(i).allowed_formats = formats
            if self.io_datatype == 'fp32':
                self.network.get_output(i).dtype = trt.float32
            elif self.io_datatype == 'fp16':
                self.network.get_output(i).dtype = trt.float16

    def create_engine(self):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        :param calib_input: The path to a directory holding the calibration images.
        :param calib_cache: The path where to write the calibration cache to, or if it already exists, load it from.
        :param calib_num_images: The maximum number of images to use for calibration.
        :param calib_batch_size: The batch size to use for the calibration process.
        :param calib_preprocessor: The ImageBatcher preprocessor algorithm to use.
        """
        engine_path = self.engine_path
        precision = self.precision
        sparsity = self.sparsity
        calib_num_images = self.calib_num_images
        calibset = self.calib_dataset
        calib_algo = self.calib_algo
        calib_batch_size = self.calib_batch_size
        calib_file = self.calib_file
        input_shapes = self.input_shapes
        quantized = self.quantized
        tactic = self.tactic

        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        log.info("Building {} Engine in {}".format(precision, engine_path))

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        assert len(inputs) == len(input_shapes), "onnx inputs number {} does not match passed inputs number {}".format(len(inputs), len(input_shapes))

        profile = self.builder.create_optimization_profile()
        for input_shape, input in zip(input_shapes, inputs):
            if len(input_shape) == 1:
                min_shape = input_shape[0]
                opt_shape = input_shape[0]
                max_shape = input_shape[0]
            else:
                min_shape = input_shape[0]
                opt_shape = input_shape[1]
                max_shape = input_shape[2]
            profile.set_shape(input.name, min=min_shape, opt=opt_shape, max=max_shape)
            # set max batch size
            self.builder.max_batch_size = max_shape[0]

        self.config.add_optimization_profile(profile)

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            if not self.builder.platform_has_fast_int8:
                log.warning("INT8 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.INT8)
                if not quantized:
                    from .calib import EngineCalibrator
                    calib_profile = self.builder.create_optimization_profile()
                    for input_shape, input in zip(input_shapes, inputs):
                        if len(input_shape) == 1:
                            min_shape = input_shape[0]
                            opt_shape = input_shape[0]
                            max_shape = input_shape[0]
                        else:
                            min_shape = input_shape[0]
                            opt_shape = input_shape[1]
                            max_shape = input_shape[2]
                        calib_profile.set_shape(input.name, min=min_shape, opt=opt_shape, max=max_shape)
                    self.config.set_calibration_profile(calib_profile)
                    self.config.int8_calibrator = EngineCalibrator(calib_dataset=calibset,
                                                                   calib_batchsize=calib_batch_size,
                                                                   calib_num_images=calib_num_images,
                                                                   calib_algo=calib_algo,
                                                                   calib_dtype=self.io_datatype,
                                                                   calib_file=calib_file)

        self.config.set_tactic_sources(tactic)
        if sparsity:
            log.warning("Setting sparsity flag on builder_config.")
            self.config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

        self.set_io_format()
        with self.builder.build_engine(self.network, self.config) as engine, open(engine_path, "wb") as f:
            log.info("Serializing engine to file: {:}".format(engine_path))
            f.write(engine.serialize())

        return engine_path

def trt_engine_build(onnx_path, input_shapes, verbose, **kwargs):
    builder = EngineBuilder(onnx_path, input_shapes, verbose, **kwargs)
    builder.create_network()
    builder.create_engine()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", help="The input ONNX model file to load")
    parser.add_argument("--engine", help="The output path for the TRT engine")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"],
                        help="The precision mode to build in, either 'fp32', 'fp16' or 'int8', default: 'fp16'")
    parser.add_argument("--quantized", dest="quantized", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Enable more verbose log output")
    parser.add_argument("--calib_input", help="The directory holding images to use for calibration")
    parser.add_argument("--calib_num_images", default=128, type=int,
                        help="The maximum number of images to use for calibration, default: 128")
    parser.add_argument("--calib_batch_size", default=8, type=int,
                        help="The batch size for the calibration process, default: 1")
    parser.add_argument("--mean", default=[0.485, 0.456, 0.406],
                        help="mean of image dataset")
    parser.add_argument("--std", default=[0.229, 0.224, 0.225],
                        help="std of image dataset")
    parser.add_argument("--cropt_pct", default=0.875)
    parser.add_argument("--interpolation", default='bilinear', type=str,
                        help='interpolation method')
    parser.add_argument("--calib_algo", default="entropy2", choices=["entropy2", "minmax"],
                        help='calibration algorithm')
    parser.add_argument("--calib_file", default="",
                        help="calibration cache file")
    parser.add_argument("--sparsity", action="store_true",
                        help="enable sparsity")
    parser.add_argument('--input_shapes', default=[[1, 3, 224, 224]],
                        help='input shapes')
    parser.add_argument('--io_format', default='kLINEAR', type=str, choices=['kLINEAR'])
    parser.add_argument('--io_datatype', default='fp32', type=str, choices=['fp16', 'fp32'])

    # TACTIC_SOURCES_CASES = [
    #     (None, 7),  # By default, all sources are enabled.
    #     ([], 0),
    #     ([trt.TacticSource.CUBLAS], 1),
    #     ([trt.TacticSource.CUBLAS_LT], 2),
    #     ([trt.TacticSource.CUDNN], 4),
    #     ([trt.TacticSource.CUBLAS, trt.TacticSource.CUBLAS_LT], 3),
    #     ([trt.TacticSource.CUBLAS, trt.TacticSource.CUDNN], 5),
    #     ([trt.TacticSource.CUBLAS_LT, trt.TacticSource.CUDNN], 6),
    #     ([trt.TacticSource.CUDNN, trt.TacticSource.CUBLAS, trt.TacticSource.CUBLAS_LT], 7),
    # ]
    parser.add_argument("-t", "--tactic", type=int, default=7,
                        help="Set tactic policy, default: enable all tactic policies")
    args = parser.parse_args()
    print(args)

    from .data import ImageNetCalibDataset
    calibset = ImageNetCalibDataset(args.calib_input,
                                    args.input_shapes[0][2],
                                    interpolation=args.interpolation,
                                    mean=args.mean,
                                    std=args.std,
                                    cropt_pct=args.cropt_pct
                                    )
    kwargs = dict()
    for k, v in args.item():
        kwargs[k] = v
    kwargs['calib_dataset'] = calibset
    trt_engine_build(args.onnx, args.input_shapes, args.verbose, kwargs)



