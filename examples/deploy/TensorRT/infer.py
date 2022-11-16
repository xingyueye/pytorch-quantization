import logging
import torch
import tensorrt as trt

logging.basicConfig(level=logging.INFO)
logging.getLogger("TRTInfer").setLevel(logging.INFO)
log = logging.getLogger("TRTInfer")

def torch_dtype_from_trt(dtype):
    if dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError('%s is not supported by torch' % dtype)

def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError('%s is not supported by torch' % device)

class TensorRTInfer:
    """
    Implements inference for the EfficientNet TensorRT engine.
    """

    def __init__(self, engine_path):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

        print("engine max batch size = {}".format(self.engine.max_batch_size))
        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        self.input_num = 0
        self.output_num = 0
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                self.input_num += 1
            else:
                self.output_num += 1

        log.info("engine input_num = {} output_num = {}".format(self.input_num, self.output_num))

    def infer(self, *inputs):
        bindings = [None] * (self.input_num + self.output_num)

        for idx in range(self.input_num):
            self.context.set_binding_shape(idx, tuple(inputs[idx].shape))
            bindings[idx] = int(inputs[idx].contiguous().data_ptr())

        # create output tensors
        outputs = [None] * self.output_num
        for idx in range(self.input_num, self.input_num + self.output_num):
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))

            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[idx - self.input_num] = output
            bindings[idx] = int(output.data_ptr())

        self.context.execute_v2(bindings)

        outputs = tuple(outputs)
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs
