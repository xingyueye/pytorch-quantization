#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


"""TensorQuantizer Module"""
import math
from absl import logging

import torch
from torch import nn

from pytorch_quantization.tensor_quant import QuantDescriptor, tensor_quant, lsq_fake_tensor_quant

from pytorch_quantization import calib

import pytorch_quantization.utils as quant_utils
from pytorch_quantization.nn import TensorQuantizer
from pytorch_quantization.nn.functional import GradScaleFunction

__all__ = ['LSQTensorQuantizer']

class LSQTensorQuantizer(TensorQuantizer):
    """Tensor quantizer module

    This module uses tensor_quant or fake_tensor_quant funßtion to quantize a tensor. And wrappers variable, moving
    statistics we'd want when training a quantized network.

    Experimental features:
        ``clip`` stage learns range before enabling quantization.
        ``calib`` stage runs calibration

    Args:
        quant_desc: An instance of :func:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
        disabled: A boolean. If True, by pass the whole module returns input. Default False.
        if_quant: A boolean. If True, run main quantization body. Default True.
        if_clip: A boolean. If True, clip before quantization and learn amax. Default False.
        if_calib: A boolean. If True, run calibration. Not implemented yet. Settings of calibration will probably
            go to :func:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.

    Raises:

    Readonly Properties:
        - axis:
        - fake_quant:
        - scale:
        - step_size:

    Mutable Properties:
        - num_bits:
        - unsigned:
        - amax:
    """

    # An experimental static switch for using pytorch's native fake quantization
    # Primary usage is to export to ONNX
    # use_fb_fake_quant = False

    def __init__(self, quant_desc=QuantDescriptor(), disabled=False, if_quant=True, if_clip=False, if_calib=False):
        """Initialize quantizer and set up required variables"""
        super(LSQTensorQuantizer, self).__init__(quant_desc, disabled, if_quant, if_clip, if_calib)

        self._learn_scale = quant_desc._learn_scale
        self._learn_scale_type = quant_desc._learn_scale_type
        assert quant_desc._learn_scale, "LSQ series Quantizer need the learnable scale!"
        self._learn_scale_init = False
        self.scale_for_grad = None

    @property
    def scale(self):
        if self._scale is None:
            logging.critical("Accessing scale before quantizing any tensor!")
        return self._scale

    # pylint:enable=missing-docstring
    def load_calib_amax(self, *args, **kwargs):
        """Load amax from calibrator.

        Updates the amax buffer with value computed by the calibrator, creating it if necessary.
        *args and **kwargs are directly passed to compute_amax, except "strict" in kwargs. Refer to
        compute_amax for more details.
        """
        strict = kwargs.pop("strict", True)
        if getattr(self, '_calibrator', None) is None:
            raise RuntimeError("Calibrator not created.")
        if self._learn_scale and self._learn_scale_type == 'lsq':
            calib_amax = self._calibrator.compute_amax_lsq()
        else:
            raise "Unsupported amax computing method for LSQ Quantizer!"

        if calib_amax is None:
            err_msg = "Calibrator returned None. This usually happens when calibrator hasn't seen any tensor."
            if not strict:
                logging.warning(err_msg)
                logging.warning("Set amax to NaN!")
                calib_amax = torch.tensor(math.nan)
            else:
                raise RuntimeError(err_msg + " Passing 'strict=False' to `load_calib_amax()` will ignore the error.")
        logging.warning("Load calibrated amax, shape={}.".format(calib_amax.shape))
        logging.log_first_n(
            logging.WARNING, "Call .cuda() if running on GPU after loading calibrated amax.", 1)
        if not hasattr(self, '_amax'):
            self.register_buffer('_amax', calib_amax.data)
        else:
            self._amax.copy_(calib_amax)

    def _param_init(self, inputs=None):
        init_weight = self._amax
        value = torch.nn.Parameter(init_weight * 2 / ((2.0**(self._num_bits - 1 + int(self._unsigned)) - 1.0) ** 0.5), requires_grad=True)
        epsilon = 1. / (1 << 24)
        if value.min() <= epsilon:
            zero_amax_mask = (value <= epsilon)
            value.data[zero_amax_mask] = 1.

        if hasattr(self, '_scale'):
            self._scale.data = value.data
        else:
            self.register_parameter('_scale', value)

    def init_learn_scale(self, inputs=None):
        """Initialize learned scale from PTQ amax or lsq_init"""
        self._param_init(inputs)
        self._learn_scale_init = True  

    def _quant_forward(self, inputs):
        """Quantized forward pass."""
        if self.scale_for_grad is None:
            numel = inputs.numel()
            self.max_bound = torch.tensor((2.0**(self.num_bits - 1 + int(self.unsigned))) - 1.0, device=inputs.device)
            self.scale_for_grad = torch.tensor(1.0 / ((self.max_bound * numel) ** 0.5), device=inputs.device)
        
        if not self._learn_scale_init:
            self.init_learn_scale(inputs)

        _scale = GradScaleFunction.apply(self._scale, self.scale_for_grad)
        amax = _scale * self.max_bound

        if self._fake_quant:
            if not TensorQuantizer.use_fb_fake_quant:
                outputs = lsq_fake_tensor_quant(inputs, _scale, self._num_bits, self._unsigned, self._narrow_range)
            else:
                if inputs.dtype == torch.half or amax.dtype == torch.half:
                    raise Exception("Exporting to ONNX in fp16 is not supported. Please export in fp32, i.e. disable AMP.")
                outputs = self._fb_fake_quant(inputs, amax)
        else:
            outputs, self._scale = tensor_quant(inputs, amax, self._num_bits, self._unsigned)
        # print(outputs)
        # print((inputs-outputs).max())

        return outputs

    def forward(self, inputs):
        """Apply tensor_quant function to inputs

        Args:
            inputs: A Tensor of type float32.

        Returns:
            outputs: A Tensor of type output_dtype
        """
        if self._disabled:
            return inputs

        outputs = inputs

        if self._if_calib:
            if self._calibrator is None:
                raise RuntimeError("Calibrator was not created.")
            self._calibrator.lsq_collect(inputs)

        if self._if_quant:
            outputs = self._quant_forward(inputs)

        return outputs