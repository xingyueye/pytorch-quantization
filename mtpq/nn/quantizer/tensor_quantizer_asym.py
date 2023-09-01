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

from mtpq.tensor_quant import QuantDescriptor, tensor_quant, fake_affine_tensor_quant
from mtpq.nn.modules.clip import Clip
# from mtpq.nn.modules.grad_scale import GradScale
from mtpq.nn.quantizer.tensor_quantizer import TensorQuantizer
from mtpq import calib

import mtpq.utils as quant_utils

__all__ = ['TensorQuantizer_asym']

# FAKE_QUANT_FUNC_MAP={
#     "qat": NaiveFakeQuantizer,
#     "lsq": LSQFakeQuantizer,
#     "stable_lsq": StableLSQFakeQuantizer,
#     "lsq_plus": LSQPlusFakeQuantizer
# }

CALIB_METHOD_MAP={
    "histogram": calib.HistogramCalibrator,
    "asym_histogram": calib.AsymHistogramCalibrator,
    "max": calib.MaxCalibrator,
    "minmax": calib.MinMaxCalibrator,
    "lsq": calib.LSQCalibrator
}

class TensorQuantizer_asym(TensorQuantizer):
    """Tensor quantizer module asymmertry

    This module uses tensor_quant or fake_tensor_quant funßtion to quantize a tensor. And wrappers variable, moving
    statistics we'd want when training a quantized network.

    Experimental features:
        ``clip`` stage learns range before enabling quantization.
        ``calib`` stage runs calibration

    Args:
        quant_desc: An instance of :func:`QuantDescriptor <mtpq.tensor_quant.QuantDescriptor>`.
        disabled: A boolean. If True, by pass the whole module returns input. Default False.
        if_quant: A boolean. If True, run main quantization body. Default True.
        if_clip: A boolean. If True, clip before quantization and learn amax. Default False.
        if_calib: A boolean. If True, run calibration. Not implemented yet. Settings of calibration will probably
            go to :func:`QuantDescriptor <mtpq.tensor_quant.QuantDescriptor>`.

    Raises:

    Readonly Properties:
        - axis:
        - fake_quant:
        - scale:
        - step_size:

    Mutable Properties:
        - num_bits:
        - unsigned:
        - max:
        - min
    """

    # An experimental static switch for using pytorch's native fake quantization
    # Primary usage is to export to ONNX

    def __init__(self, quant_desc=QuantDescriptor(), disabled=False, if_quant=True, if_clip=False, if_calib=False):
        """Initialize quantizer and set up required variables"""
        super(TensorQuantizer_asym, self).__init__()
        # Expand quant_desc. Use quant_desc.dict would be eaiser, but adding one-by-one explicitly gives more control
        self._num_bits = quant_desc.num_bits
        self._fake_quant = quant_desc.fake_quant
        self._axis = quant_desc.axis
        self._scale_amax = quant_desc.scale_amax
        self._learn_amax = quant_desc.learn_amax
        self._unsigned = quant_desc.unsigned
        self._narrow_range = quant_desc.narrow_range

        # self._scale = None if not quant_desc.fake_quant else 1.
        self._disabled = disabled
        self._if_quant = if_quant
        self._if_clip = False
        self._if_calib = if_calib

        if quant_desc.amax is not None:
            self.register_buffer('_amax', torch.tensor(quant_desc.amax))
        if quant_desc.amin is not None:
            self.register_buffer('_amin', torch.tensor(quant_desc.amin))
        # Clip module consumes a lot of memory, so only create it if learn_amax is True
        if self._learn_amax:
            
            if quant_desc.amax is not None :
                init_amax = quant_desc.amax
                init_amin = quant_desc.amin 
            else:
                init_amax = 1
                init_amin = -1
            self.clip = Clip(init_amin, init_amax, learn_min=True, learn_max=True)
            # It makes more sense to enable clip stage (which learns amax) if learn_amax is true
            self.enable_clip()
        if if_clip:
            self.enable_clip()

        self._calibrator = CALIB_METHOD_MAP[quant_desc.calib_method](
                num_bits=self._num_bits, axis=self._axis, unsigned=self._unsigned)

    @property
    def amin(self):
        if not hasattr(self, "_amin"):
            return None
        return self._amin


    def disable_clip(self):
        """Disable clip stage"""
        self._if_clip = False
        self.clip.clip_value_min.required_grad = False
        self.clip.clip_value_max.required_grad = False

    def enable_clip(self):
        """Enable clip stage"""
        logging.warning("Enable `clip` stage for amax learning.")
        if not self._learn_amax:
            raise ValueError("learn_amax is False. Cannot enable clip.")
        self.clip.clip_value_min.required_grad = True
        self.clip.clip_value_max.required_grad = True
        self._if_clip = True

    
    @amin.setter
    def amin(self, value):
        if value is None:
            logging.error("Setting amin no None is meaningless.")
        else:
            if isinstance(value, torch.Tensor):
                logging.warning("amin setter is not designed to take tensor.")
            if not hasattr(self, "_amin"):
                self.register_buffer('_amin', torch.tensor(value))
            else:
                value = torch.tensor(value, device=self._amin.device)
                if self._amin.shape != value.shape:
                    raise TypeError("Changing shape when setting amin is not allowed.")
                self._amin.data.copy_(value.data)


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

        #calib_amax, calib_amin = self._calibrator.compute_amax(*args, **kwargs)
        calib_temp = self._calibrator.compute_amax(*args, **kwargs)

        try:
            if calib_temp is None:
                calib_amax, calib_amin = None, None
            elif isinstance(calib_temp,torch.Tensor):
                calib_amax, calib_amin = calib_temp, -calib_temp
            elif (isinstance(calib_temp,list) or isinstance(calib_temp,tuple)) and len(calib_temp) == 2:
                calib_amax, calib_amin = calib_temp
        except BaseException as e:
            print(f'error when dealing with calib_temp ({calib_temp}) in load_calib_amax: {e}')
            exit(-1)

        if calib_amax is None:
            err_msg = "Calibrator returned None. This usually happens when calibrator hasn't seen any tensor."
            if not strict:
                logging.warning(err_msg)
                logging.warning("Set amax to NaN!")
                calib_amax = torch.tensor(math.nan)
            else:
                raise RuntimeError(err_msg + " Passing 'strict=False' to `load_calib_amax()` will ignore the error.")
        logging.warning("Load calibrated max shape={}.".format(calib_amax.shape))
        logging.log_first_n(
            logging.WARNING, "Call .cuda() if running on GPU after loading calibrated amax.", 1)
        
        if calib_amin is None:
            err_msg = "Calibrator returned None. This usually happens when calibrator hasn't seen any tensor."
            if not strict:
                logging.warning(err_msg)
                logging.warning("Set amin to NaN!")
                calib_amin = torch.tensor(math.nan)
            else:
                raise RuntimeError(err_msg + " Passing 'strict=False' to `load_calib_amax()` will ignore the error.")
        logging.warning("Load calibrated  min shape={}.".format(calib_amin.shape))
        logging.log_first_n(
            logging.WARNING, "Call .cuda() if running on GPU after loading calibrated amax.", 1)
        
        if not hasattr(self, '_amax'):
            self.register_buffer('_amax', calib_amax.data)
        else:
            self._amax.copy_(calib_amax)
            
        if not hasattr(self, '_amin'):
            self.register_buffer('_amin', calib_amin.data)
        else:
            self._amin.copy_(calib_amin)
        self.init_qat_param()

    def init_learn_amax(self):
        """Initialize learned amax from fixed amax"""
        if self._learn_amax is False:
            raise RuntimeError("Called init_learn_amax with learn_amax=False.")
        logging.warning("Load amax as initial value for amax learning!")
        if self._amax.numel() != 1:
            logging.warning("Per channel learned amax not supported. Initializing with max(amax).")
            init_amax = torch.max(self._amax)
            init_amin = torch.min(self._amin)
        else:
            init_amax = self._amax
            init_amin = self._amin
        self.clip.clip_value_min.data.copy_(init_amin.data)
        self.clip.clip_value_max.data.copy_(init_amax.data)

    def _get_max_min(self, inputs):
        """get amax from buffer or compute it dynamically."""
        if hasattr(self, '_amax'):
            amax = self._amax
        else:
            if self._axis is None:
                reduce_axis = None
            else:
                reduce_axis = []
                # Swap axis to reduce
                axis = self._axis if isinstance(self._axis, (list, tuple)) else [self._axis]
                for i in range(inputs.dim()):
                    if not i in axis:
                        reduce_axis.append(i)
            amax = quant_utils.reduce_max(inputs, axis=reduce_axis, keepdims=True).detach()
            
        if hasattr(self, '_amin'):
            amin = self._amin
        else:
            if self._axis is None:
                reduce_axis = None
            else:
                reduce_axis = []
                # Swap axis to reduce
                axis = self._axis if isinstance(self._axis, (list, tuple)) else [self._axis]
                for i in range(inputs.dim()):
                    if not i in axis:
                        reduce_axis.append(i)
            amin = quant_utils.reduce_min(inputs, axis=reduce_axis, keepdims=True).detach()
        if self._scale_amax is not None:
            amax = amax.detach() * self._scale_amax
            amin = amin.detach() * self._scale_amax

        return amax, amin

    def _fb_fake_quant(self, inputs, amax, amin):
        """Native pytorch fake quantization."""
        logging.log_first_n(logging.WARNING, "Use Pytorch's native experimental fake quantization.", 1)
        #bound = (1 << (self._num_bits - 1 + int(self._unsigned))) - 1
        if self._unsigned:
            min_bound = 0
            max_bound = (1 << self._num_bits) - 1
        else:
            max_bound = (1 << (self._num_bits - 1)) - 1
            min_bound = -max_bound - 1 
        # To be consistent with ONNX, full range is used. e.g. range is [-128, 127] in int8
        if amax.numel() == 1:
            step_size = (amax- amin) / (2.0**self.num_bits - 1)
            zero_point = max_bound - torch.round(amax/step_size)
            outputs = torch.fake_quantize_per_tensor_affine(
                inputs, step_size.item(), zero_point.int().item(),
                min_bound, max_bound)
        else:
            amax_sequeeze = amax.squeeze().detach()
            amin_sequeeze = amin.squeeze().detach()
            if len(amax_sequeeze.shape) != 1:
                raise TypeError("Pytorch's native quantization doesn't support multiple axes")
            quant_dim = list(amax.shape).index(list(amax_sequeeze.shape)[0])
            scale = (amax_sequeeze - amin_sequeeze) / (2.0**self.num_bits - 1)
            zero_point = torch.round(amin_sequeeze/scale)- min_bound
            # Set dtype of zero_points as "torch.long" for torch.1.9, and "torch.int32" for higher version
            dtype_of_zeropoints = torch.long if '1.9' in torch.__version__ else torch.int32
            outputs = torch.fake_quantize_per_channel_affine(
                inputs, scale.data, zero_point.data, quant_dim,
                min_bound, max_bound)

        return outputs
    
    def init_qat_param(self,):
        pass

    def _quant_forward(self, inputs):
        """Quantized forward pass."""
        if self._learn_amax:
            inputs = self.clip(inputs)
            amax = self.clip.clip_value_max.detach()
            amin = self.clip.clip_value_min.detach()
        else:
            amax, amin = self._get_max_min(inputs)

        if self._fake_quant:
            if not TensorQuantizer_asym.use_fb_fake_quant:
                outputs = fake_affine_tensor_quant(inputs, amin, amax, self._unsigned, self._num_bits)
            else:
                if inputs.dtype == torch.half or amax.dtype == torch.half:
                    raise Exception("Exporting to ONNX in fp16 is not supported. Please export in fp32, i.e. disable AMP.")
                outputs = self._fb_fake_quant(inputs, amax, amin)
        else:
            outputs, self._scale = tensor_quant(inputs, amax, self._num_bits, self._unsigned)

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
            # Shape is only know when it sees the first tensor
            self._calibrator.collect(inputs)

        if self._if_clip:
            if not self._learn_amax:
                raise RuntimeError("Clip without learning amax is not implemented.")
            outputs = self.clip(inputs)

        if self._if_quant:
            outputs = self._quant_forward(inputs)

        return outputs

    def _short_amax(self, fmt='.4f'):
        """Short description of amax

        Returns:
            'dynamic': if _amax is not registered
            'amax': if _amax is per-tensor
            '[min, max](size)': if _amax is per-channel
        """
        if not hasattr(self, '_amax'):
            return 'dynamic'
        if self._amax.numel() == 1:
            return '{:{fmt}}'.format(self._amax.item(), fmt=fmt)
        return '[{:{fmt}}, {:{fmt}}]({})'.format(self._amax.min().item(), self._amax.max().item(),
                                                 self._amax.numel(), fmt=fmt)
    def _short_amin(self, fmt='.4f'):
        """Short description of amom

        Returns:
            'dynamic': if _amin is not registered
            'amax': if _amin is per-tensor
            '[min, max](size)': if _amax is per-channel
        """
        if not hasattr(self, '_amax'):
            return 'dynamic'
        if self._amin.numel() == 1:
            return '{:{fmt}}'.format(self._amin.item(), fmt=fmt)
        return '[{:{fmt}}, {:{fmt}}]({})'.format(self._amin.min().item(), self._amin.max().item(),
                                                 self._amin.numel(), fmt=fmt)

    def extra_repr(self):
        if self._disabled:
            return "disabled"
        s = "{}{}bit".format("unsigned " if self._unsigned else "", self._num_bits)
        s += " narrow" if (self._narrow_range) else ""
        s += " fake" if (self._fake_quant) else ""
        s += " axis={}".format(self._axis) if self._axis is not None else " per-tensor"
        s += " amax={}".format(self._short_amax())
        s += " amin={}".format(self._short_amin())
        s += " *{}".format(self._scale_amax) if self._scale_amax else ""
        s += " learned" if (self._learn_amax) else ""
        s += " calibrator={}".format(self._calibrator.__class__.__name__) if (self._calibrator is not None) else ""
        s += " scale={}".format(self._short_scale())
        s += " quant" if (self._if_quant) else ""
        s += " clip" if (self._if_clip) else ""
        s += " calib" if (self._if_calib) else ""
        return s

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Overloaded module function

        Adds warnings during state_dict loading.
        A workaround is implemented for loading amax from checkpoint and only supports CUDA.

        Args:
            state_dict: A dict containing the state of the top level module
            prefix: A string that prefixes all of this modules state in state_dict, e.g. 'model.conv1.'
        """
        dst_has_amax = '_amax' in self._buffers
        src_has_amax = prefix + '_amax' in state_dict
        
        dst_has_amin = '_amin' in self._buffers
        src_has_amin = prefix + '_amin' in state_dict

        if not src_has_amax and dst_has_amax:
            logging.error("{}: No amax in state_dict.".format(prefix[:-1]))
        elif src_has_amax and not dst_has_amax:
            logging.debug(("{}: No '_amax' buffer to load amax into."
                           " '_amax` will be created as WAR for now. "
                           "This behavior will change in future.").format(prefix[:-1]))
            self.register_buffer("_amax", state_dict[prefix + '_amax'].data.cuda())
        elif src_has_amax and dst_has_amax:
            logging.warning("{}: Overwriting amax.".format(prefix[:-1]))
            
        if not src_has_amin and dst_has_amin:
            logging.error("{}: No amin in state_dict.".format(prefix[:-1]))
        elif src_has_amin and not dst_has_amin:
            logging.debug(("{}: No '_amin' buffer to load amin into."
                           " '_amin` will be created as WAR for now. "
                           "This behavior will change in future.").format(prefix[:-1]))
            self.register_buffer("_amin", state_dict[prefix + '_amin'].data.cuda())
        elif src_has_amin and dst_has_amin:
            logging.warning("{}: Overwriting amin.".format(prefix[:-1]))

        super(TensorQuantizer_asym, self)._load_from_state_dict(state_dict, prefix, *args, **kwargs)
