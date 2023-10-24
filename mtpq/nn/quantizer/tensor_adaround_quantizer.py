"""AdaRoundQuantizer Module"""
import math
from absl import logging

import torch
import torch.nn as nn
from mtpq.tensor_quant import QuantDescriptor
from mtpq.nn.quantizer.tensor_quantizer_asym import TensorQuantizer_asym
from mtpq import calib
__all__ = ['AdaRoundQuantizer']

CALIB_METHOD_MAP={
    "histogram": calib.HistogramCalibrator,
    "asym_histogram": calib.AsymHistogramCalibrator,
    "max": calib.MaxCalibrator,
    "minmax": calib.MinMaxCalibrator,
    "lsq": calib.LSQCalibrator
}

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

class AdaRoundQuantizer(TensorQuantizer_asym):
    """
    Adaptive Rounding Quantizer, used to optimize the rounding policy
    by reconstructing the intermediate output.
    Based on
     Up or Down? Adaptive Rounding for Post-Training Quantization: https://arxiv.org/abs/2004.10568

    :param uaq: UniformAffineQuantizer, used to initialize quantization parameters in this quantizer
    :param round_mode: controls the forward pass in this quantizer
    :param weight_tensor: initialize alpha
    """

    def __init__(self,  quant_desc=QuantDescriptor(), disabled=False, if_quant=True, if_calib=False):
        super(AdaRoundQuantizer, self).__init__()
        self.quant_desc = quant_desc
        self._num_bits = quant_desc.num_bits
        self.n_levels = 2 ** self._num_bits
        self.delta, self.zero_point = None, None
        self.eps = torch.tensor(1e-8, dtype=torch.float32)
        self.symmetry = quant_desc.symmetry
        self.method = 'percentile' if self.symmetry else None
        self._disabled = disabled
        self._if_quant = if_quant
        self._if_calib = if_calib
        self._axis = quant_desc.axis
        # params for sigmoid function
        self.alpha = None
        self.soft_targets = False
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3
        self._calibrator = CALIB_METHOD_MAP[quant_desc.calib_method](
                num_bits=self._num_bits, axis=self._axis, unsigned=self._unsigned)
        self._export_weight = None
        
    def calculate_qparams(self, min_val, max_val):
        # one_dim or one element
        quant_min, quant_max = 0, self.n_levels - 1
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        scale = torch.max(scale, self.eps)
        zero_point = quant_min - torch.round(min_val_neg / scale)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)
        return scale, zero_point
    
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
            zero_point = max_bound - torch.round(amax/step_size) if not self.symmetry else torch.tensor(0)
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
            dtype_of_zeropoints = torch.long if '1.9' in torch.__version__ else torch.int32
            zero_point = torch.round(amin_sequeeze/scale)- min_bound if not self.symmetry else torch.zeros_like(scale, dtype=dtype_of_zeropoints)
            # Set dtype of zero_points as "torch.long" for torch.1.9, and "torch.int32" for higher version
            outputs = torch.fake_quantize_per_channel_affine(
                inputs, scale.data, zero_point.data, quant_dim,
                min_bound, max_bound)

        return outputs
    
    def _quant_forward(self, inputs):
        if self._fake_quant and self.use_fb_fake_quant:
            # replace the adaquantizer    
            output = self._fb_fake_quant(self._export_weight, self.amax, self.amin)
        else:
            self.delta, self.zero_point = self.calculate_qparams(self.amin,self.amax)
            if self.axis is not None:
                new_shape = [1] * len(inputs.shape)
                new_shape[0] = inputs.shape[0]
                self.delta = self.delta.reshape(new_shape)
                self.zero_point = self.zero_point.reshape(new_shape)
            if self.alpha == None:
                x_int = round_ste(inputs / self.delta) + self.zero_point
                x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                output = (x_quant - self.zero_point) * self.delta
            else:    
                x_floor = torch.floor(inputs / self.delta)
                if self.soft_targets:
                    x_int = x_floor + self.get_soft_targets()
                else:
                    x_int = x_floor + (self.alpha >= 0).float()
                    
                x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
                output = (x_quant - self.zero_point) * self.delta  
                self._export_weight = output  
        return output
        
    def forward(self, x):
        if self._disabled:
            return x
        output = x
        if self._if_calib:
            if self._calibrator is None:
                raise RuntimeError("Calibrator was not created.")
            # Shape is only know when it sees the first tensor
            self._calibrator.collect(x)
            self.load_calib_amax(self.method)
        if self._if_quant:
            output = self._quant_forward(x)
        return output

    def get_soft_targets(self): 
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def init_alpha(self, x: torch.Tensor):
        x_floor = torch.floor(x / self.delta)
        print('Init alpha to be FP32')
        rest = (x / self.delta) - x_floor
        alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)
        self.alpha = nn.Parameter(alpha)

    @torch.jit.export
    def extra_repr(self):
        return 'bit={}'.format(self.n_bits)
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
        
        dst_has_alpha = 'alpha' in self._parameters
        src_has_alpha = prefix + 'alpha' in state_dict

        if not src_has_alpha and dst_has_alpha:
            logging.error("{}: No alpha in state_dict.".format(prefix[:-1]))
        elif src_has_alpha and not dst_has_alpha:
            logging.debug(("{}: No 'alpha' parameter to load alpha into."
                        " 'alpha` will be created as WAR for now. "
                        "This behavior will change in future.").format(prefix[:-1]))
            self.alpha = nn.Parameter(state_dict[prefix + 'alpha'].data.cuda())
        elif src_has_alpha and dst_has_alpha:
            logging.warning("{}: Overwriting alpha.".format(prefix[:-1]))

        super(TensorQuantizer_asym, self)._load_from_state_dict(state_dict, prefix, *args, **kwargs)

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
