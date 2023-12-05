"""Quantized Linear"""
import torch
from torch import nn
from torch.nn import functional as F

from mtpq import tensor_quant

from . import _utils

__all__ = ["LinearFT", "QuantLinearFT"]

class QuantLinearFT(nn.Linear, _utils.QuantGemmMixin):
    """FasterTransformer Quantized version of nn.Linear

    Apply quantized linear to the incoming data, y = dequant(quant(x)quant(A)^T) + b.

    Keep Module name "LinearFT" instead of "QuantLinearFT" so that it can be easily dropped into preexisting model and load
    pretrained weights. An alias "QuantLinearFT" is defined below. The base code is a copy of nn.Linear, see detailed
    comment of original arguments there.

    Quantization descriptors are passed in in kwargs. If not presents, default_quant_desc_input
    and default_quant_desc_output and default_quant_desc_weight are used.

    Keyword Arguments:
        quant_desc_input: An instance of :class:`QuantDescriptor <mtpq.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of input.
        quant_desc_output: An instance of :class:`QuantDescriptor <mtpq.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of output.
        quant_desc_wegiht: An instance of :class:`QuantDescriptor <mtpq.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of weight.

    Raises:
        ValueError: If unsupported arguments are passed in.
        KeyError: If unsupported kwargs are passed in.

    Readonly properties:
        - input_quantizer:
        - ouptut_quantizer:
        - weight_quantizer:

    Static methods:
        - set_default_quant_desc_input: Set default_quant_desc_input
        - set_default_quant_desc_output: Set default_quant_desc_output
        - set_default_quant_desc_weight: Set default_quant_desc_weight
    """

    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_output = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW

    def __init__(self, in_features, out_features, bias=True, ft_half=True, **kwargs):
        super(QuantLinearFT, self).__init__(in_features, out_features, False)
        quant_desc_input, quant_desc_weight, quant_desc_output = _utils.pop_quant_desc_in_kwargs(self.__class__, output_pop=True, **kwargs)
        self.init_quantizer(quant_desc_input, quant_desc_weight, quant_desc_output)
        # self.bias is already registered in nn.Linear
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        self._ft_half = ft_half

    def save_tmp(self, save_prefix=''):
        self._save_tmp = True
        self._save_prefix = save_prefix

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        if hasattr(self, '_save_tmp') and self._save_tmp:
            print(f'{self._save_prefix} will save proj after input quant')
            torch.save(quant_input.detach().cpu(), f'tensor_cache/{self._save_prefix}after_quant_input.pt')
        quant_weight = self._weight_quantizer(self.weight)
        output = F.linear(quant_input, quant_weight, bias=None)
        if hasattr(self, '_save_tmp') and self._save_tmp:
            print(f'{self._save_prefix} will save proj before after gemm quant+bias, show input amax: {self._input_quantizer._get_amax(input)}, w amax: {self._weight_quantizer._get_amax(self.weight)}, will out amax: {self._aftergemm_quantizer._get_amax(output)}')
            torch.save(output.detach().cpu(), f'tensor_cache/{self._save_prefix}before_after_quant.pt')
        output = self._aftergemm_quantizer(output)
        if hasattr(self, '_save_tmp') and self._save_tmp:
            print(f'{self._save_prefix} will save proj before bias, show dtype: input: {input.dtype}, {quant_input.dtype}, weight: {self.weight.dtype}, {quant_weight.dtype}, output: {output.dtype}')
            torch.save(output.detach().cpu(), f'tensor_cache/{self._save_prefix}before_bias_output.pt')
        if self.bias is not None:
            if self._ft_half:
                if self.bias.dtype==torch.float32:
                    self.bias.data = self.bias.data.to(torch.float16)
            output = output + self.bias.to(output.dtype)
            if hasattr(self, '_save_tmp') and self._save_tmp:
                torch.save(self.bias, f'tensor_cache/{self._save_prefix}bias.pt')
        return output

LinearFT = QuantLinearFT