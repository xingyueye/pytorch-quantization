#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


"""Quantized convolution
Base code is from nn.Conv, details of Module and original argument can be found there.
Module names are intentionally kept same as unquantized version so that they can be dropped into preexisting model
easily, and load pretrained weight. Aliases with Quant prefix are defined and are encouraged to be used explicitly
when start scratch.
"""

import torch
import torch.nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.modules.conv import _ConvTransposeNd

from mtpq import tensor_quant

from . import _utils

__all__ = [
   "QuantConv2dBNFuse"
]

class _QuantConvNd(torch.nn.modules.conv._ConvNd, _utils.QuantMixin):
    """base class of quantized Conv inherited from _ConvNd

    Comments of original arguments can be found in torch.nn.modules.conv

    Arguments:
        quant_desc_input: An instance of :class:`QuantDescriptor <mtpq.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of input.
        quant_desc_weight: An instance of :class:`QuantDescriptor <mtpq.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of weight.

    Raises:
        ValueError: If unsupported arguments are passed in.

    Readonly properties:
        - input_quantizer:
        - weight_quantizer:

    Static methods:
        - set_default_quant_desc_input: Set default_quant_desc_input
        - set_default_quant_desc_weight: Set default_quant_desc_weight
    """

    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, quant_desc_input, quant_desc_weight):
        super(_QuantConvNd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                           transposed, output_padding, groups, bias, padding_mode)
        self.init_quantizer(quant_desc_input, quant_desc_weight)

    def _quant(self, input):
        """Apply quantization on input and weight

        Function called by the classes lower in the hierarchy, which actually performs the quantization before forward
        in the derivate class the particular Function.

        Arguments:
            input: in_features to quantize
        Returns:
            A tuple: (quant_in_feature, quant_weight)
        """
        quant_input = self._input_quantizer(input)
        quant_weight = self._weight_quantizer(self.weight)

        return (quant_input, quant_weight)


class QuantConv2dBNFuse(_QuantConvNd):
    """Quantized 2D conv"""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 eps=1e-05, 
                 momentum=0.1,
                 bias=True,
                 freeze_bn=False,
                 padding_mode='zeros',
                 **kwargs):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        quant_desc_input, quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)
        super(QuantConv2dBNFuse, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                                          _pair(0), groups, bias, padding_mode,
                                          quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight)

        self.bn = torch.nn.BatchNorm2d(out_channels, eps, momentum, True, True)

        if self.training:
            if freeze_bn:
                self.freeze_bn_stats()
            else:
                self.update_bn_stats()
        else:
            self.freeze_bn_stats()

    def reset_running_stats(self):
        self.bn.reset_running_stats()

    def update_bn_stats(self):
        self.freeze_bn = False
        self.bn.training = True
        return self

    def freeze_bn_stats(self):
        self.freeze_bn = True
        self.bn.training = False
        return self

    def train(self, mode=True):
        """
        Batchnorm's training behavior is using the self.training flag. Prevent
        changing it if BN is frozen. This makes sure that calling `model.train()`
        on a model with a frozen BN will behave properly.
        """
        self.training = mode
        if not self.freeze_bn:
            for module in self.children():
                module.train(mode)

    def _fuse_bn_tensor(self):
        assert self.bn.running_var is not None
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        scaled_weight = self.weight * scale_factor.reshape(weight_shape)
        # using zero bias here since the bias for original conv
        # will be added later
        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias)
            conv_bias = self.bias 
        else:
            zero_bias = torch.zeros(self.out_channels, device=scaled_weight.device)
            conv_bias = torch.zeros_like(zero_bias, device=scaled_weight.device)

        full_bias = (conv_bias - self.bn.running_mean) / running_std * self.bn.weight + self.bn.bias 
        # if self.bn.affine:
        #     full_bias = (conv_bias - self.bn.running_mean) / running_std * self.bn.weight + self.bn.bias 
        # else:
        #     full_bias = (conv_bias - self.bn.running_mean) / running_std 

        return scaled_weight, full_bias

    def forward(self, input):
        # the actual quantization happens in the next level of the class hierarchy
        quant_input = self._input_quantizer(input)
        fused_weight, fused_bias = self._fuse_bn_tensor()
        quant_weight = self._weight_quantizer(fused_weight)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            output = F.conv2d(F.pad(quant_input, expanded_padding, mode='circular'),
                              quant_weight, fused_bias, self.stride,
                              _pair(0), self.dilation, self.groups)
        else:
            output = F.conv2d(quant_input, quant_weight, fused_bias, self.stride, self.padding, self.dilation,
                              self.groups)
        return output

# Define alias with Quant prefix
Conv2dBNFuse = QuantConv2dBNFuse