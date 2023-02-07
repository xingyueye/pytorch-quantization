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

import torch
from torch import nn
from mtpq.nn import functional as QF

__all__ = ['GradScale']

class GradScale(nn.Module):
    """Scale Gradient

    Args:
        num_bits: An integer. Number of bits of quantization. It is used to calculate scaling factor. Default 8.
        unsigned: A boolean. Use unsigned integer range. E.g. [0, 255] for num_bits=8. Default False.
    Raises:
        ValueError:
    """

    def __init__(self, num_bits=8, unsigned=False, use_sqrt=False):
        super(GradScale, self).__init__()
        self.max_bound = 2.0 ** (num_bits - 1 + int(unsigned)) - 1.0
        self._is_init = False
        self.sqrt_scale = use_sqrt

    def init_grad_scale(self, inputs, scale, device):
        # numel = inputs.numel() if len(scale.size()) > 0 else inputs.numel() // inputs.size()[0]
        numel = inputs.numel()
        self._grad_scale = torch.tensor(1.0 / ((self.max_bound * numel) ** 0.5), device=device)
        self._is_init = True

    def forward(self, inputs, scale):
        if not self._is_init:
            self.init_grad_scale(inputs, scale, inputs.device)
        scale = scale ** 2 if self.sqrt_scale is True else scale.abs()
        outputs = QF.scalegrad(scale, self._grad_scale)
        return outputs, outputs * self.max_bound
