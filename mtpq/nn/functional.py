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

"""Some supportive functions"""
from absl import logging

import torch
from torch.autograd import Function


class ClipFunction(Function):
    """An universal tensor clip function

    Pytorch's clamp() only supports scalar range and doesn't support broadcast. This implementation uses min/max which
    is more genaral. The gradient is defined according to IBM's PACT paper https://arxiv.org/abs/1805.06085, which is
    also the behavior of Tensorflow's clip_by_value()
    """

    @staticmethod
    def forward(ctx, input, clip_value_min, clip_value_max):
        output = torch.min(input, clip_value_max)
        output = torch.max(output, clip_value_min)
        ctx.save_for_backward(input, clip_value_min, clip_value_max)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_value_min, clip_value_max = ctx.saved_tensors
        min_mask = (input > clip_value_min).to(grad_output.dtype)
        max_mask = (input < clip_value_max).to(grad_output.dtype)
        grad_input = grad_output * min_mask * max_mask

        if clip_value_min.requires_grad or clip_value_max.requires_grad:
            logging.log_first_n(logging.WARNING, "Learning clip min/max is experimental, use at your own risk :).", 1)
        if clip_value_min.numel() != 1 or clip_value_max.numel() != 1:
            raise ValueError("Learnable min/max can only be scalar, got size %s and %s." % (clip_value_min.size(),
                                                                                            clip_value_max.size()))

        # Ensure the dtypes of min/max grads matches the input dtype
        # This might be necessary if running w/ AMP which will cast to fp32 before `sum()`
        grad_clip_value_min = (grad_output * (1. - min_mask)).sum().to(clip_value_min.dtype) if clip_value_min.requires_grad else None
        grad_clip_value_max = (grad_output * (1. - max_mask)).sum().to(clip_value_min.dtype) if clip_value_max.requires_grad else None

        return grad_input, grad_clip_value_min, grad_clip_value_max

class GradScaleFunction(Function):
    """An universal gradient scale function

    Multiple grad_scales to gradient for learnable _scale. This is used to balance the parameter updating between
    conv-weights and quantization-scales
    """

    @staticmethod
    def forward(ctx, input, grad_scale):
        ctx.save_for_backward(input, grad_scale)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, grad_scale = ctx.saved_tensors
        grad_input = grad_output * grad_scale
        return grad_input, None

clip = ClipFunction.apply
scalegrad = GradScaleFunction.apply
