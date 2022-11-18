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


"""Abstract base class for calibrators"""
from absl import logging
import torch
from pytorch_quantization import utils as quant_utils

class _Calibrator():
    """Abstract base class of calibrators
    Args:
        num_bits: An integer. Number of bits of quantization.
        axis: A tuple. see QuantDescriptor.
        unsigned: A boolean. using unsigned quantization.

    Readonly Properties:
        axis:
    """
    def __init__(self, num_bits, axis, unsigned):
        self._num_bits = num_bits
        self._axis = axis
        self._unsigned = unsigned

    def collect(self, x):
        """Abstract method: collect tensor statistics used to compute amax

        Args:
            x: A tensor
        """
        raise NotImplementedError

    def lsq_collect(self, x):
        """Tracks the lsq-init values with float mode

        Args:
            x: A tensor

        Raises:
            RuntimeError: If amax shape changes
        """
        if not hasattr(self, '_calib_amax'):
            self._calib_amax = None
        if not hasattr(self, '_calib_iter'):
            self._calib_iter = 0
        self._calib_iter += 1

        if torch.min(x) < 0.:
            logging.log_first_n(
                logging.INFO,
                ("Calibrator encountered negative values. It shouldn't happen after ReLU. "
                 "Make sure this is the right tensor to calibrate."),
                1)
            x = x.abs()

        # Swap axis to reduce.
        axis = self._axis if isinstance(self._axis, (list, tuple)) else [self._axis]
        reduce_axis = []
        for i in range(x.dim()):
            if not i in axis:
                reduce_axis.append(i)
        local_amax = quant_utils.reduce_amax_mean(x, axis=reduce_axis).detach()
        if self._calib_amax is None:
            self._calib_amax = local_amax
        else:
            if local_amax.shape != self._calib_amax.shape:
                raise RuntimeError("amax shape changed!")
            self._calib_amax.copy_(((self._calib_amax * (self._calib_iter-1) + local_amax)/self._calib_iter).data)

    def lsq_plus_collect(self, x):
        """Tracks the minimum / maximun values with float mode, to initialize LSQ_PLUS
        Args:
            x: A tensor
        Raises:
            RuntimeError: If amax shape changes
        """
        if not hasattr(self, '_calib_amax'):
            self._calib_amax = None
        if not hasattr(self, '_calib_amin'):
            self._calib_amin = None
        if not hasattr(self, '_calib_iter'):
            self._calib_iter = 0
        self._calib_iter += 1

        local_min = x.min()
        local_max = x.max()
        if self._calib_amax is None:
            self._calib_amax = local_max
        else:
            self._calib_amax.copy_(torch.max(self._calib_amax, local_max).data)
        
        if self._calib_amin is None:
            self._calib_amin = local_min
        else:
            self._calib_amin.copy_(torch.min(self._calib_amin, local_min).data)


    def reset(self):
        """Abstract method: reset calibrator to initial state"""
        raise NotImplementedError

    def compute_amax(self, *args, **kwargs):
        """Abstract method: compute the amax from the collected data

        Returns:
            amax: a tensor
        """
        raise NotImplementedError

    def compute_amax_lsq(self):
        """Return the absolute value of all tensors collected"""
        return self._calib_amax

    def __repr__(self):
        s = "num_bits={_num_bits}"
        s += " axis={_axis}"
        s += " unsigned={_unsigned}"
        return s.format(**self.__dict__)
