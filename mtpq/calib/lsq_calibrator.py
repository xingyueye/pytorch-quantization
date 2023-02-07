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


"""Calibrator that returns the absolute max of all collected tensors"""
from absl import logging
import torch

from mtpq.calib.calibrator import _Calibrator
from mtpq import utils as quant_utils

class LSQCalibrator(_Calibrator):
    """Max calibrator, tracks the maximum value globally

    Args:
        calib_desc: A MaxCalibDescriptor.
        num_bits: An integer. Number of bits of quantization.
        axis: A tuple. see QuantDescriptor.
        unsigned: A boolean. using unsigned quantization.

    Readonly Properties:
        amaxs: A list of amax. Numpy array is saved as it is likely to be used for some plot.
    """
    def __init__(self, num_bits, axis, unsigned, track_amax=False):
        super(LSQCalibrator, self).__init__(num_bits, axis, unsigned)
        self._track_amax = track_amax
        if self._track_amax:
            self._amaxs = []  # shall we have a better name?
        self._calib_amax = None

    # pylint:disable=missing-docstring
    @property
    def amaxs(self):
        return self._amaxs
    # pylint:enable=missing-docstring

    def collect(self, x):
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
        local_amax = quant_utils.reduce_abs_mean(x, axis=reduce_axis).detach()
        if self._calib_amax is None:
            self._calib_amax = local_amax
        else:
            if local_amax.shape != self._calib_amax.shape:
                raise RuntimeError("amax shape changed!")
            self._calib_amax.copy_(((self._calib_amax * (self._calib_iter-1) + local_amax)/self._calib_iter).data)

    def reset(self):
        """Reset the collected absolute max"""
        self._calib_amax = None

    def compute_amax(self, *args, **kwargs):
        """Return the absolute max of all tensors collected"""
        return self._calib_amax

    # pylint:disable=missing-docstring
    def __str__(self):
        s = "LSQCalibrator("
        s += "track_amax={_track_amax}"
        s += ")"
        return s.format(**self.__dict__)

    def __repr__(self):
        s = "LSQCalibrator("
        s += super(LSQCalibrator, self).__repr__()
        s += " calib_amax={_calib_amax}"
        s += " track_amax={_track_amax}"
        if self._track_amax:
            s += " amaxs={_amaxs}"
        s += ")"
        return s.format(**self.__dict__)
    # pylint:enable=missing-docstring
