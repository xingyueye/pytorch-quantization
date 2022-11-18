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

from pytorch_quantization.calib.calibrator import _Calibrator
from pytorch_quantization import utils as quant_utils

class MinMaxCalibrator(_Calibrator):
    """Max calibrator, tracks the maximum and minumum value globally

    Args:
        calib_desc: A MaxCalibDescriptor.
        num_bits: An integer. Number of bits of quantization.
        axis: A tuple. see QuantDescriptor.
        unsigned: A boolean. using unsigned quantization.

    Readonly Properties:
        amaxs: A list of amax. Numpy array is saved as it is likely to be used for some plot.
        amin: A list of amin. Numpy array is saved as it is likely to be used for some plot.
    """
    def __init__(self, num_bits, axis, unsigned, track_amax=False):
        super(MinMaxCalibrator, self).__init__(num_bits, axis, unsigned)
        self._track_amax = track_amax
        if self._track_amax:
            self._amaxs = []  # shall we have a better name?
        self._calib_amax = None
        self._calib_amin = None


    # pylint:disable=missing-docstring
    @property
    def amaxs(self):
        return self._amaxs
    # pylint:enable=missing-docstring

    def collect(self, x):
        """Tracks the absolute max of all tensors

        Args:
            x: A tensor

        Raises:
            RuntimeError: If amax shape changes
        """
        if torch.min(x) > 0.:
            logging.log_first_n(
                logging.INFO,
                ("Calibrator encountered postive minor values. It shouldn't happen after Swish"),
                1)
            # x = x.abs()

        # Swap axis to reduce.
        axis = self._axis if isinstance(self._axis, (list, tuple)) else [self._axis]
        reduce_axis = []
        for i in range(x.dim()):
            if not i in axis:
                reduce_axis.append(i)
        local_amax = quant_utils.reduce_amax(x, axis=reduce_axis).detach()
        local_amin = quant_utils.reduce_amin(x, axis=reduce_axis).detach()

        if self._calib_amax is None:
            self._calib_amax = local_amax
            self._calib_amin = local_amin
        else:
            if local_amax.shape != self._calib_amax.shape:
                raise RuntimeError("amax shape changed!")
            self._calib_amax.copy_(torch.max(self._calib_amax, local_amax).data)
            self._calib_amin.copy_(torch.min(self._calib_amin, local_amin).data)

        if self._track_amax:
            self._amaxs.append(local_amax.cpu().numpy())

    def reset(self):
        """Reset the collected absolute max"""
        self._calib_amax = None

    def compute_amax(self):
        """Return the absolute max of all tensors collected"""
        return self._calib_amax, self._calib_amin

    # pylint:disable=missing-docstring
    def __str__(self):
        s = "MaxCalibrator("
        s += "track_amax={_track_amax}"
        s += ")"
        return s.format(**self.__dict__)

    def __repr__(self):
        s = "MinMaxCalibrator("
        s += super(MinMaxCalibrator, self).__repr__()
        s += " calib_amax={_calib_amax}"
        s += " calib_amin={_calib_amin}"
        s += " track_amax={_track_amax}"
        if self._track_amax:
            s += " amaxs={_amaxs}"
        s += ")"
        return s.format(**self.__dict__)
    # pylint:enable=missing-docstring
