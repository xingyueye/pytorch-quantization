import os
import numpy as np
import torch
import tensorrt as trt

CalibAlgorithmDict = \
{
    "entropy2": trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2,
    "minmax": trt.CalibrationAlgoType.MINMAX_CALIBRATION
}

class EngineCalibrator(trt.IInt8Calibrator):
    """
    Implements the INT8 Entropy Calibrator 2.
    """

    def __init__(self, calib_dataset, calib_batchsize, calib_num_images, calib_algo, calib_dtype='fp32', calib_file=''):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.calib_dataset = calib_dataset
        self.calib_batchsize = calib_batchsize
        self.calib_algo = CalibAlgorithmDict[calib_algo]
        self.calib_dtype = calib_dtype
        self.calib_file = calib_file
        self.batch_holder = None

        calib_num_images = min(calib_num_images, len(self.calib_dataset))
        # Subdivide the list of images into batches
        self.num_batches = calib_num_images // self.calib_batchsize
        assert self.num_batches > 0, "dataset length is {}, calib_batchsize is {}".\
            format(len(self.calib_dataset), self.calib_batchsize)
        # Indices
        self.image_index = 0
        self.batch_index = 0
        # self.cache_file = "quantization.cache"

    def get_batch_size(self):
        return self.calib_batchsize

    def get_algorithm(self):
        return self.calib_algo

    def get_batch(self, names):
        if self.batch_index < self.num_batches:
            batch_list = []
            for _ in range(self.calib_batchsize):
                batch_list.append(self.calib_dataset[self.image_index][np.newaxis,:])
                self.image_index += 1
            self.batch_index += 1
            batch_data = torch.concat(batch_list, axis=0)
            self.batch_holder = batch_data.contiguous().to('cuda') if self.calib_dtype == 'fp32' \
                                else batch_data.contiguous().to('cuda').half()
            # print(self.batch_holder.size())
            # print(self.batch_holder)
            return [int(self.batch_holder.data_ptr())]
        else:
            return None

    def read_calibration_cache(self):
        if os.path.exists(self.calib_file):
            with open(self.calib_file, "rb") as f:
                return f.read()
        else:
            return None

    def write_calibration_cache(self, cache):
        with open(self.calib_file, "wb") as f:
            f.write(cache)
