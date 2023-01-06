import copy

import torch
import torch.nn as nn

import logging
logger = logging.getLogger(__name__)

from pytorch_quantization.quant_intf import *

class ModelQuantizer:
    def __init__(self, model, config, model_name=None, save_ori_model=False, calib_weights=''):
        self.quant_config = parse_config(config)
        if save_ori_model:
            self.ori_model = copy.deepcopy(model)
        self.calib_weights = calib_weights
        self.model_name = model_name if model_name else "model"
        self.model = self._init_quant_model(model)

    def _init_quant_model(self, model):
        self.model = quant_model_init(model, self.quant_config, self.calib_weights)

    def calibration(self, data_loader, batch_size, save_calib_model=False):
        try:
            quant_model_calib(self.model, data_loader, self.quant_config, batch_size)
        except:
            raise NotImplementedError("The dataloader formate of unknown plateform need to be support independently")

        if save_calib_model:
            self._save_calib_weights()

    def _save_calib_weights(self):
        if not self.calib_weights:
            self.calib_weights = "{}_calib_{}_w{}a{}_{}.pt".format(self.model_name,
                                                          self.quant_config.calib_data_nums,
                                                          self.quant_config.w_qscheme.bit,
                                                          self.quant_config.a_qscheme.bit,
                                                          self.quant_config.a_qscheme.quantizer_type)
        save_calib_model(self.calib_weights, self.model)
                                        
    def export():
        pass

class TimmModelQuantizer(ModelQuantizer):

    def calibration(self, data_loader, batch_size, save_calib_model=False):
        quant_model_calib_timm(self.model, data_loader, self.quant_config, batch_size)
        if save_calib_model:
            self._save_calib_weights()

class MMlabModelQuantizer(ModelQuantizer):
    NotImplementedError
    # def calibration(self, data_loader, batch_size, save_calib_model=False):
    #     quant_model_calib_mmlab(self.model, data_loader, self.quant_config, batch_size)
    #     if save_calib_model:
    #         self._save_calib_weights()