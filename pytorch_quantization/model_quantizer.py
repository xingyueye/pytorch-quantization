import copy


import logging
logger = logging.getLogger(__name__)

from pytorch_quantization.quant_intf import *
from pytorch_quantization.quant_partial import top1_sensitivity, fast_sensitivity, do_partial_quant
from pytorch_quantization.quant_utils import model_quant_enable, model_quant_disable

def quantable_layers_gather(model):
    quantable_layers = []
    for name, module in model.named_modules():
        if name.endswith("_quantizer"):
            layer_name = name.replace("._input_quantizer", "").replace("._weight_quantizer", "")
            if layer_name not in quantable_layers:
                quantable_layers.append(layer_name)
    return quantable_layers

class ModelQuantizer:
    def __init__(self, model_name, model, config, calib_weights='', save_ori_model=False):
        self.quant_config = parse_config(config)
        if save_ori_model:
            self.ori_model = copy.deepcopy(model)
        self.calib_weights = calib_weights
        self.model_name = model_name
        self.model = self._quant_model_init(model, self.quant_config, self.calib_weights)

    def _quant_model_init(self, model, config, calib_weights):
        return quant_model_init(model, config, calib_weights, type_str='CNN', do_trace=True)

    def calibration(self, data_loader, batch_size, save_calib_model=False, custom_predict=None):
        try:
            quant_model_calib(self.model, data_loader, self.quant_config, batch_size, custom_predict)
        except:
            raise NotImplementedError("The dataloader format of unknown plateform need to be support independently")

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

    def partial_quant(self, eval_loader, eval_func, mini_eval_loader=None):
        pass

    def eval(self, eval_loader, eval_func):
        return eval_func(eval_loader, self.model)

    def quant_enable(self):
        model_quant_enable(self.model)

    def quant_disable(self):
        model_quant_disable(self.model)

    def export(self):
        pass


class TimmModelQuantizer(ModelQuantizer):
    def __init__(self, model_name, model, config, calib_weights='', save_ori_model=False):
        super(TimmModelQuantizer, self).__init__(model_name, model, config, calib_weights=calib_weights, save_ori_model=save_ori_model)


    def calibration(self, data_loader, batch_size, save_calib_model=False, custom_predict=None):
        quant_model_calib_timm(self.model, data_loader, self.quant_config, batch_size, custom_predict)
        if save_calib_model:
            self._save_calib_weights()

    def partial_quant(self, eval_loader, eval_func, mini_eval_loader=None):
        self.quant_disable()
        ori_acc = self.eval(eval_loader, eval_func)

        self.quant_enable()
        ptq_acc = self.eval(eval_loader, eval_func)

        if ori_acc - ptq_acc > self.quant_config.partial_ptq.drop:
            if self.quant_config.partial_ptq.sensitivity_method == 'top1':
                _loader = mini_eval_loader if mini_eval_loader is not None else eval_loader
                sensitivity_list = top1_sensitivity(self.model, _loader, eval_func)
                sensitivity_list.sort(key=lambda tup: tup[1], reverse=False)
            else:
                sensitivity_list = fast_sensitivity(self.model, eval_loader, self.quant_config.partial_ptq.sensitivity_method)
                sensitivity_list.sort(key=lambda tup: tup[1], reverse=True)

            print(sensitivity_list)
            skip_layers, partial_acc = do_partial_quant(sensitivity_list,
                                                         self.model,
                                                         eval_loader,
                                                         eval_func,
                                                         ori_acc,
                                                         ptq_acc,
                                                         self.quant_config.partial_ptq.drop)

            return skip_layers, ori_acc, partial_acc, self.quant_config.partial_ptq.sensitivity_method
        else:
            return [], ori_acc, ptq_acc, 'None'

    def export_onnx(self, data_shape):
        onnx_path = self.calib_weights.replace(".pt", "_qat.onnx")
        quant_model_export(self.model, onnx_path, data_shape)
        logger.info("Export QAT models with QDQ nodes as {}".format(onnx_path))
        remove_qdq_nodes_from_qat_onnx(onnx_path)


class MMlabModelQuantizer(ModelQuantizer):
    NotImplementedError
    # def calibration(self, data_loader, batch_size, save_calib_model=False):
    #     quant_model_calib_mmlab(self.model, data_loader, self.quant_config, batch_size)
    #     if save_calib_model:
    #         self._save_calib_weights()


class BERTModelQuantizer(ModelQuantizer):
    def __init__(self, model_name, model, config, calib_weights='', save_ori_model=False):
        super(BERTModelQuantizer, self).__init__(model_name, model, config, calib_weights=calib_weights, save_ori_model=save_ori_model)

    def _quant_model_init(self, model, config, calib_weights):
        from transformers.utils.fx import symbolic_trace
        input_names = ["input_ids", "attention_mask", "token_type_ids"]
        bert_traced_model = symbolic_trace(model, input_names=input_names)
        return quant_model_init(bert_traced_model, config, calib_weights, type_str='BERT', do_trace=False)

    def calibration(self, data_loader, batch_size, save_calib_model=False, custom_predict=None):
        quant_model_calib_bert(self.model, data_loader, self.quant_config, batch_size, custom_predict)
        if save_calib_model:
            self._save_calib_weights()