import os
import copy

import logging
import os.path

logger = logging.getLogger(__name__)

from pytorch_quantization.quant_intf import *
from pytorch_quantization.quant_partial import top1_sensitivity, fast_sensitivity, do_partial_quant
from pytorch_quantization.quant_utils import model_quant_enable, model_quant_disable, set_quantizer_by_name
from pytorch_quantization.utils.onnx_utils import remove_qdq_nodes_from_qat_onnx

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

    def disable_quantizer_by_name(self, quantizer_name_list):
        set_quantizer_by_name(self.model, quantizer_name_list, _disabled=True)

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

            return skip_layers, ori_acc, ptq_acc, partial_acc, self.quant_config.partial_ptq.sensitivity_method
        else:
            return [], ori_acc, ptq_acc, 'None', 'None'

    def export_onnx(self, data_shape, onnx_path='./', dynamic_axes=None):
        onnx_name = self.calib_weights.replace(".pt", ".onnx") if dynamic_axes is None else self.calib_weights.replace(".pt", "_dynamic.onnx")
        onnx_path = os.path.join(onnx_path, onnx_name)
        quant_model_export(self.model, onnx_path, data_shape, dynamic_axes=dynamic_axes)
        logger.info("Export QAT models with QDQ nodes as {}".format(onnx_path))
        remove_qdq_nodes_from_qat_onnx(onnx_path)


<<<<<<< HEAD
class MMLabModelQuantizer(ModelQuantizer):
    NotImplementedError
    # def calibration(self, data_loader, batch_size, save_calib_model=False):
    #     quant_model_calib_mmlab(self.model, data_loader, self.quant_config, batch_size)
    #     if save_calib_model:
    #         self._save_calib_weights()
=======
class MMlabModelQuantizer(ModelQuantizer):
    def __init__(self, model_name, model, config, calib_weights='', save_ori_model=False):
        super(MMlabModelQuantizer, self).__init__(model_name, model, config, calib_weights=calib_weights, save_ori_model=save_ori_model)

    def _quant_model_init(self, model, config, calib_weights):
        return quant_model_init_mmlab(model, config, calib_weights)

    def calibration(self, data_loader, batch_size, save_calib_model=False, custom_predict=None):
        multi_task = hasattr(data_loader.dataset, 'cumulative_sizes')
        batch_idx = None
        if multi_task:
            batch_idx = data_loader.dataset.__getattribute__('cumulative_sizes')
            batch_idx = [i // data_loader.batch_size for i in batch_idx]
        self.batch_idx = batch_idx
        quant_model_calib_timm(self.model, data_loader, self.quant_config, batch_size, self._calib_predict)
        if save_calib_model:
            self._save_calib_weights()

    def load_calib_weights(self):
        assert os.path.exists(self.calib_weights), "Calibrated weights {} does not exist, please provide correct file".format(self.calib_weights)
        state_dict = torch.load(self.calib_weights, map_location='cpu')
        if 'model' in state_dict.keys():
            self.model.load_state_dict(state_dict['model'].state_dict())
        else:
            self.model.load_state_dict(state_dict)

    def _save_calib_weights(self):
        logger.info("Save calibrated models as: {}.".format(self.calib_weights))
        torch.save(self.model.state_dict(), self.calib_weights)

    def _calib_predict(self, model, calib_dataloader, num_batches):
        total = num_batches if self.batch_idx is None else len(self.batch_idx)
        with tqdm(total=total) as pbar:
            for i, data in enumerate(calib_dataloader):
                if self.batch_idx is None:
                    data['img'] = data['img'].to('cuda:0')
                    _ = model(return_loss=False, **data)
                    pbar.update(1)
                    if i >= num_batches:
                        break
                else:
                    if i+1 in self.batch_idx:
                        data['img'] = data['img'].to('cuda:0')
                        _ = model(return_loss=False, **data)
                        pbar.update(1)
>>>>>>> add MMCls ModelQuantizer


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

class FTSWINModelQuantizer(ModelQuantizer):
    def __init__(self, model_name, model, config, calib_weights='', save_ori_model=False):
        super(FTSWINModelQuantizer, self).__init__(model_name, model, config, calib_weights=calib_weights, save_ori_model=save_ori_model)

    def _quant_model_init(self, model, config, calib_weights):
        self.swin_buffer = dict()
        # record buffer from origin model, because we will lose buffer name after torch.fx trace
        for name, buffer in model.named_buffers():
            self.swin_buffer[name] = buffer
        return quant_model_init(model, config, calib_weights, type_str='FTSWIN', do_trace=True)


    def calibration(self, data_loader, batch_size, save_calib_model=False, custom_predict=None):
        quant_model_calib_timm(self.model, data_loader, self.quant_config, batch_size, custom_predict)
        self.disable_quantizer_by_name(['layernorm_input', 'softmax_input', 'local_input', 'residual_input'])
        if save_calib_model:
            self._save_calib_weights()

    def _save_calib_weights(self):
        if not self.calib_weights:
            self.calib_weights = "{}_calib_{}_w{}a{}_{}.pth.tar".format(self.model_name,
                                                          self.quant_config.calib_data_nums,
                                                          self.quant_config.w_qscheme.bit,
                                                          self.quant_config.a_qscheme.bit,
                                                          self.quant_config.a_qscheme.quantizer_type)
        # save_calib_model(self.calib_weights, self.model)
        state_dict = self.model.state_dict()
        for name, buffer in self.swin_buffer.items():
            state_dict[name] = buffer
        torch.save({'model':state_dict}, self.calib_weights)

class ModelQuantizerFactory(object):
    @classmethod
    def get_model_quantizer(cls, type_str, *args, **kwargs):
        valid_str_list = ['', 'Timm', 'MMLab', 'BERT', 'FTSWIN']
        assert type_str in valid_str_list, 'Unsupported {}ModelQuantizer'.format(type_str)
        return eval("{}ModelQuantizer".format(type_str))(*args, **kwargs)