# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
# Copyright 2021 NVIDIA Corporation. All rights reserved.
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
"""
A subclass of `Trainer` specific to Question-Answering tasks
"""

import logging
import os

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from transformers import Trainer, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput

from mtpq.model_quantizer import BERTModelQuantizer


logger = logging.getLogger(__name__)

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, quant_trainer_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
        self.quant_trainer_args = quant_trainer_args
        self.calib_num = 128  # default number of calibration samples
        self.quantizer = BERTModelQuantizer("BERT_QA",
                                        self.model,
                                        self.quant_trainer_args.quant_config,
                                        self.quant_trainer_args.pretrained_calib)
        self.quantizer.model.to(self.args.device)
        self.model = self.quantizer.model
        self.model_wrapped = self.quantizer.model

    def get_calib_dataloader(self, calib_dataset=None):
        """
        Returns the calibration dataloader :class:`~torch.utils.data.DataLoader`.

        Args:
            calib_dataset (:obj:`torch.utils.data.Dataset`, `optional`)
        """
        if calib_dataset is None and self.calib_dataset is None:
            raise ValueError("Trainer: calibration requires an calib_dataset.")
        calib_dataset = calib_dataset if calib_dataset is not None else self.calib_dataset

        calib_dataset = self._remove_unused_columns(calib_dataset, description="Calibration")

        return DataLoader(
            calib_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=True,
        )

    def calib_predict(self, model, calib_dataloader, num_batches):
        for step, inputs in enumerate(calib_dataloader):
            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only=True)
            if (step + 1)  >= num_batches:
                break

    def calibrate(self, calib_dataset=None):
        calib_dataset = self.train_dataset if calib_dataset is None else calib_dataset
        calib_dataloader = self.get_calib_dataloader(calib_dataset)
        self.quantizer.calibration(calib_dataloader,
                                   calib_dataloader.batch_size,
                                   save_calib_model=True,
                                   custom_predict=self.calib_predict)

    def compute_loss(self, model, inputs, return_outputs=False):

        start_positions = inputs.pop('start_positions')
        end_positions = inputs.pop('end_positions')

        outputs = model(**inputs)

        start_logits = outputs['start_logits']
        end_logits = outputs['end_logits']
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return total_loss

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        self.args.remove_unused_columns = True
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, predict_dataset, predict_examples, ignore_keys=None, metric_key_prefix: str = "test"):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        predictions = self.post_process_function(predict_examples, predict_dataset, output.predictions, "predict")
        metrics = self.compute_metrics(predictions)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=metrics)

    def save_onnx(self, output_dir="./"):
        eval_dataset = self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        batch = next(iter(eval_dataloader))

        # saving device - to make it consistent
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # convert to tuple
        input_tuple = tuple(v.to(device) for k, v in batch.items())

        logger.info("Converting model to be onnx compatible")
        from mtpq.nn import TensorQuantizer

        TensorQuantizer.use_fb_fake_quant = True

        model = self.model.to(device)

        model.eval()
        model.float()

        model_to_save = model.module if hasattr(model, "module") else model
        # quant_trainer.configure_model(model_to_save, self.quant_trainer_args)

        output_model_file = os.path.join(output_dir, "bert_qa_calib_{}_w{}a{}_{}_qat.onnx".
                                         format(self.quantizer.quant_config.calib_data_nums,
                                                self.quantizer.quant_config.w_qscheme.bit,
                                                self.quantizer.quant_config.a_qscheme.bit,
                                                self.quantizer.quant_config.a_qscheme.quantizer_type))
        logger.info(f"exporting model to {output_model_file}")

        axes = {0: "batch_size", 1: "seq_len"}

        torch.onnx.export(
            model_to_save,
            input_tuple,
            output_model_file,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["output_start_logits", "output_end_logits"],
            dynamic_axes={
                "input_ids": axes,
                "attention_mask": axes,
                "token_type_ids": axes,
                "output_start_logits": axes,
                "output_end_logits": axes,
            },
            verbose=True,
        )
        logger.info("onnx export finished")
