import re
import numpy as np
import torch
import random
from absl import logging

class FTQuantArgs:
    def __init__(self, quant_mode='ft1'):
        self.set_args(quant_mode)

    def set_args(self, quant_mode):
        if quant_mode == 'ft1':
            self.weight_quant_per_tensor = False
            self.quant_disable_keyword = ['final_input', 'layernorm_input', 'softmax_input', 'residual_input', 'local_input', 'aftergemm']
            self.fuse_qkv = False
            self.narrow_range = False
        elif quant_mode == 'ft2':
            self.weight_quant_per_tensor = True
            self.quant_disable_keyword = ['final_input', 'layernorm_input', 'softmax_input', 'local_input']
            self.fuse_qkv = True
            self.narrow_range = False
        elif quant_mode == 'ft3':
            self.weight_quant_per_tensor = True
            self.quant_disable_keyword = ['final_input', 'layernorm_input', 'local_input']
            self.fuse_qkv = True
            self.narrow_range = False
        elif quant_mode == 'trt':
            # for demobert
            self.weight_quant_per_tensor = True
            self.quant_disable_keyword = ['layernorm_input', 'softmax_input', 'aftergemm']
            self.fuse_qkv = True
            self.narrow_range = False
        else:
            raise ValueError(f"wrong argument value for 'quant_mode ({quant_mode})' when setting FTQuantArgs, only support ['ft1', 'ft2', 'ft3']")
        self.quant_mode = quant_mode

def configure_model(model, quant_args):
    """Function called before the training loop."""

    if quant_args.quant_disable_keyword:
        set_quantizer_by_name(model, quant_args.quant_disable_keyword, _disabled=True)

    if quant_args.fuse_qkv:
        fuse_qkv(model, quant_args)

def fuse_qkv(model, args):
    """Adjust quantization ranges to match an implementation where the QKV projections are implemented with a single GEMM.
    Force the weight and output scale factors to match by taking the max of (Q,K,V).
    """

    def fuse3(qq, qk, qv):
        if not hasattr(qq, '_amax') or not hasattr(qk, '_amax') or not hasattr(qv, '_amax'):
            logging.warn('missing amax buffer, unable to fuse')
            return
        q = qq._amax.detach().item()
        k = qk._amax.detach().item()
        v = qv._amax.detach().item()

        amax = max(q, k, v)
        qq._amax.fill_(amax)
        qk._amax.fill_(amax)
        qv._amax.fill_(amax)
        logging.info(f'          q={q:7.4f} k={k:7.4f} v={v:7.4f} -> {amax:7.4f}')

    for name, mod in model.named_modules():
        if name.endswith('.attn'):
            logging.info(f'FUSE_QKV: {name}')
            fuse3(mod.matmul_q_input_quantizer, mod.matmul_k_input_quantizer, mod.matmul_v_input_quantizer)
            fuse3(mod.q_proj._weight_quantizer, mod.k_proj._weight_quantizer, mod.v_proj._weight_quantizer)
            fuse3(mod.q_proj._aftergemm_quantizer, mod.k_proj._aftergemm_quantizer, mod.v_proj._aftergemm_quantizer)


def set_quantizer(name, mod, quantizer, k ,v):
    """Set attributes for mod.quantizer."""

    quantizer_mod = getattr(mod, quantizer, None)
    if quantizer_mod is not None:
        assert hasattr(quantizer_mod, k)
        setattr(quantizer_mod, k, v)
    else:
        logging.warn(f'{name} has no {quantizer}')


def set_quantizers(name, mod, which='both', **kwargs):
    """Set quantizer attributes for mod."""

    s = f'Warning: changing {which} quantizers of {name}'
    for k, v in kwargs.items():
        s += (f' {k}={v}')
        if which in ['input', 'both']:
            set_quantizer(name, mod, '_input_quantizer', k, v)
        if which in ['weight', 'both']:
            set_quantizer(name, mod, '_weight_quantizer', k, v)
    logging.info(s)


def set_quantizer_by_name(model, names, **kwargs):
    """Set quantizer attributes for layers where name contains a substring in names."""

    for name, mod in model.named_modules():
        if hasattr(mod, '_input_quantizer') or hasattr(mod, '_weight_quantizer'):
            for n in names:
                if re.search(n, name):
                    set_quantizers(name, mod, **kwargs)
        elif name.endswith('_quantizer'):
            for n in names:
                if re.search(n, name):
                    s = f'Warning: changing {name}'
                    for k, v in kwargs.items():
                        s += (f' {k}={v}')
                        setattr(mod, k, v)
                    logging.info(s)