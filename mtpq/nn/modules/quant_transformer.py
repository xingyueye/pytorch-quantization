import torch
Tensor = torch.Tensor

from typing import Optional, Tuple,List
import copy

from torch import nn
from torch.nn import functional as F

from mtpq.utils.config_utils import parse_config
from mtpq import tensor_quant, quant_intf
from mtpq.nn import TensorQuantizer
from mtpq.nn.modules.quant_linear_ft import *
from mtpq.nn.modules.quant_mha import *

from . import _utils
from collections import OrderedDict

__all__ = ['construct_quant_transformer_encoder', 'QuantTransformerEncoder', 'QuantTransformerEncoderLayer']

def smooth_lm(model, hist_data, alpha):
    pass

def construct_quant_transformer_encoder(d_model, nhead, num_layers,
                                        dim_feedforward=2048, dropout=0.1, activation=F.relu, layer_norm_eps=1e-5, batch_first=False, 
                                        encoder_module=None, int8_mode=1, quant_config=None):
    if int8_mode==1:
        quant_res = False
        quant_output = False
        weight_per_channel = True
    elif int8_mode==2:
        quant_res = True
        quant_output = True
        weight_per_channel = False
    
    quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW if weight_per_channel else tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    quant_desc_output = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    
    if quant_config is not None:
        assert type(quant_config) is str, f"quant_config provided for construct_quant_transformer_encoder should be str(path of yaml file)"
        print(f'spec quant config from: {quant_config}')
        quant_config = parse_config(quant_config)
        quant_config.w_qscheme.per_channel = weight_per_channel
        quant_desc = quant_intf.get_quant_desc(quant_config)
        if 'input_desc' in quant_desc:
            quant_desc_input = quant_desc['input_desc']
        if 'conv_weight_desc' in quant_desc:
            quant_desc_weight = quant_desc['conv_weight_desc']
        if 'output_desc' in quant_desc:
            quant_desc_output = quant_desc['output_desc']
        
    q_encoder_layer = QuantTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, 
                                                   quant_res=quant_res, 
                                                   quant_desc_weight=quant_desc_weight, quant_desc_input=quant_desc_input, quant_desc_output=quant_desc_output, output_pop=True)
    q_encoder = QuantTransformerEncoder(q_encoder_layer, num_layers=num_layers, quant_output=quant_output, 
                                        quant_desc_weight=quant_desc_weight, quant_desc_input=quant_desc_input, quant_desc_output=quant_desc_output, output_pop=True)
    if encoder_module is not None:
        q_encoder.load_pretrain_state_dict(encoder_module.state_dict())
    return q_encoder

class QuantTransformerEncoder(nn.Module, _utils.QuantMixin):
    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW
    default_quant_desc_output = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    
    def __init__(self, encoder_layer, num_layers, quant_output=False, **kwargs) -> None:
        super(QuantTransformerEncoder, self).__init__()
        quant_desc_input, quant_desc_weight, quant_desc_output = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.final_quantizer = TensorQuantizer(quant_desc_input) if quant_output else None
        
    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src
        for i, layer in enumerate(self.layers):
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            if i== len(self.layers)-1 and self.final_quantizer is not None:
                output = self.final_quantizer(output)
        return output
    
    def load_pretrain_state_dict(self, state_dict, strict=True, assign=False):
        for i, layer in enumerate(self.layers):
            flag = f'layers.{i}.'
            layer_state_dict = OrderedDict({k.replace(flag, ''): state_dict[k] for k in state_dict if flag in k})
            layer.load_pretrain_state_dict(layer_state_dict, strict, assign)
            

class QuantTransformerEncoderLayer(nn.Module, _utils.QuantMixin):
    r'''
    参数：
        d_model: 词嵌入的维度
        nhead: 多头注意力中平行头的数目
        dim_feedforward: 全连接层的神经元的数目，又称经过此层输入的维度（Default = 2048）
        dropout: dropout的概率
        activation: 两个线性层中间的激活函数，默认relu或gelu
        lay_norm_eps: layer normalization中的微小量，防止分母为0
        batch_first: 若`True`，则为(batch, seq, feture)，若为`False`，则为(seq, batch, feature)

    例子：
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.randn((32, 10, 512))
        >>> out = encoder_layer(src)
    '''

    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW
    default_quant_desc_output = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, quant_res=False, **kwargs) -> None:
        super(QuantTransformerEncoderLayer, self).__init__()
        quant_desc_input, quant_desc_weight, quant_desc_output = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)
        
        self.self_attn = QuantMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, 
                                                 quant_desc_weight=quant_desc_weight, quant_desc_input=quant_desc_input, quant_desc_output=quant_desc_output, output_pop=True)
        self.self_out = QuantSelfOutput(d_model, layer_norm_eps=layer_norm_eps, dropout=dropout, quant_res=quant_res,
                                        quant_desc_weight=quant_desc_weight, quant_desc_input=quant_desc_input, quant_desc_output=quant_desc_output, output_pop=True)
        self.intermediate = QuantIntermediate(d_model, dim_feedforward, activation,
                                              quant_desc_weight=quant_desc_weight, quant_desc_input=quant_desc_input, quant_desc_output=quant_desc_output, output_pop=True)
        self.out = QuantEncoderOutput(dim_feedforward, d_model, layer_norm_eps=layer_norm_eps, drop_rate=dropout, quant_res=quant_res,
                                      quant_desc_weight=quant_desc_weight, quant_desc_input=quant_desc_input, quant_desc_output=quant_desc_output, output_pop=True)


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        attention_output = self.self_out(src2, src)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.out(intermediate_output, attention_output)
        return layer_output
    
    def load_pretrain_state_dict(self, state_dict, strict=True, assign=False):
        self_attn_flag = f'self_attn.'
        attn_state_dict = OrderedDict({k.replace(self_attn_flag, ''): state_dict[k] for k in state_dict if self_attn_flag in k})
        self.self_attn.load_pretrain_state_dict(attn_state_dict)
        
        with torch.no_grad():        
            self.self_out.norm.weight.copy_(state_dict['norm1.weight'])
            if 'norm1.bias' in state_dict:
                self.self_out.norm.bias.copy_(state_dict['norm1.bias'])
                
            self.intermediate.proj.weight.copy_(state_dict['linear1.weight'])
            if 'linear1.bias' in state_dict:
                self.intermediate.proj.bias.copy_(state_dict['linear1.bias'])
            
            self.out.proj.weight.copy_(state_dict['linear2.weight'])
            if 'linear2.bias' in state_dict:
                self.out.proj.bias.copy_(state_dict['linear2.bias'])
                
            self.out.norm.weight.copy_(state_dict['norm2.weight'])
            if 'norm2.bias' in state_dict:
                self.out.norm.bias.copy_(state_dict['norm2.bias'])
        
    
class QuantSelfOutput(nn.Module, _utils.QuantMixin):
    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW
    default_quant_desc_output = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    
    def __init__(self, embed_dim, layer_norm_eps=1e-5, dropout=0., bias=True, quant_res=False, **kwargs):
        super(QuantSelfOutput, self).__init__()
        quant_desc_input, quant_desc_weight, quant_desc_output = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)
        self.quant_res = quant_res
        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

        self.add_local_input_quantizer = TensorQuantizer(quant_desc_input)
        self.add_residual_input_quantizer = TensorQuantizer(quant_desc_input)
        self.layernorm_input_quantizer = TensorQuantizer(quant_desc_input)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dropout(hidden_states)

        add_local = self.add_local_input_quantizer(hidden_states)
        add_residual = input_tensor
        if self.quant_res:
            add_residual = self.add_residual_input_quantizer(input_tensor)
        lnorm_input = self.layernorm_input_quantizer(add_local + add_residual)
        hidden_states = self.norm(lnorm_input)
        return hidden_states


class QuantIntermediate(nn.Module, _utils.QuantMixin):
    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW
    default_quant_desc_output = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    
    def __init__(self, hidden_dim, ffn_dim, activation, **kwargs):
        super(QuantIntermediate, self).__init__()
        quant_desc_input, quant_desc_weight, quant_desc_output = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)
        self.proj = QuantLinearFT(hidden_dim, ffn_dim, quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight, quant_desc_output=quant_desc_output)
        self.activation = activation

    def forward(self, hidden_states):
        return self.activation(self.proj(hidden_states))


class QuantEncoderOutput(nn.Module, _utils.QuantMixin):
    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW
    default_quant_desc_output = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    
    def __init__(self, ffn_dim, hidden_dim, layer_norm_eps=1e-5, drop_rate=0., quant_res=False, **kwargs):
        super(QuantEncoderOutput, self).__init__()
        quant_desc_input, quant_desc_weight, quant_desc_output = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)
        self.quant_res = quant_res

        self.proj = QuantLinearFT(ffn_dim, hidden_dim, quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight, quant_desc_output=quant_desc_output)
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(drop_rate)

        self.add_local_input_quantizer = TensorQuantizer(quant_desc_input)
        self.add_residual_input_quantizer = TensorQuantizer(quant_desc_input)
        self.layernorm_input_quantizer = TensorQuantizer(quant_desc_input)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        add_local = self.add_local_input_quantizer(hidden_states)
        add_residual = input_tensor
        if self.quant_res:
            add_residual = self.add_residual_input_quantizer(input_tensor)
        lnorm_input = self.layernorm_input_quantizer(add_local + add_residual)
        hidden_states = self.norm(lnorm_input)
        return hidden_states
    