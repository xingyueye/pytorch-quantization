import torch
Tensor = torch.Tensor

from typing import Optional, Tuple,List
import math

from torch import nn
from torch.nn import functional as F
from mtpq.nn import TensorQuantizer

from torch.nn.parameter import Parameter
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_

from mtpq import tensor_quant
from mtpq.nn.modules.quant_linear_ft import *

from . import _utils

__all__ = ['QuantMultiheadAttention']

class QuantMultiheadAttention(nn.Module, _utils.QuantMixin):
    r'''
    参数：
        embed_dim: 词嵌入的维度
        num_heads: 平行头的数量
        batch_first: 若`True`，则为(batch, seq, feture)，若为`False`，则为(seq, batch, feature)
    
    例子：
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    '''
    
    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW
    default_quant_desc_output = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, kdim=None, vdim=None,batch_first=False, **kwargs) -> None:
        super(QuantMultiheadAttention, self).__init__()
        quant_desc_input, quant_desc_weight, quant_desc_output = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)
        
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.num_heads = num_heads
        self._dropout_rate = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
                
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = QuantLinearFT(embed_dim, embed_dim, bias, quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight, quant_desc_output=quant_desc_output)
        self.k_proj = QuantLinearFT(embed_dim, self.kdim, bias, quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight, quant_desc_output=quant_desc_output)
        self.v_proj = QuantLinearFT(embed_dim, self.vdim, bias, quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight, quant_desc_output=quant_desc_output)
        self.out_proj = QuantLinearFT(embed_dim, embed_dim, bias=bias, quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight, quant_desc_output=quant_desc_output)
        
        self.dropout = nn.Dropout(self._dropout_rate)
        
        self.matmul_q_input_quantizer = TensorQuantizer(quant_desc_input)
        self.matmul_k_input_quantizer = TensorQuantizer(quant_desc_input)
        self.matmul_v_input_quantizer = TensorQuantizer(quant_desc_input)
        self.matmul_a_input_quantizer = TensorQuantizer(quant_desc_input)
        self.softmax_input_quantizer = TensorQuantizer(quant_desc_input)
        self._reset_parameters()

    def save_tmp(self):
        self._save_tmp = True
        self.out_proj.save_tmp()

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.k_proj.weight)
        xavier_uniform_(self.v_proj.weight)
        xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            constant_(self.q_proj.bias, 0.)
            constant_(self.k_proj.bias, 0.)
            constant_(self.v_proj.bias, 0.)
            constant_(self.out_proj.bias, 0.)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = torch.reshape(x, new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_key_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = torch.reshape(x, new_x_shape)
        return x.permute(0, 2, 3, 1)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        # print(f'show attn input shape: {query.shape}, {key.shape}, {value.shape}, {attn_mask.shape if attn_mask is not None else None}')
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        num_heads = self.num_heads
        head_dim = embed_dim // num_heads
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        ## mask proc...
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                attn_mask = attn_mask.to(torch.bool)
            else:
                assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                    f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"

            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz* num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
                # attn_mask=attn_mask.reshape(bsz, num_heads, tgt_len, src_len)
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)
        
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
                expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))
        #### 若attn_mask值是布尔值，则将mask转换为float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask
        
        # q = self.transpose_for_scores(q)
        # k = self.transpose_key_for_scores(k)
        # v = self.transpose_for_scores(v)
        
        # reshape q,k,v将Batch放在第一维以适合点积注意力
        # 同时为多头机制，将不同的头拼在一起组成一层
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1).transpose(-2,-1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # print(f'before qxk, score: {q.shape}, mask: {k.shape}, value: {v.shape}')

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(self.matmul_q_input_quantizer(q),
                                        self.matmul_k_input_quantizer(k))
        attention_scores = self.softmax_input_quantizer(attention_scores)

        attention_scores = attention_scores / math.sqrt(self.head_dim)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # print(f'before add mask, score: {attention_scores.shape}, mask: {attn_mask.shape}, value: {v.shape}')
        attention_scores = attention_scores + attn_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if self.training:
            attention_probs = self.dropout(attention_probs)

        attn_output = torch.matmul(self.matmul_a_input_quantizer(attention_probs),
                                   self.matmul_v_input_quantizer(v))
        
        # attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        # new_shape = attn_output.size()[:-2] + (self.embed_dim,)
        # attn_output = torch.reshape(attn_output, new_shape)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        # print(f'show shape out of self attn: {attn_output.shape}')
        return attn_output

    def load_pretrain_state_dict(self, state_dict, strict=True, assign=False):
        with torch.no_grad():
            in_proj_weight = state_dict['in_proj_weight']
            w_q, w_k, w_v = in_proj_weight.chunk(3)
            self.q_proj.weight.copy_(w_q)
            self.k_proj.weight.copy_(w_k)
            self.v_proj.weight.copy_(w_v)

            in_proj_bias = state_dict.get('in_proj_bias',None)     
            if in_proj_bias is not None:
                b_q, b_k, b_v = in_proj_bias.chunk(3)
                self.q_proj.bias.copy_(b_q)
                self.k_proj.bias.copy_(b_k)
                self.v_proj.bias.copy_(b_v)    

            out_proj_weight = state_dict['out_proj.weight']
            self.out_proj.weight.copy_(out_proj_weight)
            
            out_proj_bias = state_dict.get('out_proj.bias', None)
            if out_proj_bias is not None:
                self.out_proj.bias.copy_(out_proj_bias)