import torch
from pytorch_quantization import nn as quant_nn

__all__ = ['ConvBnResReluTypePattern', 'SEReLUTypePattern','SESiLUTypePattern', 'DropActDropPathAddTypePattern',
           'MeanTypePattern', 'SEAvgPoolTypePattern', 'HardSigmoidTypePattern', 'BERTQueryKeyTypePattern', 'BERTAttnOutTypePattern',
           'BERTResAddTypePattern', 'FTSWINQKMatmulTypePattern','FTSWINAVMatmulTypePattern','FTSWINSoftmaxTypePattern']

"""For residual add block of resnet"""
class ConvBnResReluTypePattern(torch.nn.Module):
    """Create a pattern of conv2d followed by residual add

    Argeuments of each submodule are not important because we only match types for a given node
    """
    def __init__(self,):
        super().__init__()
        self.conv = quant_nn.QuantConv2d(16, 32, 3)
        self.bn = torch.nn.BatchNorm2d(32)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x, identity):
        x = self.conv(x)
        x = self.bn(x)
        x = x + identity
        x = self.relu(x)
        return x

"""For SEReLU of resnetrs"""
class SEReLUTypePattern(torch.nn.Module):
    """Create a pattern of conv2d followed by residual add

    Argeuments of each submodule are not important because we only match types for a given node
    """
    def __init__(self,):
        super().__init__()
        self.squeeze = quant_nn.QuantConv2d(32, 8, 1)
        self.expand = quant_nn.QuantConv2d(8,32, 1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.ident = torch.nn.Identity()

    def forward(self, x, identity, identity2):
        x = self.squeeze(x)
        x = self.ident(x)
        x = self.relu(x)
        x = self.expand(x)
        x = identity * x.sigmoid()
        x = x + identity2
        return x

"""SESiLU Block of EfficientNet"""
class SESiLUTypePattern(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.conv_reduce = quant_nn.QuantConv2d(32, 8, 1)
        self.conv_expand = quant_nn.QuantConv2d(8,32, 1)
        self.act = torch.nn.SiLU(inplace=True)
        self.gate = torch.nn.Sigmoid()

    def forward(self, x, identity):
        x = self.conv_reduce(x)
        x = self.act(x)
        x = self.conv_expand(x)
        x = identity * self.gate(x)
        return x

"""Residual Type of EfficientNet"""
class DropActDropPathAddTypePattern(torch.nn.Module):
    """Create a pattern of conv2d followed by residual add

    Argeuments of each submodule are not important because we only match types for a given node
    """
    def __init__(self,):
        super().__init__()
        self.drop = torch.nn.Identity()
        self.act = torch.nn.Identity()
        self.drop_path = torch.nn.Identity()

    def forward(self, x, identity):
        x = self.drop(x)
        x = self.act(x)
        x = self.drop_path(x)
        x = x + identity
        return x

"""MeanTypePattern of EfficientNet"""
class MeanTypePattern(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=(2,3), keepdim=True)


"""SEAvgPool Block of FANet"""
class SEAvgPoolTypePattern(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.avgpool = quant_nn.QuantAdaptiveAvgPool2d((1,1))
        self.conv_expand = quant_nn.QuantConv2d(32,32,1)
        self.bn = torch.nn.BatchNorm2d(32)
        self.gate = torch.nn.Sigmoid()

    def forward(self, x, identity):
        x = self.avgpool(x)
        x = self.conv_expand(x)
        x = self.bn(x)
        x = identity * self.gate(x)
        return x


"""HardSigmoid Block of MobileNetV3"""
class HardSigmoidTypePattern(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.gate = torch.nn.Hardsigmoid()

    def forward(self, x, identity):
        x = identity * self.gate(x)
        return x

"""QueryKey of Huggingface BERT"""
class BERTQueryKeyTypePattern(torch.nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, q, k):
        q = q.permute(0,2,1,3)
        x = torch.matmul(q, k.transpose(-1,-2))
        return x

"""AttnOut of Hugginface BERT"""
class BERTAttnOutTypePattern(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout()

    def forward(self, qk, v_trans):
        x = self.softmax(qk)
        x = self.dropout(x)
        x = torch.matmul(x, v_trans)
        return x

"""ResAdd of Hugginface BERT"""
class BERTResAddTypePattern(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.dense = quant_nn.QuantLinear(32,32)
        self.dropout = torch.nn.Dropout()

    def forward(self, x, identity):
        x = self.dense(x)
        x = self.dropout(x)
        x = x + identity
        return x

"""QK Matmul of FTSwin MHA"""
class FTSWINQKMatmulTypePattern(torch.nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, q, k):
        x = torch.matmul(q, k.transpose(-2,-1))
        return x

"""AV Matmul of FTSwin MHA"""
class FTSWINAVMatmulTypePattern(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.dropout = torch.nn.Dropout()

    def forward(self, a, v):
        x = self.dropout(a)
        x = torch.matmul(x, v)
        return x

"""Softmax of FTSwin MHA"""
class FTSWINSoftmaxTypePattern(torch.nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x):
        x = torch.nn.functional.softmax(x, dim=-1)
        return x

"""ResAdd1 of FTSwin"""
class FTSWINResAdd1TypePattern(torch.nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, identity):
        x = x + x.view(1,2,3)
        return x

"""ResAdd2 of FTSwin"""
class FTSWINResAdd2TypePattern(torch.nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x):
        x = torch.nn.functional.softmax(x, dim=-1)
        return x

