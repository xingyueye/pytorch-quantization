import torch
from pytorch_quantization import nn as quant_nn

__all__ = ['ConvBnResReluTypePattern', 'SEReLUTypePattern','SESiLUTypePattern', 'DropActDropPathAddTypePattern',
           'MeanTypePattern']

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