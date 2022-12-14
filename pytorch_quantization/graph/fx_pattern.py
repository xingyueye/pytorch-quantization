import torch


"""For residual add block of resnet"""
class ConvBnResReluTypePattern(torch.nn.Module):
    """Create a pattern of conv2d followed by residual add

    Argeuments of each submodule are not important because we only match types for a given node
    """
    def __init__(self, lower_conv=False):
        super().__init__()
        self.lower_conv= lower_conv
        self.conv = torch.nn.Conv2d(16, 32, 3)
        self.bn = torch.nn.BatchNorm2d(32)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x, identity):
        if not self.lower_conv:
            x = self.conv(x)
        else:
            x = torch.nn.functional.conv2d(x, self.conv.weight)
        x = self.bn(x)
        x = x + identity
        x = self.relu(x)
        return x

class ConvBnResTypePattern(torch.nn.Module):
    """Create a pattern of conv2d followed by residual add

    Argeuments of each submodule are not important because we only match types for a given node
    """
    def __init__(self, lower_conv=False):
        super().__init__()
        self.lower_conv= lower_conv
        self.conv = torch.nn.Conv2d(16, 32, 3)
        self.bn = torch.nn.BatchNorm2d(32)

    def forward(self, x, identity):
        if not self.lower_conv:
            x = self.conv(x)
        else:
            x = torch.nn.functional.conv2d(x, self.conv.weight)
        x = self.bn(x)
        x = x + identity
        return x

"""SESiLU Block of EfficientNet"""
class SESiLUTypePattern(torch.nn.Module):
    def __init__(self, lower_conv=False):
        super().__init__()
        self.lower_conv = lower_conv
        self.conv_reduce = torch.nn.Conv2d(32, 8, 1)
        self.conv_expand = torch.nn.Conv2d(8,32, 1)
        self.act = torch.nn.SiLU(inplace=True)
        self.gate = torch.nn.Sigmoid()

    def forward(self, x, identity):
        if not self.lower_conv:
            x = self.conv_reduce(x)
        else:
            x = torch.nn.functional.conv2d(x, self.conv_reduce.weight)
        x = self.act(x)
        if not self.lower_conv:
            x = self.conv_expand(x)
        else:
            x = torch.nn.functional.conv2d(x, self.conv_expand.weight)
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

class MeanTypePattern(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=(2,3), keepdim=True)