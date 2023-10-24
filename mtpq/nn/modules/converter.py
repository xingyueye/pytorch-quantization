import torch
import torch.nn as nn
from mtpq import nn as quant_nn

class Converter():
    def __init__(self, quant_desc):
        self.quant_desc = quant_desc

    def convert(self, module):
        raise NotImplementedError


class Conv1dConverter(Converter):
    pass


class Conv2dConverter(Converter):
    def convert(self, module):
        in_channels = module.in_channels
        out_channels = module.out_channels
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding
        groups = module.groups
        dilation = module.dilation
        quant_conv = quant_nn.QuantConv2d(in_channels,
                                            out_channels,
                                            kernel_size,
                                            stride,
                                            padding,
                                            groups=groups,
                                            dilation=dilation,
                                            quant_desc_input = self.quant_desc.input_desc,
                                            quant_desc_weight = self.quant_desc.conv_weight_desc)

        quant_conv.weight.data.copy_(module.weight.detach())

        if module.bias is not None:
            quant_conv.bias.data.copy_(module.bias.detach())
        else:
            quant_conv.bias = None
        
        return quant_conv

class Conv3dConverter(Converter):
    pass


class ConvTranspose1dConverter(Converter):
    pass


class ConvTranspose2dConverter(Converter):
    def convert(self, module):
        in_channels = module.in_channels
        out_channels = module.out_channels
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding
        groups = module.groups
        dilation = module.dilation
        quant_convtrans = quant_nn.QuantConvTranspose2d(in_channels,
                                                    out_channels,
                                                    kernel_size,
                                                    stride,
                                                    padding,
                                                    groups=groups,
                                                    dilation=dilation,
                                                    quant_desc_input = self.quant_desc.input_desc,
                                                    quant_desc_weight = self.quant_desc.deconv_weight_desc)
        quant_convtrans.weight.data.copy_(module.weight.detach())
        if module.bias is not None:
            quant_convtrans.bias.data.copy_(module.bias.detach())
        else:
            quant_convtrans.bias = None
        
        return quant_convtrans


class ConvTranspose3dConverter(Converter):
    pass


class LinearConverter(Converter):
    def convert(self, module):
        quant_linear = quant_nn.QuantLinear(
                                    module.in_features,
                                    module.out_features,
                                    quant_desc_input = self.quant_desc.input_desc,
                                    quant_desc_weight = self.quant_desc.conv_weight_desc)
        quant_linear.weight.data.copy_(module.weight.detach())
        if module.bias is not None:
            quant_linear.bias.data.copy_(module.bias.detach())
        else:
            quant_linear.bias = None
        return quant_linear

class LSTMConverter(Converter):
    pass


class LSTMCellConverter(Converter):
    pass


class MaxPool1dConverter(Converter):
    def convert(self, module):
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding
        dilation = module.dilation
        ceil_mode = module.ceil_mode
        return_indices = module.return_indices
        quant_maxpool1d = quant_nn.QuantMaxPool1d(kernel_size,
                                                    stride,
                                                    padding,
                                                    dilation,
                                                    ceil_mode=ceil_mode,
                                                    return_indices=return_indices,
                                                    quant_desc_input = self.quant_desc.input_desc)

        return quant_maxpool1d

class MaxPool2dConverter(Converter):
    def convert(self, module):
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding
        dilation = module.dilation
        ceil_mode = module.ceil_mode
        return_indices = module.return_indices
        quant_maxpool2d = quant_nn.QuantMaxPool2d(kernel_size,
                                                    stride,
                                                    padding,
                                                    dilation,
                                                    ceil_mode=ceil_mode,
                                                    return_indices=return_indices,
                                                    quant_desc_input = self.quant_desc.input_desc)

        return quant_maxpool2d


class MaxPool3dConverter(Converter):
    def convert(self, module):
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding
        dilation = module.dilation
        ceil_mode = module.ceil_mode
        return_indices = module.return_indices
        quant_maxpool3d = quant_nn.QuantMaxPool3d(kernel_size,
                                                    stride,
                                                    padding,
                                                    dilation,
                                                    ceil_mode=ceil_mode,
                                                    return_indices=return_indices,
                                                    quant_desc_input = self.quant_desc.input_desc)

        return quant_maxpool3d


class AvgPool1dConverter(Converter):
    def convert(self, module):
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding
        ceil_mode = module.ceil_mode
        count_include_pad = module.count_include_pad
        quant_avgpool1d = quant_nn.QuantAvgPool1d(kernel_size,
                                                    stride,
                                                    padding,
                                                    ceil_mode,
                                                    count_include_pad=count_include_pad,
                                                    quant_desc_input = self.quant_desc.input_desc)

        return quant_avgpool1d

class AvgPool2dConverter(Converter):
    def convert(self, module):
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding
        ceil_mode = module.ceil_mode
        count_include_pad = module.count_include_pad
        divisor_override = module.divisor_override
        quant_avgpool2d = quant_nn.QuantAvgPool2d(kernel_size,
                                                    stride,
                                                    padding,
                                                    ceil_mode,
                                                    count_include_pad=count_include_pad,
                                                    divisor_override=divisor_override,
                                                    quant_desc_input = self.quant_desc.input_desc)

        return quant_avgpool2d


class AvgPool3dConverter(Converter):
    def convert(self, module):
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding
        ceil_mode = module.ceil_mode
        count_include_pad = module.count_include_pad
        divisor_override = module.divisor_override
        quant_avgpool3d = quant_nn.QuantAvgPool3d(kernel_size,
                                                    stride,
                                                    padding,
                                                    ceil_mode,
                                                    count_include_pad=count_include_pad,
                                                    divisor_override=divisor_override,
                                                    quant_desc_input = self.quant_desc.input_desc)

        return quant_avgpool3d

class AdaptiveAvgPool1dConverter(Converter):
    def convert(self, module):
        output_size = module.output_size
        quant_avgpool1d = quant_nn.QuantAdaptiveAvgPool1d(output_size,
                                                    quant_desc_input = self.quant_desc.input_desc)
        return quant_avgpool1d


class AdaptiveAvgPool2dConverter(Converter):
    def convert(self, module):
        output_size = module.output_size
        quant_avgpool2d = quant_nn.QuantAdaptiveAvgPool2d(output_size,
                                                    quant_desc_input = self.quant_desc.input_desc)
        return quant_avgpool2d


class AdaptiveAvgPool3dConverter(Converter):
    def convert(self, module):
        output_size = module.output_size
        quant_avgpool3d = quant_nn.QuantAdaptiveAvgPool3d(output_size,
                                                    quant_desc_input = self.quant_desc.input_desc)
        return quant_avgpool3d


class HardswishCustomConverter(Converter):
    def convert(self, module):
        quant_hardswish = quant_nn.HardswishReplace()
        return quant_hardswish

class LinearCustomConverter(Converter):
    def convert(self, module):
        quant_linear_ft = quant_nn.QuantLinearFT(
            module.in_features,
            module.out_features,
            quant_desc_input=self.quant_desc.input_desc,
            quant_desc_weight=self.quant_desc.conv_weight_desc,
            quant_desc_output=self.quant_desc.output_desc)
        quant_linear_ft.weight.data.copy_(module.weight.detach())
        if module.bias is not None:
            quant_linear_ft.bias.data.copy_(module.bias.detach())
        else:
            quant_linear_ft.bias = None
        return quant_linear_ft

class Conv2dBNFuseConverter(Converter):
    def convert(self, module):
        if not hasattr(module, 'is_bn_following') or not module.is_bn_following :
            return module
        in_channels = module.in_channels
        out_channels = module.out_channels
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding
        groups = module.groups
        dilation = module.dilation
        eps = module.follow_bn['bn_module'].eps
        momentum = module.follow_bn['bn_module'].momentum
        quant_conv2dbn_fuse = quant_nn.QuantConv2dBNFuse(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            eps=eps,
            momentum=momentum,
            quant_desc_input=self.quant_desc.input_desc,
            quant_desc_weight=self.quant_desc.conv_weight_desc)
        quant_conv2dbn_fuse.weight.data.copy_(module.weight.detach())
        if module.bias is not None:
            quant_conv2dbn_fuse.bias.data.copy_(module.bias.detach())
        else:
            quant_conv2dbn_fuse.bias = None
            

        quant_conv2dbn_fuse.bn.weight.data.copy_(module.follow_bn['bn_module'].weight.detach())
        quant_conv2dbn_fuse.bn.bias.data.copy_(module.follow_bn['bn_module'].bias.detach())
        quant_conv2dbn_fuse.bn.running_mean.data.copy_(module.follow_bn['bn_module'].running_mean.detach())
        quant_conv2dbn_fuse.bn.running_var.data.copy_(module.follow_bn['bn_module'].running_var.detach())
        
        return quant_conv2dbn_fuse

class Conv2dBNFuseInPlaceConverter(Converter):
    def _fold_bn(self,conv_module, bn_module):
        w = conv_module.weight.data
        y_mean = bn_module.running_mean
        y_var = bn_module.running_var
        safe_std = torch.sqrt(y_var + bn_module.eps)
        w_view = (conv_module.out_channels, 1, 1, 1)
        if bn_module.affine:
            weight = w * (bn_module.weight / safe_std).view(w_view)
            beta = bn_module.bias - bn_module.weight * y_mean / safe_std
            if conv_module.bias is not None:
                bias = bn_module.weight * conv_module.bias / safe_std + beta
            else:
                bias = beta
        else:
            weight = w / safe_std.view(w_view)
            beta = -y_mean / safe_std
            if conv_module.bias is not None:
                bias = conv_module.bias / safe_std + beta
            else:
                bias = beta
        return weight, bias
    def convert(self, module):
        if not hasattr(module, 'is_bn_following') or not module.is_bn_following :
            return module
        in_channels = module.in_channels
        out_channels = module.out_channels
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding
        groups = module.groups
        dilation = module.dilation
        eps = module.follow_bn['bn_module'].eps
        momentum = module.follow_bn['bn_module'].momentum
        quant_conv2dbn_fuse_inplace = quant_nn.QuantConv2dBNFuseInPlace(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            eps=eps,
            momentum=momentum,
            quant_desc_input=self.quant_desc.input_desc,
            quant_desc_weight=self.quant_desc.conv_weight_desc)
        
        fuse_weight, fuse_bias = self._fold_bn(module,module.follow_bn['bn_module'])
        quant_conv2dbn_fuse_inplace.weight.data.copy_(fuse_weight)
        quant_conv2dbn_fuse_inplace.bias.data.copy_(fuse_bias)
    
        return quant_conv2dbn_fuse_inplace