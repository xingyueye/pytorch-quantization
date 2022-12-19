import torch.fx as fx
import torch.nn as nn
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn import TensorQuantizer
from pytorch_quantization.graph.fx_pattern import *
from pytorch_quantization.graph import fx_utils

class PatternMatcher(object):
    def __init__(self):
        pass

    def match_and_insert(self, model_traced, quantizer_desc):
        pass


class ConvBnResReluTypePatternMatcher(PatternMatcher):
    def __init__(self):
        super(ConvBnResReluTypePatternMatcher, self).__init__()
        self.pattern = fx.symbolic_trace(ConvBnResReluTypePattern(lower_conv=True))

    def match_and_insert(self, model_traced, **kwargs):
        for node in model_traced.graph.nodes:
            if fx_utils.end_node_a_matches_graph_b_types(node, model_traced, self.pattern.graph, self.pattern):
                # Add quantizer to one edge of the residual add
                print('node: ', node, node.args[0].name)

                res_add_quantizer_name = F"{'.'.join(node.args[0].name.split('.'))}.input_quantizer"
                res_add_quantizer = TensorQuantizer(QuantDescriptor(num_bits=kwargs['num_bits'],
                                                                    calib_method=kwargs['method']))
                model_traced.add_submodule(res_add_quantizer_name, res_add_quantizer)
                # The matched end node is ReLU, whose args[0] is the add node we want to add quantizer to
                fx_utils.add_quantizer(node.args[0], model_traced, (1,), (res_add_quantizer_name,))

class SESiLUTypePatternMatcher(PatternMatcher):
    def __init__(self):
        super(SESiLUTypePatternMatcher, self).__init__()
        self.pattern = fx.symbolic_trace(SESiLUTypePattern(lower_conv=True))

    def match_and_insert(self, model_traced, **kwargs):
        for node in model_traced.graph.nodes:
            if fx_utils.end_node_a_matches_graph_b_types(node, model_traced, self.pattern.graph, self.pattern):
                # Add quantizer to one edge of the residual mul
                print('node: ', node, node.name, node.args[1].name)

                res_mul_quantizer_name = F"{'.'.join(node.name.split('.'))}.input_quantizer"
                res_mul_quantizer = TensorQuantizer(QuantDescriptor(num_bits=kwargs['num_bits'],
                                                                    calib_method=kwargs['method']))
                model_traced.add_submodule(res_mul_quantizer_name, res_mul_quantizer)
                # The matched end node is Mul, whose 1st input we want to add quantizer to
                fx_utils.add_quantizer(node, model_traced, (0,), (res_mul_quantizer_name,))

                # insert quantizer before sigmoid
                sigmoid_quantizer_name = F"{'.'.join(node.args[1].name.split('.'))}.input_quantizer"
                sigmoid_quantizer = TensorQuantizer(QuantDescriptor(num_bits=kwargs['num_bits'],
                                                                    calib_method=kwargs['method']))
                model_traced.add_submodule(sigmoid_quantizer_name, sigmoid_quantizer)
                # node.args[1] is sigmoid node, whose 1st input we want to add quantizer to
                fx_utils.add_quantizer(node.args[1], model_traced, (0,), (sigmoid_quantizer_name,))


class DropActDropPathAddTypePatternMatcher(PatternMatcher):
    def __init__(self):
        super(DropActDropPathAddTypePatternMatcher, self).__init__()
        self.pattern = fx.symbolic_trace(DropActDropPathAddTypePattern())

    def match_and_insert(self, model_traced, **kwargs):
        for node in model_traced.graph.nodes:
            if fx_utils.end_node_a_matches_graph_b_types(node, model_traced, self.pattern.graph, self.pattern):
                print('node: ', node, node.name)

                res_add_quantizer_name = F"{'.'.join(node.name.split('.'))}.input_quantizer"
                res_add_quantizer = TensorQuantizer(QuantDescriptor(num_bits=kwargs['num_bits'],
                                                                    calib_method=kwargs['method']))
                model_traced.add_submodule(res_add_quantizer_name, res_add_quantizer)
                # The matched end node is Add, whose snd input we want to add quantizer to
                fx_utils.add_quantizer(node, model_traced, (1,), (res_add_quantizer_name,))


class MeanTypePatternMatcher(PatternMatcher):
    def __init__(self):
        super(MeanTypePatternMatcher, self).__init__()
        self.pattern = fx.symbolic_trace(MeanTypePattern())

    def match_and_insert(self, model_traced, **kwargs):
        for node in model_traced.graph.nodes:
            if fx_utils.end_node_a_matches_graph_b_types(node, model_traced, self.pattern.graph, self.pattern):
                print('node: ', node, node.name)
                mean_quantizer_name = F"{'.'.join(node.name.split('.'))}.input_quantizer"
                mean_quantizer = TensorQuantizer(QuantDescriptor(num_bits=kwargs['num_bits'],
                                                                 calib_method=kwargs['method']))
                model_traced.add_submodule(mean_quantizer_name, mean_quantizer)

                # The matched end node is mean, whose snd input we want to add quantizer to
                fx_utils.add_quantizer(node, model_traced, (0,), (mean_quantizer_name,))


def get_internal_pattern_matcher():
    pattern_matchers = list()
    pattern_matchers.append(ConvBnResReluTypePatternMatcher())
    pattern_matchers.append(SESiLUTypePatternMatcher())
    pattern_matchers.append(DropActDropPathAddTypePatternMatcher())
    pattern_matchers.append(MeanTypePatternMatcher())
    return pattern_matchers


class OpMatcher(object):
    def __init__(self):
        pass

    def match_and_insert(self, model_traced, input_quantizer_desc, weight_quanatizer_desc):
        pass

_OPS_QUANT_INPUT_AND_WEIGHTS = [
    nn.functional.conv1d,
    nn.functional.conv2d,
    nn.functional.conv3d,
    nn.functional.linear,
]

_OPS_QUANT_INPUT_AND_WEIGHTS_TRANS = [
    nn.functional.conv_transpose1d,
    nn.functional.conv_transpose2d,
    nn.functional.conv_transpose3d
]

_OPS_QUANT_INPUT_ONLY = [

    nn.functional.avg_pool1d,
    nn.functional.avg_pool2d,
    nn.functional.avg_pool3d,
    nn.functional.adaptive_avg_pool1d,
    nn.functional.adaptive_avg_pool2d,
    nn.functional.adaptive_avg_pool3d,
    nn.functional.max_pool1d,
    nn.functional.max_pool2d,
    nn.functional.max_pool3d
]

class QuantInputWeightOpMatcher(OpMatcher):
    def __init__(self):
        super(QuantInputWeightOpMatcher, self).__init__()
        self.ops = _OPS_QUANT_INPUT_AND_WEIGHTS
    def match_and_insert(self, model_traced, **kwargs):
        for node in model_traced.graph.nodes:
            if node.target in self.ops:
                layer_name = ".".join(node.args[1].name.split("_")[:-1])
                input_quantizer_name = F"{layer_name}.input_quantizer"
                weight_quantizer_name = F"{layer_name}.weight_quantizer"
                input_quantizer = TensorQuantizer(QuantDescriptor(num_bits=kwargs['num_bits'],
                                                                  calib_method=kwargs['method']))
                weight_quantizer = TensorQuantizer(QuantDescriptor(num_bits=kwargs['num_bits'],
                                                                   calib_method='max',
                                                                   axis=0))
                # Must add quantizer module first before creating call_module node
                model_traced.add_submodule(input_quantizer_name, input_quantizer)
                model_traced.add_submodule(weight_quantizer_name, weight_quantizer)
                fx_utils.add_quantizer(node, model_traced, (0, 1), (input_quantizer_name, weight_quantizer_name))

class QuantInputWeightTransOpMatcher(OpMatcher):
    def __init__(self):
        super(QuantInputWeightTransOpMatcher, self).__init__()
        self.ops = _OPS_QUANT_INPUT_AND_WEIGHTS_TRANS
    def match_and_insert(self, model_traced, **kwargs):
        for node in model_traced.graph.nodes:
            if node.target in self.ops:
                layer_name = ".".join(node.args[1].name.split("_")[:-1])
                input_quantizer_name = F"{layer_name}.input_quantizer"
                weight_quantizer_name = F"{layer_name}.weight_quantizer"
                input_quantizer = TensorQuantizer(QuantDescriptor(num_bits=kwargs['num_bits'],
                                                                  calib_method=kwargs['method']))
                weight_quantizer = TensorQuantizer(QuantDescriptor(num_bits=kwargs['num_bits'],
                                                                   calib_method='max',
                                                                   axis=1))
                # Must add quantizer module first before creating call_module node
                model_traced.add_submodule(input_quantizer_name, input_quantizer)
                model_traced.add_submodule(weight_quantizer_name, weight_quantizer)
                fx_utils.add_quantizer(node, model_traced, (0, 1), (input_quantizer_name, weight_quantizer_name))

class QuantInputOnlyOpMatcher(OpMatcher):
    def __init__(self):
        super(QuantInputOnlyOpMatcher, self).__init__()
        self.ops = _OPS_QUANT_INPUT_ONLY
    def match_and_insert(self, model_traced, **kwargs):
        for node in model_traced.graph.nodes:
            if node.target in self.ops:
                input_quantizer_name = F"{node.target}.input_quantizer"
                input_quantizer = TensorQuantizer(QuantDescriptor(num_bits=kwargs['num_bits'],
                                                                  calib_method=kwargs['method']))
                model_traced.add_submodule(input_quantizer_name, input_quantizer)
                fx_utils.add_quantizer(node, model_traced, (0,), (input_quantizer_name,))

def get_internal_op_matcher():
    op_matchers = list()
    op_matchers.append(QuantInputWeightOpMatcher())
    op_matchers.append(QuantInputWeightTransOpMatcher())
    op_matchers.append(QuantInputOnlyOpMatcher())
    return op_matchers


