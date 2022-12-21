import torch.fx as fx
import torch.nn as nn
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
        self.pattern = ConvBnResReluTypePattern()
        self.pattern_graph = fx_utils.LowerQuantOpTracer().trace(self.pattern)
        self.pattern_traced = fx.GraphModule(self.pattern, self.pattern_graph)

    def match_and_insert(self, model_traced, quantizer_desc):
        for node in model_traced.graph.nodes:
            if fx_utils.end_node_a_matches_graph_b_types(node, model_traced, self.pattern_graph, self.pattern_traced):
                # Add quantizer to one edge of the residual add
                print('node: ', node, node.args[0].name)

                res_add_quantizer_name = F"{'.'.join(node.args[0].name.split('.'))}.input_quantizer"
                res_add_quantizer = TensorQuantizer(quantizer_desc)
                model_traced.add_submodule(res_add_quantizer_name, res_add_quantizer)
                # The matched end node is ReLU, whose args[0] is the add node we want to add quantizer to
                fx_utils.add_quantizer(node.args[0], model_traced, (1,), (res_add_quantizer_name,))


class SESiLUTypePatternMatcher(PatternMatcher):
    def __init__(self):
        super(SESiLUTypePatternMatcher, self).__init__()
        self.pattern = SESiLUTypePattern()
        self.pattern_graph = fx_utils.LowerQuantOpTracer().trace(self.pattern)
        self.pattern_traced = fx.GraphModule(self.pattern, self.pattern_graph)

    def match_and_insert(self, model_traced, quantizer_desc):
        for node in model_traced.graph.nodes:
            if fx_utils.end_node_a_matches_graph_b_types(node, model_traced, self.pattern_graph, self.pattern_traced):
                # Add quantizer to one edge of the residual mul
                print('node: ', node, node.name, node.args[1].name)

                res_mul_quantizer_name = F"{'.'.join(node.name.split('.'))}.input_quantizer"
                res_mul_quantizer = TensorQuantizer(quantizer_desc)
                model_traced.add_submodule(res_mul_quantizer_name, res_mul_quantizer)
                # The matched end node is Mul, whose 1st input we want to add quantizer to
                fx_utils.add_quantizer(node, model_traced, (0,), (res_mul_quantizer_name,))

                # insert quantizer before sigmoid
                sigmoid_quantizer_name = F"{'.'.join(node.args[1].name.split('.'))}.input_quantizer"
                sigmoid_quantizer = TensorQuantizer(quantizer_desc)
                model_traced.add_submodule(sigmoid_quantizer_name, sigmoid_quantizer)
                # node.args[1] is sigmoid node, whose 1st input we want to add quantizer to
                fx_utils.add_quantizer(node.args[1], model_traced, (0,), (sigmoid_quantizer_name,))


class DropActDropPathAddTypePatternMatcher(PatternMatcher):
    def __init__(self):
        super(DropActDropPathAddTypePatternMatcher, self).__init__()
        self.pattern = DropActDropPathAddTypePattern()
        self.pattern_graph = fx_utils.LowerQuantOpTracer().trace(self.pattern)
        self.pattern_traced = fx.GraphModule(self.pattern, self.pattern_graph)

    def match_and_insert(self, model_traced, quantizer_desc):
        for node in model_traced.graph.nodes:
            if fx_utils.end_node_a_matches_graph_b_types(node, model_traced, self.pattern_graph, self.pattern_traced):
                print('node: ', node, node.name)

                res_add_quantizer_name = F"{'.'.join(node.name.split('.'))}.input_quantizer"
                res_add_quantizer = TensorQuantizer(quantizer_desc)
                model_traced.add_submodule(res_add_quantizer_name, res_add_quantizer)
                # The matched end node is Add, whose snd input we want to add quantizer to
                fx_utils.add_quantizer(node, model_traced, (1,), (res_add_quantizer_name,))


class MeanTypePatternMatcher(PatternMatcher):
    def __init__(self):
        super(MeanTypePatternMatcher, self).__init__()
        self.pattern = MeanTypePattern()
        self.pattern_graph = fx_utils.LowerQuantOpTracer().trace(self.pattern)
        self.pattern_traced = fx.GraphModule(self.pattern, self.pattern_graph)

    def match_and_insert(self, model_traced, quantizer_desc):
        for node in model_traced.graph.nodes:
            if fx_utils.end_node_a_matches_graph_b_types(node, model_traced, self.pattern_graph, self.pattern_traced):
                print('node: ', node, node.name)
                mean_quantizer_name = F"{'.'.join(node.name.split('.'))}.input_quantizer"
                mean_quantizer = TensorQuantizer(quantizer_desc)
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
