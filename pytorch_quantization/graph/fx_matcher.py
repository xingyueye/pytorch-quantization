import torch.fx as fx
from pytorch_quantization.nn import TensorQuantizer, LSQTensorQuantizer, StableLSQTensorQuantizer, LSQPlusTensorQuantizer
from pytorch_quantization.graph.fx_pattern import *
from pytorch_quantization.graph import fx_utils

FX_TENSOR_QUANT_MAP = {
    "naive": TensorQuantizer,
    "lsq": LSQTensorQuantizer,
    "stable_lsq": StableLSQTensorQuantizer,
    "lsq_plus": LSQPlusTensorQuantizer
}

class PatternMatcher(object):
    def __init__(self):
        pass

    def match_and_insert(self, model_traced, quantizer_desc):
        pass


class ConvBnResReluTypePatternMatcher(PatternMatcher):
    def __init__(self):
        super(ConvBnResReluTypePatternMatcher, self).__init__()
        self.pattern = ConvBnResReluTypePattern()
        self.pattern_graph = fx_utils.CNNLowerQuantOpTracer().trace(self.pattern)
        self.pattern_traced = fx.GraphModule(self.pattern, self.pattern_graph)

    def match_and_insert(self, model_traced, quantizer_desc):
        for node in model_traced.graph.nodes:
            if fx_utils.end_node_a_matches_graph_b_types(node, model_traced, self.pattern_graph, self.pattern_traced):
                # Add quantizer to one edge of the residual add
                print('node: ', node, node.args[0].name)

                res_add_quantizer_name = F"{'.'.join(node.args[0].name.split('.'))}._input_quantizer"
                res_add_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(res_add_quantizer_name, res_add_quantizer)
                # The matched end node is ReLU, whose args[0] is the add node we want to add quantizer to
                fx_utils.add_quantizer(node.args[0], model_traced, (1,), (res_add_quantizer_name,))

class SEReLUTypePatternMatcher(PatternMatcher):
    def __init__(self):
        super(SEReLUTypePatternMatcher, self).__init__()
        self.pattern = SEReLUTypePattern()
        self.pattern_graph = fx_utils.CNNLowerQuantOpTracer().trace(self.pattern)
        self.pattern_traced = fx.GraphModule(self.pattern, self.pattern_graph)

    def match_and_insert(self, model_traced, quantizer_desc):
        for node in model_traced.graph.nodes:
            if fx_utils.end_node_a_matches_graph_b_types(node, model_traced, self.pattern_graph, self.pattern_traced):
                # End Node is Add
                print('node: ', node, node.name, node.args[0].name)

                res_add_quantizer_name = F"{'.'.join(node.name.split('.'))}._input_quantizer"
                res_add_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(res_add_quantizer_name, res_add_quantizer)
                # The matched end node is Add, whose 2nd input we want to add quantizer to
                fx_utils.add_quantizer(node, model_traced, (1,), (res_add_quantizer_name,))
                # insert quantizer to mul identity branch
                mul_quantizer_name = F"{'.'.join(node.args[0].name.split('.'))}._input_quantizer"
                mul_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(mul_quantizer_name, mul_quantizer)
                fx_utils.add_quantizer(node.args[0], model_traced, (0,), (mul_quantizer_name,))
                # insert quantizer before sigmoid
                sigmoid_node = node.args[0].args[1]
                sigmoid_quantizer_name = F"{'.'.join(sigmoid_node.name.split('.'))}._input_quantizer"
                sigmoid_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(sigmoid_quantizer_name, sigmoid_quantizer)
                # node.args[1] is sigmoid node, whose 1st input we want to add quantizer to
                fx_utils.add_quantizer(sigmoid_node, model_traced, (0,), (sigmoid_quantizer_name,))


class SESiLUTypePatternMatcher(PatternMatcher):
    def __init__(self):
        super(SESiLUTypePatternMatcher, self).__init__()
        self.pattern = SESiLUTypePattern()
        self.pattern_graph = fx_utils.CNNLowerQuantOpTracer().trace(self.pattern)
        self.pattern_traced = fx.GraphModule(self.pattern, self.pattern_graph)

    def match_and_insert(self, model_traced, quantizer_desc):
        for node in model_traced.graph.nodes:
            if fx_utils.end_node_a_matches_graph_b_types(node, model_traced, self.pattern_graph, self.pattern_traced):
                # Add quantizer to one edge of the residual mul
                print('node: ', node, node.name, node.args[1].name)

                res_mul_quantizer_name = F"{'.'.join(node.name.split('.'))}._input_quantizer"
                res_mul_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(res_mul_quantizer_name, res_mul_quantizer)
                # The matched end node is Mul, whose 1st input we want to add quantizer to
                fx_utils.add_quantizer(node, model_traced, (0,), (res_mul_quantizer_name,))

                # insert quantizer before sigmoid
                sigmoid_quantizer_name = F"{'.'.join(node.args[1].name.split('.'))}._input_quantizer"
                sigmoid_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(sigmoid_quantizer_name, sigmoid_quantizer)
                # node.args[1] is sigmoid node, whose 1st input we want to add quantizer to
                fx_utils.add_quantizer(node.args[1], model_traced, (0,), (sigmoid_quantizer_name,))


class DropActDropPathAddTypePatternMatcher(PatternMatcher):
    def __init__(self):
        super(DropActDropPathAddTypePatternMatcher, self).__init__()
        self.pattern = DropActDropPathAddTypePattern()
        self.pattern_graph = fx_utils.CNNLowerQuantOpTracer().trace(self.pattern)
        self.pattern_traced = fx.GraphModule(self.pattern, self.pattern_graph)

    def match_and_insert(self, model_traced, quantizer_desc):
        for node in model_traced.graph.nodes:
            if fx_utils.end_node_a_matches_graph_b_types(node, model_traced, self.pattern_graph, self.pattern_traced):
                print('node: ', node, node.name)

                res_add_quantizer_name = F"{'.'.join(node.name.split('.'))}._input_quantizer"
                res_add_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(res_add_quantizer_name, res_add_quantizer)
                # The matched end node is Add, whose snd input we want to add quantizer to
                fx_utils.add_quantizer(node, model_traced, (1,), (res_add_quantizer_name,))


class MeanTypePatternMatcher(PatternMatcher):
    def __init__(self):
        super(MeanTypePatternMatcher, self).__init__()
        self.pattern = MeanTypePattern()
        self.pattern_graph = fx_utils.CNNLowerQuantOpTracer().trace(self.pattern)
        self.pattern_traced = fx.GraphModule(self.pattern, self.pattern_graph)

    def match_and_insert(self, model_traced, quantizer_desc):
        for node in model_traced.graph.nodes:
            if fx_utils.end_node_a_matches_graph_b_types(node, model_traced, self.pattern_graph, self.pattern_traced):
                print('node: ', node, node.name)
                mean_quantizer_name = F"{'.'.join(node.name.split('.'))}._input_quantizer"
                mean_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(mean_quantizer_name, mean_quantizer)

                # The matched end node is mean, whose snd input we want to add quantizer to
                fx_utils.add_quantizer(node, model_traced, (0,), (mean_quantizer_name,))


class SEAvgPoolTypePatternMatcher(PatternMatcher):
    def __init__(self):
        super(SEAvgPoolTypePatternMatcher, self).__init__()
        self.pattern = SEAvgPoolTypePattern()
        self.pattern_graph = fx_utils.CNNLowerQuantOpTracer().trace(self.pattern)
        self.pattern_traced = fx.GraphModule(self.pattern, self.pattern_graph)

    def match_and_insert(self, model_traced, quantizer_desc):
        for node in model_traced.graph.nodes:
            if fx_utils.end_node_a_matches_graph_b_types(node, model_traced, self.pattern_graph, self.pattern_traced):
                # Add quantizer to one edge of the residual mul
                print('node: ', node, node.name, node.args[1].name)

                res_mul_quantizer_name = F"{'.'.join(node.name.split('.'))}._input_quantizer"
                res_mul_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(res_mul_quantizer_name, res_mul_quantizer)
                # The matched end node is Mul, whose 1st input we want to add quantizer to
                fx_utils.add_quantizer(node, model_traced, (0,), (res_mul_quantizer_name,))

                # insert quantizer before sigmoid
                sigmoid_quantizer_name = F"{'.'.join(node.args[1].name.split('.'))}._input_quantizer"
                sigmoid_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(sigmoid_quantizer_name, sigmoid_quantizer)
                # node.args[1] is sigmoid node, whose 1st input we want to add quantizer to
                fx_utils.add_quantizer(node.args[1], model_traced, (0,), (sigmoid_quantizer_name,))


class HardSigmoidTypePatternMatcher(PatternMatcher):
    def __init__(self):
        super(HardSigmoidTypePatternMatcher, self).__init__()
        self.pattern = HardSigmoidTypePattern()
        self.pattern_graph = fx_utils.CNNLowerQuantOpTracer().trace(self.pattern)
        self.pattern_traced = fx.GraphModule(self.pattern, self.pattern_graph)

    def match_and_insert(self, model_traced, quantizer_desc):
        for node in model_traced.graph.nodes:
            if fx_utils.end_node_a_matches_graph_b_types(node, model_traced, self.pattern_graph, self.pattern_traced):
                # Add quantizer to one edge of the residual mul
                print('node: ', node, node.name, node.args[1].name)

                res_mul_quantizer_name = F"{'.'.join(node.name.split('.'))}._input_quantizer"
                res_mul_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(res_mul_quantizer_name, res_mul_quantizer)
                # The matched end node is Mul, whose 1st input we want to add quantizer to
                fx_utils.add_quantizer(node, model_traced, (0,), (res_mul_quantizer_name,))

                # insert quantizer before sigmoid
                hardsigmoid_quantizer_name = F"{'.'.join(node.args[1].name.split('.'))}._input_quantizer"
                hardsigmoid_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(hardsigmoid_quantizer_name, hardsigmoid_quantizer)
                # node.args[1] is sigmoid node, whose 1st input we want to add quantizer to
                fx_utils.add_quantizer(node.args[1], model_traced, (0,), (hardsigmoid_quantizer_name,))


class BERTQueryKeyTypePatternMatcher(PatternMatcher):
    def __init__(self):
        super(BERTQueryKeyTypePatternMatcher, self).__init__()
        self.pattern = BERTQueryKeyTypePattern()
        self.pattern_graph = fx_utils.BERTLowerQuantOpTracer().trace(self.pattern)
        self.pattern_graph.print_tabular()
        self.pattern_traced = fx.GraphModule(self.pattern, self.pattern_graph)

    def match_and_insert(self, model_traced, quantizer_desc):
        for node in model_traced.graph.nodes:
            if fx_utils.end_node_a_matches_graph_b_types(node, model_traced, self.pattern_graph, self.pattern_traced):
                # Add quantizers to two input branches
                print('node: ', node, node.name, node.args[0].name, node.args[1].name)

                query_quantizer_name = F"{'.'.join(node.name.split('.'))}_query._input_quantizer"
                query_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(query_quantizer_name, query_quantizer)
                key_quantizer_name = F"{'.'.join(node.name.split('.'))}_key._input_quantizer"
                key_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(query_quantizer_name, query_quantizer)
                model_traced.add_submodule(key_quantizer_name, key_quantizer)

                fx_utils.add_quantizer(node, model_traced, [0, 1], [query_quantizer_name, key_quantizer_name])

class BERTAttnOutTypePatternMatcher(PatternMatcher):
    def __init__(self):
        super(BERTAttnOutTypePatternMatcher, self).__init__()
        self.pattern = BERTAttnOutTypePattern()
        self.pattern_graph = fx_utils.BERTLowerQuantOpTracer().trace(self.pattern)
        self.pattern_graph.print_tabular()
        self.pattern_traced = fx.GraphModule(self.pattern, self.pattern_graph)

    def match_and_insert(self, model_traced, quantizer_desc):
        for node in model_traced.graph.nodes:
            if fx_utils.end_node_a_matches_graph_b_types(node, model_traced, self.pattern_graph, self.pattern_traced):
                # Add quantizer to two input branches
                print('node: ', node, node.name, node.args[0].name, node.args[1].name)

                attn_quantizer_name = F"{'.'.join(node.name.split('.'))}_attn._input_quantizer"
                attn_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(attn_quantizer_name, attn_quantizer)
                value_quantizer_name = F"{'.'.join(node.name.split('.'))}_value._input_quantizer"
                value_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(attn_quantizer_name, attn_quantizer)
                model_traced.add_submodule(value_quantizer_name, value_quantizer)

                fx_utils.add_quantizer(node, model_traced, [0, 1], [attn_quantizer_name, value_quantizer_name])


class BERTResAddTypePatternMatcher(PatternMatcher):
    def __init__(self):
        super(BERTResAddTypePatternMatcher, self).__init__()
        self.pattern = BERTResAddTypePattern()
        self.pattern_graph = fx_utils.BERTLowerQuantOpTracer().trace(self.pattern)
        self.pattern_graph.print_tabular()
        self.pattern_traced = fx.GraphModule(self.pattern, self.pattern_graph)

    def match_and_insert(self, model_traced, quantizer_desc):
        for node in model_traced.graph.nodes:
            if fx_utils.end_node_a_matches_graph_b_types(node, model_traced, self.pattern_graph, self.pattern_traced):
                # Add quantizer to identity branch
                print('node: ', node, node.name, node.args[0].name, node.args[1].name)

                out_quantizer_name = F"{'.'.join(node.name.split('.'))}_out._input_quantizer"
                out_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(out_quantizer_name, out_quantizer)

                res_quantizer_name = F"{'.'.join(node.name.split('.'))}_res._input_quantizer"
                res_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(res_quantizer_name, res_quantizer)

                fx_utils.add_quantizer(node, model_traced, [0, 1], [out_quantizer_name, res_quantizer_name])

def _ftswin_get_block_name(node):
    while (node.op != 'call_module'):
        node = node.args[0]
    return node.name

class FTSWINQKMatmulTypePatternMatcher(PatternMatcher):
    def __init__(self):
        super(FTSWINQKMatmulTypePatternMatcher, self).__init__()
        self.pattern = FTSWINQKMatmulTypePattern()
        self.pattern_graph = fx_utils.FTSWINLowerQuantOpTracer().trace(self.pattern)
        self.pattern_graph.print_tabular()
        self.pattern_traced = fx.GraphModule(self.pattern, self.pattern_graph)

    def match_and_insert(self, model_traced, quantizer_desc):
        for node in model_traced.graph.nodes:
            if fx_utils.end_node_a_matches_graph_b_types(node, model_traced, self.pattern_graph, self.pattern_traced):
                # Add quantizer to identity branch
                print('node: ', node, node.name, node.args[0].name, node.args[1].name)
                block_name = _ftswin_get_block_name(node)
                block_name_list = block_name.split('_')[:5]
                # print(block_name_list)
                # print('.'.join(node.args[1].name.split('.')))
                block_name = '.'.join(block_name_list)
                # print(block_name)

                query_quantizer_name = F"{block_name}.matmul_q_input_quantizer"
                query_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(query_quantizer_name, query_quantizer)

                key_quantizer_name = F"{block_name}.matmul_k_input_quantizer"
                key_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(key_quantizer_name, key_quantizer)

                fx_utils.add_quantizer(node, model_traced, [0, 1], [query_quantizer_name, key_quantizer_name])

class FTSWINAVMatmulTypePatternMatcher(PatternMatcher):
    def __init__(self):
        super(FTSWINAVMatmulTypePatternMatcher, self).__init__()
        self.pattern = FTSWINAVMatmulTypePattern()
        self.pattern_graph = fx_utils.FTSWINLowerQuantOpTracer().trace(self.pattern)
        self.pattern_graph.print_tabular()
        self.pattern_traced = fx.GraphModule(self.pattern, self.pattern_graph)

    def match_and_insert(self, model_traced, quantizer_desc):
        for node in model_traced.graph.nodes:
            if fx_utils.end_node_a_matches_graph_b_types(node, model_traced, self.pattern_graph, self.pattern_traced):
                # Add quantizer to identity branch
                print('node: ', node, node.name, node.args[0].name, node.args[1].name)
                block_name_list = node.args[0].name.split('_')[:5]
                # print(block_name_list)
                block_name = '.'.join(block_name_list)
                # print(block_name)
                attn_quantizer_name = F"{block_name}.matmul_a_input_quantizer"
                attn_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(attn_quantizer_name, attn_quantizer)

                value_quantizer_name = F"{block_name}.matmul_v_input_quantizer"
                value_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(value_quantizer_name, value_quantizer)

                fx_utils.add_quantizer(node, model_traced, [0, 1], [attn_quantizer_name, value_quantizer_name])

class FTSWINSoftmaxTypePatternMatcher(PatternMatcher):
    def __init__(self):
        super(FTSWINSoftmaxTypePatternMatcher, self).__init__()
        self.pattern = FTSWINSoftmaxTypePattern()
        self.pattern_graph = fx_utils.FTSWINLowerQuantOpTracer().trace(self.pattern)
        self.pattern_graph.print_tabular()
        self.pattern_traced = fx.GraphModule(self.pattern, self.pattern_graph)

    def match_and_insert(self, model_traced, quantizer_desc):
        for node in model_traced.graph.nodes:
            if fx_utils.end_node_a_matches_graph_b_types(node, model_traced, self.pattern_graph, self.pattern_traced):
                # Add quantizer to identity branch
                print('node: ', node, node.name, node.args[0].name)
                block_name_list = node.name.split('_')[:5]
                # print(block_name_list)
                # print('.'.join(node.args[1].name.split('.')))
                block_name = '.'.join(block_name_list)
                # print(block_name)
                softmax_quantizer_name = F"{block_name}.softmax_input_quantizer"
                softmax_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(softmax_quantizer_name, softmax_quantizer)

                fx_utils.add_quantizer(node, model_traced, [0], [softmax_quantizer_name])

class FTSWINResAdd1TypePatternMatcher(PatternMatcher):
    def __init__(self):
        super(FTSWINResAdd1TypePatternMatcher, self).__init__()
        self.pattern = FTSWINResAdd1TypePattern()
        self.pattern_graph = fx_utils.FTSWINLowerQuantOpTracer().trace(self.pattern)
        self.pattern_graph.print_tabular()
        self.pattern_traced = fx.GraphModule(self.pattern, self.pattern_graph)

    def match_and_insert(self, model_traced, quantizer_desc):
        for node in model_traced.graph.nodes:
            if fx_utils.end_node_a_matches_graph_b_types(node, model_traced, self.pattern_graph, self.pattern_traced):
                # Add quantizer to identity branch
                print('node: ', node, node.name, node.args[0].name)
                block_name_list = node.args[1].name.split('_')[:4]
                # print(block_name_list)
                block_name = '.'.join(block_name_list)
                # print(block_name)
                add1_res_quantizer_name = F"{block_name}.add1_residual_input_quantizer"
                add1_res_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(add1_res_quantizer_name, add1_res_quantizer)

                add1_local_quantizer_name = F"{block_name}.add1_local_input_quantizer"
                add1_local_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(add1_local_quantizer_name, add1_local_quantizer)

                fx_utils.add_quantizer(node, model_traced, [0, 1], [add1_res_quantizer_name, add1_local_quantizer_name])

class FTSWINNorm1TypePatternMatcher(PatternMatcher):
    def __init__(self):
        super(FTSWINNorm1TypePatternMatcher, self).__init__()
        self.pattern = FTSWINNorm1TypePattern()
        self.pattern_graph = fx_utils.FTSWINLowerQuantOpTracer().trace(self.pattern)
        self.pattern_graph.print_tabular()
        self.pattern_traced = fx.GraphModule(self.pattern, self.pattern_graph)

    def match_and_insert(self, model_traced, quantizer_desc):
        for node in model_traced.graph.nodes:
            if fx_utils.end_node_a_matches_graph_b_types(node, model_traced, self.pattern_graph, self.pattern_traced):
                # Add quantizer to identity branch
                print('node: ', node, node.name, node.args[0].name)

                # insert quantizer to layernorm
                norm1_quantizer_name = F"{'.'.join(node.args[0].name.split('_'))}.layernorm_input1_quantizer"
                norm1_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(norm1_quantizer_name, norm1_quantizer)
                fx_utils.add_quantizer(node.args[0], model_traced, [0], [norm1_quantizer_name])

class FTSWINResAdd2TypePatternMatcher(PatternMatcher):
    def __init__(self):
        super(FTSWINResAdd2TypePatternMatcher, self).__init__()
        self.pattern = FTSWINResAdd2TypePattern()
        self.pattern_graph = fx_utils.FTSWINLowerQuantOpTracer().trace(self.pattern)
        self.pattern_graph.print_tabular()
        self.pattern_traced = fx.GraphModule(self.pattern, self.pattern_graph)

    def match_and_insert(self, model_traced, quantizer_desc):
        for node in model_traced.graph.nodes:
            if fx_utils.end_node_a_matches_graph_b_types(node, model_traced, self.pattern_graph, self.pattern_traced):
                # Add quantizer to identity branch
                print('node: ', node, node.name, node.args[0].name, node.args[1].name)
                block_name_list = node.args[1].name.split('_')[:4]
                # print(block_name_list)
                block_name = '.'.join(block_name_list)
                # print(block_name)
                add2_res_quantizer_name = F"{block_name}.add2_residual_input_quantizer"
                add2_res_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(add2_res_quantizer_name, add2_res_quantizer)

                add2_local_quantizer_name = F"{block_name}.add2_local_input_quantizer"
                add2_local_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(add2_local_quantizer_name, add2_local_quantizer)

                # insert quantizer to add
                fx_utils.add_quantizer(node, model_traced, [0, 1], [add2_res_quantizer_name, add2_local_quantizer_name])


class FTSWINNorm2TypePatternMatcher(PatternMatcher):
    def __init__(self):
        super(FTSWINNorm2TypePatternMatcher, self).__init__()
        self.pattern = FTSWINNorm2TypePattern()
        self.pattern_graph = fx_utils.FTSWINLowerQuantOpTracer().trace(self.pattern)
        self.pattern_graph.print_tabular()
        self.pattern_traced = fx.GraphModule(self.pattern, self.pattern_graph)

    def match_and_insert(self, model_traced, quantizer_desc):
        for node in model_traced.graph.nodes:
            if fx_utils.end_node_a_matches_graph_b_types(node, model_traced, self.pattern_graph, self.pattern_traced):
                # Add quantizer to identity branch
                print('node: ', node, node.name, node.args[0].name)

                # insert quantizer to layernorm
                norm2_quantizer_name = F"{'.'.join(node.args[0].name.split('_'))}.layernorm_input2_quantizer"
                norm2_quantizer = FX_TENSOR_QUANT_MAP[quantizer_desc.quantizer_type](quantizer_desc)
                model_traced.add_submodule(norm2_quantizer_name, norm2_quantizer)
                fx_utils.add_quantizer(node.args[0], model_traced, [0], [norm2_quantizer_name])

class InstPatternMatcher(object):
    def __init__(self):
        self.pattern_matchers = list()

    def get_pattern_matchers(self):
        return self.pattern_matchers

class CNNPatternMatcher(InstPatternMatcher):
    def __init__(self):
        super().__init__()
        self.pattern_matchers.append(ConvBnResReluTypePatternMatcher())
        self.pattern_matchers.append(SEReLUTypePatternMatcher())
        self.pattern_matchers.append(SESiLUTypePatternMatcher())
        self.pattern_matchers.append(DropActDropPathAddTypePatternMatcher())
        self.pattern_matchers.append(MeanTypePatternMatcher())
        self.pattern_matchers.append(SEAvgPoolTypePatternMatcher())
        self.pattern_matchers.append(HardSigmoidTypePatternMatcher())

class BERTPatternMatcher(InstPatternMatcher):
    def __init__(self):
        super().__init__()
        self.pattern_matchers.append(BERTQueryKeyTypePatternMatcher())
        self.pattern_matchers.append(BERTAttnOutTypePatternMatcher())
        self.pattern_matchers.append(BERTResAddTypePatternMatcher())

class FTSWINPatternMatcher(InstPatternMatcher):
    def __init__(self):
        super().__init__()
        self.pattern_matchers.append(FTSWINQKMatmulTypePatternMatcher())
        self.pattern_matchers.append(FTSWINAVMatmulTypePatternMatcher())
        self.pattern_matchers.append(FTSWINSoftmaxTypePatternMatcher())
        self.pattern_matchers.append(FTSWINResAdd1TypePatternMatcher())
        self.pattern_matchers.append(FTSWINNorm1TypePatternMatcher())
        self.pattern_matchers.append(FTSWINResAdd2TypePatternMatcher())
        self.pattern_matchers.append(FTSWINNorm2TypePatternMatcher())


class PatternMatcherFactory(object):
    @classmethod
    def get_pattern_matcher(cls, type_str):
        return eval("{}PatternMatcher".format(type_str))()


