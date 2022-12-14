from torch import nn
from torch import fx

from pytorch_quantization.nn import TensorQuantizer
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.graph import fx_utils
from pytorch_quantization.graph.fx_pattern import ConvBnResReluTypePattern, SESiLUTypePattern

_MODULES_QUANT_INPUT_AND_WEIGHTS = [
    nn.functional.conv2d,
    nn.functional.linear,
]

_MODULES_QUANT_INPUT_AND_WEIGHTS_TRANS = [
    nn.functional.conv_transpose2d,
]

_MODULES_QUANT_ONLY_INPUT = [
    # "maxpool",
    # "avgpool",
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


def insert_qdq_nodes(model, calib_method, num_bits=8):
    """Dynamic module insert with torch.fx.

    Main steps:
    1. Dynamically insert quantizers to the modules according to different types.
       Two types:
       1) For modules with weights, we need to insert both weight quantizer and input quantizer.
       torch.nn.Conv1d
       torch.nn.Conv2d
       torch.nn.Conv3d
       torch.nn.ConvTranspose1d
       torch.nn.ConvTranspose2d
       torch.nn.ConvTranspose3d
       torch.nn.Linear

       2) For modules without weights, we only need to insert the input quantizer.
       torch.nn.AvgPool1d
       torch.nn.AvgPool2d
       torch.nn.AvgPool3d
       torch.nn.AdaptiveAvgPool1d
       torch.nn.AdaptiveAvgPool2d
       torch.nn.AdaptiveAvgPool3d
       torch.nn.MaxPool1d
       torch.nn.MaxPool2d
       torch.nn.MaxPool3d

    2. Using pattern match to search the graph with specific pattern, and insert extra quantizer to improve throughput.
       Such as:
       1) For resnet50, QuantizeConvWithResidualAdd;
       2) For yolov5,
       3) For CenterNet,
    """
    if calib_method == 'max':
        method = 'max'
    else:
        method = 'histogram'
    model.eval()
    lower_conv_linear = True
    if lower_conv_linear:
        model_traced = fx.GraphModule(model, fx_utils.LowerTracer().trace(model))
    else:
        model_traced = fx.symbolic_trace(model)

    conv_bn_res_pattern = fx.symbolic_trace(ConvBnResReluTypePattern(lower_conv_linear))
    se_silu_pattern = fx.symbolic_trace(SESiLUTypePattern(lower_conv_linear))

    for node in model_traced.graph.nodes:
        if node.target in _MODULES_QUANT_INPUT_AND_WEIGHTS:
            # Add quantizer to both input activation and weight

            # Because we lowered Conv and Linear module, we don't know the name of them.
            # The only place has that information is name of weight, args[1]
            layer_name = ".".join(node.args[1].name.split("_")[:-1])
            input_quantizer_name = F"{layer_name}.input_quantizer"
            weight_quantizer_name = F"{layer_name}.weight_quantizer"

            # Must add quantizer module first before creating call_module node
            model_traced.add_submodule(input_quantizer_name, TensorQuantizer(QuantDescriptor(num_bits=num_bits,
                                                                                             calib_method=method)))
            model_traced.add_submodule(weight_quantizer_name, TensorQuantizer(QuantDescriptor(num_bits=num_bits,
                                                                                              calib_method='max', axis=0)))

            fx_utils.add_quantizer(node, model_traced, (0, 1), (input_quantizer_name, weight_quantizer_name))
        elif node.target in _MODULES_QUANT_INPUT_AND_WEIGHTS_TRANS:
            # Add quantizer to both input activation and weight

            # Because we lowered Conv and Linear module, we don't know the name of them.
            # The only place has that information is name of weight, args[1]
            layer_name = ".".join(node.args[1].name.split("_")[:-1])
            input_quantizer_name = F"{layer_name}.input_quantizer"
            weight_quantizer_name = F"{layer_name}.weight_quantizer"

            # Must add quantizer module first before creating call_module node
            model_traced.add_submodule(input_quantizer_name, TensorQuantizer(QuantDescriptor(num_bits=num_bits,
                                                                                             calib_method=method)))
            model_traced.add_submodule(weight_quantizer_name, TensorQuantizer(QuantDescriptor(num_bits=num_bits,
                                                                                              calib_method='max', axis=1)))

            fx_utils.add_quantizer(node, model_traced, (0, 1), (input_quantizer_name, weight_quantizer_name))
        elif node.target in _MODULES_QUANT_ONLY_INPUT:
            # Add quantizer to global pooling
            avgpool_quantizer_name = F"{node.target}.input_quantizer"
            model_traced.add_submodule(avgpool_quantizer_name, TensorQuantizer(QuantDescriptor(num_bits=num_bits,
                                                                                               calib_method=method)))

            fx_utils.add_quantizer(node, model_traced, (0,), (avgpool_quantizer_name,))

        elif fx_utils.end_node_a_matches_graph_b_types(node, model_traced, conv_bn_res_pattern.graph, conv_bn_res_pattern):
            # Add quantizer to one edge of the residual add
            print('node: ', node, node.args[0].name)

            res_add_quantizer_name = F"{'.'.join(node.args[0].name.split('.'))}.input_quantizer"
            model_traced.add_submodule(res_add_quantizer_name, TensorQuantizer(QuantDescriptor(num_bits=num_bits,
                                                                                               calib_method=method)))

            # The matched end node is ReLU, whose args[0] is the add node we want to add quantizer to
            fx_utils.add_quantizer(node.args[0], model_traced, (1,), (res_add_quantizer_name,))
        elif fx_utils.end_node_a_matches_graph_b_types(node, model_traced, se_silu_pattern.graph, se_silu_pattern):
            # Add quantizer to one edge of the residual mul
            print('node: ', node, node.name)

            res_mul_quantizer_name = F"{'.'.join(node.name.split('.'))}.input_quantizer"
            model_traced.add_submodule(res_mul_quantizer_name, TensorQuantizer(QuantDescriptor(num_bits=num_bits,
                                                                                               calib_method=method)))

            # The matched end node is Mul, whose args[0] we want to add quantizer to
            fx_utils.add_quantizer(node, model_traced, (0,), (res_mul_quantizer_name,))

    model_traced.recompile()
    model_traced.graph.lint()
    model_traced.graph.print_tabular()

    return model_traced