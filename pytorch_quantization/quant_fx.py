import torch.nn as nn
import torch.fx as fx

from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.graph import fx_utils
from pytorch_quantization.graph.fx_matcher import get_internal_pattern_matcher


def insert_qdq_nodes(model, calib_method, num_bits=8):
    if calib_method == 'max':
        method = 'max'
    else:
        method = 'histogram'
    model.eval()
    # We use LowerQuantOpTracer
    model_traced = fx.GraphModule(model, fx_utils.LowerQuantOpTracer().trace(model))
    # Create QuantizerDescriptor
    quantizer_desc = QuantDescriptor(num_bits=num_bits, calib_method=method)
    # Pattern match
    pattern_matchers = get_internal_pattern_matcher()
    for pattern_matcher in pattern_matchers:
        pattern_matcher.match_and_insert(model_traced, quantizer_desc)
    # Recompile
    model_traced.recompile()
    model_traced.graph.lint()
    model_traced.graph.print_tabular()

    return model_traced