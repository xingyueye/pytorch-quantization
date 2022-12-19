from torch import fx

from pytorch_quantization.graph import fx_utils
from pytorch_quantization.graph.fx_matcher import get_internal_pattern_matcher, get_internal_op_matcher

def insert_qdq_nodes(model, calib_method, num_bits=8):
    if calib_method == 'max':
        method = 'max'
    else:
        method = 'histogram'
    model.eval()
    # We use LowerTracer
    model_traced = fx.GraphModule(model, fx_utils.LowerTracer().trace(model))
    # pattern match
    pattern_matchers = get_internal_pattern_matcher()
    for pattern_matcher in pattern_matchers:
        pattern_matcher.match_and_insert(model_traced, num_bits=num_bits, method=method)

    # op match
    op_matchers = get_internal_op_matcher()
    for op_matcher in op_matchers:
        op_matcher.match_and_insert(model_traced, num_bits=num_bits, method=method)

    model_traced.recompile()
    model_traced.graph.lint()
    model_traced.graph.print_tabular()

    return model_traced