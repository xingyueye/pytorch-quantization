import torch.fx as fx

from mtpq.graph.fx_utils import LowerTracerFactory
from mtpq.graph.fx_matcher import PatternMatcherFactory


# def insert_qdq_nodes(model, calib_method, num_bits=8):
#     if calib_method == 'max':
#         method = 'max'
#     else:
#         method = 'histogram'
#     model.eval()
#     # We use LowerQuantOpTracer
#     model_traced = fx.GraphModule(model, fx_utils.LowerQuantOpTracer().trace(model))
#     # Create QuantizerDescriptor
#     quantizer_desc = QuantDescriptor(num_bits=num_bits, calib_method=method)
#     # Pattern match
#     pattern_matchers = get_internal_pattern_matcher()
#     for pattern_matcher in pattern_matchers:
#         pattern_matcher.match_and_insert(model_traced, quantizer_desc)
#     # Recompile
#     model_traced.recompile()
#     model_traced.graph.lint()
#     model_traced.graph.print_tabular()
#
#     return model_traced

def insert_qdq_nodes_via_subgraph_match(model, quantizer_desc, type_str='CNN', do_trace=True):
    if do_trace:
        model.eval()
        # We use LowerQuantOpTracer
        lower_tracer = LowerTracerFactory.get_lower_tracer(type_str)
        model_traced = fx.GraphModule(model, lower_tracer.trace(model))
    else:
        model_traced = model
    # Pattern match
    inst_pattern_matcher = PatternMatcherFactory.get_pattern_matcher(type_str)
    for pattern_matcher in inst_pattern_matcher.get_pattern_matchers():
        pattern_matcher.match_and_insert(model_traced, quantizer_desc)
    # Recompile
    model_traced.recompile()
    model_traced.graph.lint()
    model_traced.graph.print_tabular()

    return model_traced