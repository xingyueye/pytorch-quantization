"""
Basic FX functions, some from pytorch source code
"""
from typing import Any, Callable, Optional

from torch import fx
from torch import nn
from pytorch_quantization import nn as quant_nn

def getattr_from_fqn(gm: fx.GraphModule, fqn: str) -> Any:
    """
    Given a gm and a fqn such as "foo.bar.baz", returns gm.foo.bar.baz.
    """
    fqn_parts = fqn.split(".")
    cur_val = gm
    for part in fqn_parts:
        cur_val = getattr(cur_val, part)
    return cur_val


def get_target_type(node: fx.Node, gm: fx.GraphModule) -> Optional[Callable]:
    """
    Given a node, returns the target type (i.e. torch.add, nn.Conv2d, etc).
    """
    if node.op == 'call_module':
        return type(getattr_from_fqn(gm, node.target))
    if node.op == 'call_function':
        return node.target
    if node.op == 'call_method':
        return node.target
    return None

class NodeIterator:
    """
    Iterates from the ouput node backwards. Raises
    StopIteration after running out of nodes.
    """
    def __init__(self, gm: fx.GraphModule, node: fx.Node):
        self.gm = gm
        self.seen_nodes = set()
        self.stack = [node]

    def __iter__(self):
        return self

    def __next__(self):
        while len(self.stack) > 0:
            cur_node = self.stack.pop()
            if cur_node in self.seen_nodes:
                continue
            self.seen_nodes.add(cur_node)
            for arg in cur_node.args:
                if isinstance(arg, fx.Node):
                    self.stack.append(arg)
            if cur_node.op not in ('call_module', 'call_method', 'call_function', 'placeholder'):
                continue
            return cur_node
        raise StopIteration

    def skip_next_node(self):
        cur_node = self.stack.pop()
        self.seen_nodes.add(cur_node)

def end_node_a_matches_graph_b_types(
    end_node_a: fx.Node,
    gm_a: fx.GraphModule,
    graph_b: fx.Graph,
    gm_b: fx.GraphModule,
) -> bool:
    """
    Iterates backwards from end_node_a until we either match every node
    in graph_b or find a mismatch.  Returns True if every node
    was matched.  Matching is done on node target type only.
    """

    cur_node_a = end_node_a
    # assume sample_graph has a single output
    # TODO(future): handle multiple outputs if needed
    output_nodes_b = [n for n in graph_b.nodes if n.op == 'output']
    assert len(output_nodes_b) == 1
    cur_node_b = output_nodes_b[0]

    iter_a, iter_b = \
        NodeIterator(gm_a, cur_node_a), NodeIterator(gm_b, cur_node_b)

    while True:
        cur_node_a, cur_node_b = None, None
        try:
            cur_node_b = next(iter_b)
        except StopIteration:
            return True  # FIXME: should this return True or None? if it reaches the last node of b and hasn't returned false
        # if we see an input from graph B, we stop graph A's
        # iterator from following the corresponding path
        if cur_node_b is not None and cur_node_b.op == 'placeholder':
            iter_a.skip_next_node()
            continue
        try:
            cur_node_a = next(iter_a)
        except StopIteration:
            pass

        if cur_node_a is None and cur_node_b is None:
            break
        elif cur_node_a is None or cur_node_b is None:
            return False
        else:
            # potential match
            type_a = get_target_type(cur_node_a, gm_a)
            type_b = get_target_type(cur_node_b, gm_b)
            if type_a != type_b:
                return False

    return True


def add_quantizer(node, graph_module, quant_arg_ids, quantizer_names):
    """Add quantizers to input(s) of a node

    Args:
        node (fx.Node):
        graph_module (fx.GraphModule): Traced model to add quantizer node into
        quant_arg_ids (list of ints): Argument ids to add quantizer node.
        quantizer_names (list of str): names of quantizer node to be added. Length must match arg_ids
    """
    new_args = []
    for arg_id, arg in enumerate(node.args):
        if arg_id in quant_arg_ids:
            with graph_module.graph.inserting_before(node):
                quant_node = graph_module.graph.call_module(
                    quantizer_names[quant_arg_ids.index(arg_id)], args=(arg,), kwargs={})
            new_args.append(quant_node)
        else:
            new_args.append(arg)

    node.args = tuple(new_args)


class LowerConvLinearTracer(fx.Tracer):
    """Tracer lowering Conv2d and Linear module

    By default, nn.Module is a leaf. When we need to trace down to nn.functional, must set is_leaf_module to False
    """
    def is_leaf_module(self, m : nn.Module, qualname : str):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            return False
        return super().is_leaf_module(m, qualname)


class LowerQuantOpTracer(fx.Tracer):
    """Tracer lowering Conv2d and Linear module

    By default, nn.Module is a leaf. When we need to trace down to nn.functional, must set is_leaf_module to False
    """
    def is_leaf_module(self, m : nn.Module, qualname : str):
        if isinstance(m, (quant_nn.QuantConv1d, quant_nn.QuantConv2d, quant_nn.QuantConv3d, quant_nn.QuantLinear)):
            return True
        if isinstance(m, (quant_nn.QuantConvTranspose1d, quant_nn.QuantConvTranspose2d, quant_nn.QuantConvTranspose3d)):
            return True
        if isinstance(m, (quant_nn.QuantMaxPool1d, quant_nn.QuantMaxPool2d, quant_nn.QuantMaxPool3d)):
            return True
        if isinstance(m, (quant_nn.QuantAvgPool1d, quant_nn.QuantAvgPool2d, quant_nn.QuantAvgPool3d)):
            return True
        if isinstance(m, (quant_nn.QuantAdaptiveAvgPool1d, quant_nn.QuantAdaptiveAvgPool2d, quant_nn.QuantAdaptiveAvgPool3d)):
            return True
        return super().is_leaf_module(m, qualname)


class LowerTracer(fx.Tracer):
    """Tracer lowering Conv2d and Linear module

    By default, nn.Module is a leaf. When we need to trace down to nn.functional, must set is_leaf_module to False
    """
    def is_leaf_module(self, m : nn.Module, qualname : str):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            return False
        if isinstance(m, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
            return False
        if isinstance(m, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)):
            return False
        if isinstance(m, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
            return False
        return super().is_leaf_module(m, qualname)