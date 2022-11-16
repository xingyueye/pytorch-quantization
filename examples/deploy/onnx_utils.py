import os.path

import onnx
import numpy as np
import struct
import sys
import copy


def onnx_remove_qdqnode(onnx_model):
    onnx_replica = copy.deepcopy(onnx_model)
    graph = onnx_replica.graph
    nodes = graph.node

    # demo for remove node with first input and output
    in_rename_map = {}
    scale_node_list = []
    zero_node_list = []
    activation_map = {}
    for node_id, node in enumerate(graph.node):
        if node.op_type == "QuantizeLinear":
            # node input
            in_name = node.input[0]
            scale_name = node.input[1]
            zero_name = node.input[2]
            # print(scale_name)
            # node output
            out_name = node.output[0]
            # record input, remove one node, set node's input to its next
            in_rename_map[out_name] = in_name
            scale_node_list.append(scale_name)
            zero_node_list.append(zero_name)
            # for i, node in enumerate(graph.node):
            #     if node.output[0] == scale_name:
            #         if len(node.attribute[0].t.dims) > 0:
            #             print(node.attribute[0].t.dims)
            #         graph.node.remove(nodes[i])
            # for i, node in enumerate(graph.node):
            #    if node.output[0] == zero_name:
            #        graph.node.remove(nodes[i])
            # record scale of activation
            for i, node in enumerate(graph.node):
                if node.output[0] == scale_name:
                    if len(node.attribute[0].t.dims) == 0:
                        # print(node.attribute[0].t.raw_data)
                        # print(np.frombuffer(node.attribute[0].t.raw_data, dtype=np.float32))
                        val = np.frombuffer(node.attribute[0].t.raw_data, dtype=np.float32)[0]
                        activation_map[in_name] = struct.pack('>f', val).hex()
            # remove QuantizeLinear node
            graph.node.remove(nodes[node_id])


    # relink
    for node_id, node in enumerate(graph.node):
       for in_id, in_name in enumerate(node.input):
           if in_name in in_rename_map.keys():
               # set node input == removed node's input
               node.input[in_id] = in_rename_map[in_name]

    in_rename_map = {}
    # activation_map = {}
    for node_id, node in enumerate(graph.node):
       if node.op_type == "DequantizeLinear":
           in_name = node.input[0]
           scale_name = node.input[1]
           zero_name = node.input[2]
           # print(scale_name)
           out_name = node.output[0]
           in_rename_map[out_name] = in_name
           graph.node.remove(nodes[node_id])
           scale_node_list.append(scale_name)
           zero_node_list.append(zero_name)

    # relink
    for node_id, node in enumerate(graph.node):
       for in_id, in_name in enumerate(node.input):
           if in_name in in_rename_map.keys():
               node.input[in_id] = in_rename_map[in_name]

    nodes = graph.node
    for node_name in (scale_node_list + zero_node_list):
        for node_id, node in enumerate(graph.node):
            if node.name == node_name:
                # print("node input={}".format(node.input))
                # for node_input in node.input:
                #     print(node_input)
                #     graph.node.remove(node_input)
                graph.node.remove(nodes[node_id])

    for node_name in (scale_node_list + zero_node_list):
        for node_id, node in enumerate(graph.node):
            if node.output[0] == node_name:
                # print("node input={}".format(node.input))
                # for node_input in node.input:
                #     print(node_input)
                #     graph.node.remove(node_input)
                graph.node.remove(nodes[node_id])

    return onnx_replica, activation_map

def save_calib_cache_file(cache_file, activation_map, headline='TRT-8XXX-EntropyCalibration2\n'):
    with open(os.path.join(cache_file), 'w') as cfile:
        cfile.write(headline)
        for k, v in activation_map.items():
            cfile.write("{}: {}\n".format(k, v))

if __name__ == '__main__':

    onnx_file = sys.argv[1]
    model = onnx.load(onnx_file)
    model_wo_qdq, activation_map = onnx_remove_qdqnode(model)

    onnx_name, onnx_dir = os.path.basename(onnx_file), os.path.dirname(onnx_file)
    onnx_new_name = onnx_name.replace('.onnx', '_remove_qdq.onnx')
    onnx.save(model, os.path.join(onnx_dir, onnx_new_name))
    cache_name = onnx_new_name.replace('.onnx', '_calibration.cache')
    save_calib_cache_file(cache_name, activation_map)
