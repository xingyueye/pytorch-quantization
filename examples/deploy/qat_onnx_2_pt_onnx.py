import os.path

import onnx
import numpy as np
import struct
import sys
from onnx import helper

if __name__ == '__main__':

    onnx_file = sys.argv[1]

    # model = onnx.load("../../../YOLOv6_origin/cwd_lsq_full_quant_10e_5e-4wd_0.001lr_420_partial_62_bs1_non_graph_opt.onnx")
    # model = onnx.load("cwd_part_quant_search_60e_5e-4wd_0.001lr_423_partial_56_bs1.onnx")
    # model = onnx.load("cwd_lsq_qdq_full_quant_10e_5e-4wd_0.001lr_417_last_qdq_bs1.onnx")
    model = onnx.load(onnx_file)
    graph = model.graph
    nodes = graph.node

    # count = 0
    # for idx, node in enumerate(nodes):
    #     # print(node)
    #     # print("node type = {}, node name={}".format(node.op_type, node.name))
    #     if node.op_type == 'QuantizeLinear':
    #         count += 1
    # print(count)
    #
    # onnx.save(model, "yolov6_pruned.onnx")

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
            print(scale_name)
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
            for i, node in enumerate(graph.node):
                if node.output[0] == scale_name:
                    if len(node.attribute[0].t.dims) == 0:
                        # print(node.attribute[0].t.raw_data)
                        print(np.frombuffer(node.attribute[0].t.raw_data, dtype=np.float32))
                        val = np.frombuffer(node.attribute[0].t.raw_data, dtype=np.float32)[0]
                        activation_map[in_name] = struct.pack('>f', val).hex()
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
           print(scale_name)
           out_name = node.output[0]
           in_rename_map[out_name] = in_name
           graph.node.remove(nodes[node_id])
           # for i, node in enumerate(graph.node):
           #     if node.output[0] == scale_name:
           #         if len(node.attribute[0].t.dims) == 0:
           #             # print(node.attribute[0].t.raw_data)
           #             print(np.frombuffer(node.attribute[0].t.raw_data, dtype=np.float32))
           #             val = np.frombuffer(node.attribute[0].t.raw_data, dtype=np.float32)[0]
           #             activation_map[out_name] = struct.pack('>f', val).hex()

           # for i, node in enumerate(graph.node):
           #     if node.output[0] == scale_name:
           #         graph.node.remove(nodes[i])
           # for i, node in enumerate(graph.node):
           #     if node.output[0] == zero_name:
           #         graph.node.remove(nodes[i])
           scale_node_list.append(scale_name)
           zero_node_list.append(zero_name)

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

    onnx_name = os.path.basename(onnx_file)
    onnx_dir = os.path.dirname(onnx_file)
    onnx_new_name = onnx_name.replace('.onnx', '_remove_qdq.onnx')
    onnx.save(model, os.path.join(onnx_dir, onnx_new_name))
    cache_name = onnx_new_name.replace('.onnx', '_calibration.cache')
    with open(os.path.join(onnx_dir, cache_name), 'w') as file_cache:
        file_cache.write('TRT-8XXX-EntropyCalibration2\n')
        for k, v in activation_map.items():
            file_cache.write("{}: {}\n".format(k, v))