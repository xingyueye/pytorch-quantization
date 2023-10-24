import os.path

import onnx
import numpy as np
import struct
import sys
import copy
import json

def search_node_by_output_id(nodes, output_id: str):
    prev_node = None
    for node_id, node in enumerate(nodes):
        if output_id in node.output:
            prev_node = node
            break
    return prev_node

def get_prev_node(nodes, node):
    node_input_list = node.input
    prev_node_list = []
    for node_id, node in enumerate(nodes):
        for node_output in node.output:
            if node_output in node_input_list:
                prev_node_list.append(node)
    return prev_node_list

def get_next_node(nodes, node):
    node_output_list = node.output
    next_node_list = []
    for node_id, node in enumerate(nodes):
        for node_input in node.input:
            if node_input in node_output_list:
                next_node_list.append(node)
    return next_node_list

def get_conv_qdq_node(nodes, conv_node):
    # get conv input
    conv_input_id = conv_node.input[0]
    # print(conv_input_id)
    dequant_node = None
    quant_node = None
    # get dequant node by conv input
    for node_id, node in enumerate(nodes):
        if node.op_type == "DequantizeLinear" and conv_input_id in node.output:
            dequant_node = node
            break
    # get quant node by dequant input
    if dequant_node is not None:
        dequant_input_id = dequant_node.input[0]
        # print(dequant_input_id)
        for node_id, node in enumerate(nodes):
            if node.op_type == "QuantizeLinear" and dequant_input_id in node.output:
                quant_node = node
                break
    # print(dequant_node)
    # print(quant_node)
    return dequant_node, quant_node

def onnx_conv_horizon_fuse(onnx_model):
    onnx_replica = copy.deepcopy(onnx_model)
    graph = onnx_replica.graph
    nodes = graph.node
    # find qualified add op
    pattern = []
    for node_id, node in enumerate(graph.node):
        if node.op_type == "Add":
            avail_count = 0
            for input_id in node.input:
                prev_node = search_node_by_output_id(graph.node, input_id)
                # prev node must be BatchNorm or Conv
                if prev_node is not None:
                    if prev_node.op_type in ['BatchNormalization', 'Conv'] and \
                            len(prev_node.output) == 1:
                        avail_count += 1
            if avail_count == 2:
                pattern.append(node)
    # print(pattern)

    # process each add
    for add_node in pattern:
        prev_add_node_list = get_prev_node(nodes, add_node)
        # collect conv node
        conv_node_list = []
        for node in prev_add_node_list:
            if node.op_type == "BatchNormalization":
                prev_node_list = get_prev_node(nodes, node)
                assert len(prev_node_list) == 1 and prev_node_list[0].op_type == "Conv", \
                    "Conv horizon fusion pattern not match"
                conv_node_list.append(prev_node_list[0])
            else:
                conv_node_list.append(node)

        # print(conv_node_list)
        # collect qdq node
        qdq_node_list = []
        for node in conv_node_list:
            dequant_node, quant_node = get_conv_qdq_node(nodes, node)
            assert dequant_node is not None and quant_node is not None, "Conv horizon fusion pattern not match"
            qdq_node_list.extend((dequant_node, quant_node))

        # find scale node
        scale_node_list = []
        for qdq_node in qdq_node_list:
            scale_iput_id = qdq_node.input[1]
            for node in nodes:
                if scale_iput_id in node.output:
                    scale_node_list.append(node)
        # print(scale_node_list)
        # get max scale
        max = 0
        for scale_node in scale_node_list:
            val = np.frombuffer(scale_node.attribute[0].t.raw_data, dtype=np.float32)[0]
            print(val)
            if max < val:
                max = val
        # rewrite max scale
        for scale_node in scale_node_list:
            scale_node.attribute[0].t.raw_data = bytes(struct.pack("f", max))

        # check
        for scale_node in scale_node_list:
            val = np.frombuffer(scale_node.attribute[0].t.raw_data, dtype=np.float32)[0]
            print(val)

    return onnx_replica

def onnx_add_insert_qdqnode(onnx_model):
    onnx_replica = copy.deepcopy(onnx_model)
    graph = onnx_replica.graph
    nodes = graph.node
    # find qualified add op
    patterns = []
    for node_id, node in enumerate(graph.node):
        if node.op_type == "Add":
            same_input_node_list = []
            same_input = None
            for add_input in node.input:
                for other_id, other_node in enumerate(nodes):
                    if other_id != node_id:
                        for other_input in other_node.input:
                            if other_input == add_input:
                                same_input_node_list.append(other_node)
                                same_input = other_input
                                break
            # Find previous node of Add, which has two output, one is QuantizeLinear, other is Add
            if len(same_input_node_list) == 1 and same_input_node_list[0].op_type == 'QuantizeLinear':
                prev_add_node = search_node_by_output_id(nodes, same_input)
                dequant_node = get_next_node(nodes, same_input_node_list[0])[0]
                patterns.append((node, prev_add_node, same_input_node_list[0], dequant_node, same_input))
    print(patterns)
    for pattern in patterns:
        add_node, prev_add_node, quant_node, dequant_node, same_input = pattern
        dq_x, dq_s, dq_z = dequant_node.input
        new_quant_node = onnx.helper.make_node('QuantizeLinear',
                                                inputs=quant_node.input,
                                                outputs=[prev_add_node.name + "_Dequant"],
                                                name=prev_add_node.name + "_QuantizeLinear")
        new_dequant_node = onnx.helper.make_node('DequantizeLinear',
                                                inputs=[prev_add_node.name + "_Dequant", dq_s, dq_z],
                                                outputs=[prev_add_node.name + "_Add"],
                                                name=prev_add_node.name + "_DequantizeLinear")

        add_node.input.remove(same_input)
        add_node.input.append(prev_add_node.name + "_Add")
        for node_id, node in enumerate(graph.node):
            if node.name == prev_add_node.name:
                graph.node.insert(node_id + 1, new_quant_node)
                graph.node.insert(node_id + 2, new_dequant_node)

    return onnx_replica

        # new_dequant_node = onnx.helper.make_node('DequantizeLinear',
        #                                         inputs=quant_node.input,
        #                                         outputs=prev_add_node.output,
        #                                         name=prev_add_node.name + "_DequantizeLinear")


def onnx_remove_qdqnode_ada(onnx_model,unsigned_flag):
    onnx_replica = copy.deepcopy(onnx_model)
    graph = onnx_replica.graph
    nodes = graph.node

    # demo for remove node with first input and output
    in_rename_map = {}
    scale_node_list = []
    zero_node_list = []
    quant_node_list = []
    Q_dict = {}
    DQ_dict = {}
    
    activation_map = {}
    weight_map = {}
    for node_id, node in enumerate(graph.node):
        if node.op_type == "QuantizeLinear":
            out_name = node.output[0]
            act_node = node.input[0]
            act_scale = node.input[1]
            act_zp = node.input[2]
            Q_dict[out_name] = [act_node, act_scale, act_zp]
            activation_map[act_node] = [0.,0]
            for i in range(node_id):
                if nodes[i].output[0] == act_scale:
                    val = np.frombuffer(nodes[i].attribute[0].t.raw_data, dtype=np.float32)[0]
                    if act_node in activation_map.keys():
                        old_val = activation_map[act_node][0]
                        if val > old_val:
                            activation_map[act_node][0] = val
                    else: 
                        activation_map[act_node][0] = val
                elif nodes[i].output[0] == act_zp:
                    if unsigned_flag:
                        val = np.frombuffer(nodes[i].attribute[0].t.raw_data, dtype=np.uint8)[0]
                    else:
                        val = np.frombuffer(nodes[i].attribute[0].t.raw_data, dtype=np.int8)[0]
                    activation_map[act_node][1] = val 
            # record input, remove one node, set node's input to its next
            in_rename_map[out_name] = act_node
            scale_node_list.append(act_scale)
            zero_node_list.append(act_zp)
            quant_node_list.append(node)
        elif node.op_type == "DequantizeLinear":
            in_name = node.input[0]
            out_name = node.output[0]
            DQ_dict[out_name] = in_name


    # Remove QuantizeLinear node
    for quant_node in quant_node_list:
        graph.node.remove(quant_node)


    # relink
    for node_id, node in enumerate(graph.node):
       for in_id, in_name in enumerate(node.input):
           if in_name in in_rename_map.keys():
               # set node input == removed node's input
               node.input[in_id] = in_rename_map[in_name]

    in_rename_map = {}
    # activation_map = {}
    dequant_node_list = []
    for node_id, node in enumerate(graph.node):
       if node.op_type == "DequantizeLinear":
           in_name = node.input[0]
           scale_name = node.input[1]
           zero_name = node.input[2]
           # print(scale_name)
           out_name = node.output[0]
           in_rename_map[out_name] = in_name
           # remove later
           dequant_node_list.append(node)
           scale_node_list.append(scale_name)
           zero_node_list.append(zero_name)

    # Remove DequantizeLinear node
    for dequant_node in dequant_node_list:
        graph.node.remove(dequant_node)

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

    return onnx_replica, activation_map, weight_map

def onnx_remove_qdqnode(onnx_model,unsigned_flag):
    onnx_replica = copy.deepcopy(onnx_model)
    graph = onnx_replica.graph
    nodes = graph.node

    # demo for remove node with first input and output
    in_rename_map = {}
    scale_node_list = []
    zero_node_list = []
    quant_node_list = []
    Q_dict = {}
    DQ_dict = {}
    
    activation_map = {}
    weight_map = {}
    for node_id, node in enumerate(graph.node):
        if node.op_type == "QuantizeLinear":
            out_name = node.output[0]
            act_node = node.input[0]
            act_scale = node.input[1]
            act_zp = node.input[2]
            Q_dict[out_name] = [act_node, act_scale, act_zp]
            activation_map[act_node] = [0.,0]
            for i in range(node_id):
                if nodes[i].output[0] == act_scale:
                    val = np.frombuffer(nodes[i].attribute[0].t.raw_data, dtype=np.float32)[0]
                    if act_node in activation_map.keys():
                        old_val = activation_map[act_node][0]
                        if val > old_val:
                            activation_map[act_node][0] = val
                    else: 
                        activation_map[act_node][0] = val
                elif nodes[i].output[0] == act_zp:
                    if unsigned_flag:
                        val = np.frombuffer(nodes[i].attribute[0].t.raw_data, dtype=np.uint8)[0]
                    else:
                        val = np.frombuffer(nodes[i].attribute[0].t.raw_data, dtype=np.int8)[0]
                    activation_map[act_node][1] = val 
            # record input, remove one node, set node's input to its next
            in_rename_map[out_name] = act_node
            scale_node_list.append(act_scale)
            zero_node_list.append(act_zp)
            quant_node_list.append(node)
        elif node.op_type == "DequantizeLinear":
            in_name = node.input[0]
            out_name = node.output[0]
            DQ_dict[out_name] = in_name
        elif node.op_type == "Conv" or node.op_type == "Gemm":
            in_name = node.input[0]
            weight_name = node.input[1]
            out_name = node.output[0]
            
            weight_node = Q_dict[DQ_dict[weight_name]][0]
            weight_scale = Q_dict[DQ_dict[weight_name]][1]
            weight_zp = Q_dict[DQ_dict[weight_name]][2]
            # delete the weight and bias in activation map
            del activation_map[weight_node]
            
            weight_map[weight_node] = [0.,0]
            for i in range(node_id):
                if nodes[i].output[0] == weight_scale:
                    val = np.frombuffer(nodes[i].attribute[0].t.raw_data, dtype=np.float32)[0]
                    if weight_node in weight_map.keys():
                        old_val = weight_map[weight_node][0]
                        if val > old_val:
                            weight_map[weight_node][0] = val
                    else: 
                        weight_map[weight_node][0] = val
                elif nodes[i].output[0] == weight_zp:
                    if unsigned_flag:
                        val = np.frombuffer(nodes[i].attribute[0].t.raw_data, dtype=np.uint8)[0]
                    else:
                        val = np.frombuffer(nodes[i].attribute[0].t.raw_data, dtype=np.int8)[0]
                    weight_map[weight_node][1] = val


    # Remove QuantizeLinear node
    for quant_node in quant_node_list:
        graph.node.remove(quant_node)


    # relink
    for node_id, node in enumerate(graph.node):
       for in_id, in_name in enumerate(node.input):
           if in_name in in_rename_map.keys():
               # set node input == removed node's input
               node.input[in_id] = in_rename_map[in_name]

    in_rename_map = {}
    # activation_map = {}
    dequant_node_list = []
    for node_id, node in enumerate(graph.node):
       if node.op_type == "DequantizeLinear":
           in_name = node.input[0]
           scale_name = node.input[1]
           zero_name = node.input[2]
           # print(scale_name)
           out_name = node.output[0]
           in_rename_map[out_name] = in_name
           # remove later
           dequant_node_list.append(node)
           scale_node_list.append(scale_name)
           zero_node_list.append(zero_name)

    # Remove DequantizeLinear node
    for dequant_node in dequant_node_list:
        graph.node.remove(dequant_node)

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

    return onnx_replica, activation_map, weight_map

def save_calib_cache_file_trt(cache_file, activation_map, headline='TRT-8XXX-EntropyCalibration2\n'):
    cache_file = cache_file.replace('.onnx', '_calibration.cache')
    with open(os.path.join(cache_file), 'w') as cfile:
        cfile.write(headline)
        for k, v in activation_map.items():
            if len(v) == 2:
                v = v[0]
            cfile.write("{}: {}\n".format(k, struct.pack('>f', v).hex()))

def save_calib_cache_file_snpe(cache_file, activation_map, weight_map, unsigned_flag):
    cache_file = cache_file.replace('.onnx', '_clip_ranges.json')
    clip_ranges = {}
    p_clip_ranges = {}
    for tensor_name, [scale, zeropoint] in activation_map.items():   
        maxbound = 127 + int(unsigned_flag)*128
        minbound = -128 + int(unsigned_flag)*128
        num_bits = 255
        clip_ranges[tensor_name] = [
                            {'bitwidth': 8,
                             'max': float((maxbound - zeropoint)*scale),
                             'min': float((minbound - zeropoint)*scale)
                             }
                        ]
    for tensor_name, [scale, zeropoint] in weight_map.items():   
        maxbound = 127 + int(unsigned_flag)*128
        minbound = -128 + int(unsigned_flag)*128
        p_clip_ranges[tensor_name] = [
                            {'bitwidth': 8,
                             'max': float((maxbound - zeropoint)*scale),
                             'min': float((minbound - zeropoint)*scale)
                             }
                        ]
    context = {'activation_encodings': clip_ranges, 'param_encodings': p_clip_ranges}
    with open(cache_file, 'w') as f:
            json.dump(context, f, indent=4)

def remove_qdq_nodes_from_qat_onnx(onnx_file, benckend='TRT',unsigned_flag = False):
    model = onnx.load(onnx_file)
    model_wo_qdq, activation_map, weight_map = onnx_remove_qdqnode(model,unsigned_flag)

    onnx_name, onnx_dir = os.path.basename(onnx_file), os.path.dirname(onnx_file)
    onnx_new_name = onnx_name.replace('.onnx', '_remove_qdq.onnx')
    onnx.save(model_wo_qdq, os.path.join(onnx_dir, onnx_new_name))
    if benckend == 'TRT':
        save_calib_cache_file_trt(os.path.join(onnx_dir, onnx_new_name), activation_map)
    elif benckend == 'SNPE':
        save_calib_cache_file_snpe(os.path.join(onnx_dir, onnx_new_name), activation_map, weight_map,unsigned_flag)
    else:
        NotImplementedError, "UnSupported BENCKEND"
def remove_qdq_nodes_from_onnx_ada(onnx_file, benckend='TRT',unsigned_flag = False):
    model = onnx.load(onnx_file)
    model_wo_qdq, activation_map, weight_map = onnx_remove_qdqnode_ada(model,unsigned_flag)

    onnx_name, onnx_dir = os.path.basename(onnx_file), os.path.dirname(onnx_file)
    onnx_new_name = onnx_name.replace('.onnx', '_remove_qdq.onnx')
    onnx.save(model_wo_qdq, os.path.join(onnx_dir, onnx_new_name))
    if benckend == 'TRT':
        save_calib_cache_file_trt(os.path.join(onnx_dir, onnx_new_name), activation_map)
    elif benckend == 'SNPE':
        save_calib_cache_file_snpe(os.path.join(onnx_dir, onnx_new_name), activation_map, weight_map,unsigned_flag)
    else:
        NotImplementedError, "UnSupported BENCKEND"


if __name__ == '__main__':

    onnx_file = sys.argv[1]
    model = onnx.load(onnx_file)
    model_wo_qdq, activation_map = onnx_remove_qdqnode(model)

    onnx_name, onnx_dir = os.path.basename(onnx_file), os.path.dirname(onnx_file)
    onnx_new_name = onnx_name.replace('.onnx', '_remove_qdq.onnx')
    onnx.save(model_wo_qdq, os.path.join(onnx_dir, onnx_new_name))
    cache_name = onnx_new_name.replace('.onnx', '_calibration.cache')
    save_calib_cache_file(os.path.join(onnx_dir, cache_name), activation_map)


    # onnx_fuse = onnx_conv_horizon_fuse(model)
    # onnx_name, onnx_dir = os.path.basename(onnx_file), os.path.dirname(onnx_file)
    # onnx_new_name = onnx_name.replace('.onnx', '_fuse_horizon.onnx')
    # onnx.save(onnx_fuse, os.path.join(onnx_dir, onnx_new_name))

    # onnx_insert = onnx_add_insert_qdqnode(model)
    # model_wo_qdq, activation_map = onnx_remove_qdqnode(onnx_insert)
    # onnx_name, onnx_dir = os.path.basename(onnx_file), os.path.dirname(onnx_file)
    # onnx_new_name = onnx_name.replace('.onnx', '_remove_qdq.onnx')
    # onnx.save(onnx_insert, os.path.join(onnx_dir, onnx_new_name))
    # cache_name = onnx_new_name.replace('.onnx', '_add_insert_qdq_calibration.cache')
    # save_calib_cache_file(cache_name, activation_map)
