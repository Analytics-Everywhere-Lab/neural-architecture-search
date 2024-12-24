# Encodes NATSBench architectures by traversing their autograd graphs. This is the final code to generate the dataset
# Integrate memory and score calculation from profile_nets.py and delete most of the code

import torch
from torchvision.models import resnet18

import numpy as np
import pandas as pd
import csv
import gc
import os
import itertools

from torchviz import make_dot 

from hw_nas_bench_api import HWNASBenchAPI as HWAPI
from hw_nas_bench_api.fbnet_models import FBNet_Infer

from cifar100loader import trainloader
from naswot import return_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_tensor = torch.randn((64, 3, 32, 32), requires_grad=True)

# These variables are hard to be managed when passed from another file. Else I would have put the functions in utils and just called them
seen = set()
node_dict = {}
node_id = 0
operation_type = '' 
conv_params = ''
dense_units = ''

def get_fn_name(fn):
    name = str(type(fn).__name__)
    if 'AccumulateGrad' not in name and 'Mul' not in name and 'T' not in name and 'View' not in name and 'Mean' not in name:
        if 'Backward' in name:
            name = name.replace('Backward0', '')
            # print(name)
        # if name == 'Add':
        #     return None
        return name

def get_string(node):
    global node_id, conv_params, dense_units, operation_type
    assert not torch.is_tensor(node) # Ensure we're traversing nodes, not tensors
    if node in seen:  
        return
    seen.add(node)

    if hasattr(node, 'next_functions'):    
        for u in node.next_functions:
            if u[0] is not None:
                get_string(u[0])

    if hasattr(node, 'variable'):
            var = node.variable
            tensor_shape = var.shape
            if len(tensor_shape) == 4: # Convolution tensors are 4D
                # print(tensor_shape)
                num_filters = tensor_shape[0]
                filters = tensor_shape[-2:] # The last two elements represent the type of filters
                # print(filters)
                conv_params = f':(num_filters={num_filters}, kernel_size={filters[0]})'

            if len(tensor_shape) == 2: # Dense layer tensors are 2D
                units = tensor_shape[-1:] # number of units in the dense layer
                dense_units = f':(num_units={units[0]})'

    fn_name = get_fn_name(node)
    child_nodes = []
    if fn_name is not None:    
        node_id += 1 
        node_dict.update({node_id: node}) # Build the dictionary for determining the child nodes for skip connections
        
        # Gets a list of indices of the next nodes
        for u in node.next_functions:
            if u[0] is not None:
                key = next((k for k, v in node_dict.items() if v == u[0]), None)  # VERY INEFFICIENT STEP. Traverses the entire dict to find the key for a given value 
                child_nodes.append(key)
       
        # Clean up the child_nodes list to remove None values
        child_nodes = list(filter(lambda x: x is not None, child_nodes))

        if 'Convolution' in fn_name:
            operation_type += '|' + f'{str(node_id)}_' + fn_name + conv_params
        if 'Addmm' in fn_name:
            operation_type += '|' + f'{str(node_id)}_' + 'Dense' + dense_units
        if 'Convolution' not in fn_name and 'Addmm' not in fn_name:
            operation_type += '|' + f'{str(node_id)}_' + fn_name + (f':{child_nodes[1]}~{child_nodes[0]}' if len(child_nodes)>1 else '') 

    return 


def get_net_string(network):
    global operation_type, conv_params, dense_units, seen
    node = None
    output = network(input_tensor)

    # make_dot(output, params=dict(list(network.named_parameters())), show_attrs=True).render('resnet18', format='png')
    if isinstance(output, (torch.Tensor)):
        node =  output.grad_fn
    else:
        node = output[0].grad_fn # NATSBench architectures output a tuple i.e output, label
    
    get_string(node)
    
    # operation_type, conv_params, dense_units = '', '', '' # Reset vars so they don't accummulate
    # seen = set()
    return operation_type


hw_api = HWAPI("HW-NAS-Bench-v1_0.pickle", search_space="fbnet")

scores = []
memory = []
architectures = []
edgegpu_latencies = []
raspi4_latencies = []
eyeriss_latencies = []
pixel3_latencies = []
fpga_latencies = []


length_of_idx = 22
possible_values = [0, 1]  
all_combinations = itertools.product(possible_values, repeat=length_of_idx)

for i, idx in enumerate(all_combinations):
    arch_id = list(idx)
    config = hw_api.get_net_config(arch_id, 'cifar100')
    if config is None:
        break
    network = FBNet_Infer(config)
    net_str = get_net_string(network)
    architectures.append(net_str)
    operation_type, conv_type, dense_units = '', '', '' 
    seen = set()
    node_dict = {}
    node_id = 0

    # score = return_score(network, trainloader)
    # scores.append(score)

    HW_metrics = hw_api.query_by_index(arch_id, "cifar100")
    edgegpu_latencies.append(HW_metrics['edgegpu_latency'])
    raspi4_latencies.append(HW_metrics['raspi4_latency'])
    eyeriss_latencies.append(HW_metrics['eyeriss_latency'])
    pixel3_latencies.append(HW_metrics['pixel3_latency'])
    fpga_latencies.append(HW_metrics['fpga_latency'])


    # WE DON'T DO MEMORY FOR FBNET BECAUSE WE WANT TO EVALUATE OUR APPROACH ON A SINGLE OBJECTIVE
    # loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
     
    # network.to(device)
    # max_memory = 0
    # network.train(True)
    # batch_mem = []

    # for batch_no, data in enumerate(trainloader):
    #     max_memory = torch.cuda.memory_allocated()
    #     batch_mem.append(max_memory)

    #     inputs, labels = data[0].to(device), data[1].to(device)
    #     max_memory = torch.cuda.memory_allocated()
    #     batch_mem.append(max_memory)

    #     optimizer.zero_grad()
    #     max_memory = torch.cuda.memory_allocated()
    #     batch_mem.append(max_memory)

    #     y = network(inputs)
    #     max_memory = torch.cuda.memory_allocated()
    #     batch_mem.append(max_memory)

    #     loss = loss_fn(y, labels)
    #     max_memory = torch.cuda.memory_allocated()
    #     batch_mem.append(max_memory)

    #     loss.backward()
    #     max_memory = torch.cuda.memory_allocated()
    #     batch_mem.append(max_memory)

    #     optimizer.step()
    #     max_memory = torch.cuda.memory_allocated()
    #     batch_mem.append(max_memory)

    #     break
    # # print(max(batch_mem) / (1024**2))
    # memory.append(max(batch_mem) / (1024**2))
    print(f'Architecture: {i}, Avg Latency: {HW_metrics['average_hw_metric']}')
    
    torch.cuda.empty_cache()
    gc.collect()

    
    with open('fbnet_lat.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Architectures', 'EdgeGPU_Lat', 'Raspi4_Lat', 'Eyeriss_Lat', 'Pixel3_Lat', 'FPGA_Lat'])
        for arch, edgegpu, raspi4, eyeriss, pixel3, fpga in zip(architectures, edgegpu_latencies, raspi4_latencies, eyeriss_latencies, pixel3_latencies, fpga_latencies):
            writer.writerow([arch, edgegpu, raspi4, eyeriss, pixel3, fpga])

