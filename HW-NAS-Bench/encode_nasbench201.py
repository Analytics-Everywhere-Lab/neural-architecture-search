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
from nas_201_api.api_201 import NASBench201API
from hw_nas_bench_api.nas_201_models import get_cell_based_tiny_net

from cifar100loader import trainloader
from naswot import return_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Might not be needed. Decided not to profile memory for these. 
input_tensor = torch.randn((64, 3, 32, 32), requires_grad=True)
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


hw_api = HWAPI("HW-NAS-Bench-v1_0.pickle", search_space="nasbench201")

scores = []
memory = []
architectures = []
accuracies = []
edgegpu_latencies = []
edgetpu_latencies = []
raspi4_latencies = []
eyeriss_latencies = []
pixel3_latencies = []
fpga_latencies = []


length_of_idx = 22
possible_values = [0, 1]  
all_combinations = itertools.product(possible_values, repeat=length_of_idx)

nasbench_api = NASBench201API()

arch_id = 0
for dataset in ['ImageNet16-120']:
    while True:
    # for arch_id in range(15626):
        config = hw_api.get_net_config(arch_id, dataset)
        if config is None:
            break
        network = get_cell_based_tiny_net(config)
        net_str = get_net_string(network)
        architectures.append(net_str)
        operation_type, conv_type, dense_units = '', '', '' 
        seen = set()
        node_dict = {}
        node_id = 0

        info = nasbench_api.get_more_info(index=arch_id, dataset=dataset)
        test_acc = info['test-accuracy']
        accuracies.append(test_acc)
        # score = return_score(network, trainloader)
        # scores.append(score)

        HW_metrics = hw_api.query_by_index(arch_id, dataset)
        edgegpu_latencies.append(HW_metrics['edgegpu_latency'])
        edgetpu_latencies.append(HW_metrics['edgetpu_latency'])
        raspi4_latencies.append(HW_metrics['raspi4_latency'])
        eyeriss_latencies.append(HW_metrics['eyeriss_latency'])
        pixel3_latencies.append(HW_metrics['pixel3_latency'])
        fpga_latencies.append(HW_metrics['fpga_latency'])


        print(f'{dataset}, Architecture: {arch_id}, Accuracy: {accuracies[arch_id]}, Avg Latency: {HW_metrics['average_hw_metric']}')

        
        with open(f'nasbench201_{dataset}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Architectures', 'Accuracy', 'EdgeGPU_Lat', 'EdgeTPU_Lat', 'Raspi4_Lat', 'Eyeriss_Lat', 'Pixel3_Lat', 'FPGA_Lat'])
            for arch, acc, edgegpu, edgetpu, raspi4, eyeriss, pixel3, fpga in zip(architectures, accuracies, edgegpu_latencies, edgetpu_latencies, raspi4_latencies, eyeriss_latencies, pixel3_latencies, fpga_latencies):
                writer.writerow([arch, acc, edgegpu, edgetpu, raspi4, eyeriss, pixel3, fpga])

        arch_id += 1
