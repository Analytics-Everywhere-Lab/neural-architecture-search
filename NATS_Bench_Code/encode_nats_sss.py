# Encodes NATSBench architectures by traversing their autograd graphs. This is the final code to generate the dataset
# Figure out how to move the arch_to_str to another file and call as a function. The bottleneck is the recursion

import torch

import numpy as np
import pandas as pd
import csv
import gc
import os
import sys 

from torchviz import make_dot 

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(root_dir)

from XAutoDL.xautodl.models import get_cell_based_tiny_net
from nats_bench import create

from naswot import return_score
from cifar10loader import trainloader
# from utils import get_net_string

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


for api_source in ['sss']:
    api = create(None, api_source, fast_mode=True, verbose=False)
    searchspace = [i for i in range(len(api))]  

    accuracy = []
    architecture = []
    latency = []
    memory = []

    for dataset in ['ImageNet16-120']:
        for i, arch_id in enumerate(searchspace):
            net = api.get_net_config(arch_id, dataset)
            network = get_cell_based_tiny_net(net)

            net_str = get_net_string(network)
            architecture.append(net_str)
            # Flush the variables so they don't get accummulated
            operation_type, conv_type, dense_units = '', '', '' 
            seen = set()
            node_dict = {}
            node_id = 0

            info = api.get_more_info(arch_id, dataset)
            test_acc = info['test-accuracy']
            accuracy.append(test_acc)

            lat = api.get_cost_info(arch_id, dataset)['latency'] * 1000 # Latency in milliseconds to facilitate the training range
            latency.append(lat)

            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
            # net['num_classes'] = 10 
            network.to(device)
            max_memory = 0
            network.train(True)

            # Reset peak memory usage before training
            torch.cuda.reset_peak_memory_stats()
            for batch_no, data in enumerate(trainloader):
                
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()

                outputs, y = network(inputs)

                loss = loss_fn(y, labels)

                loss.backward()

                optimizer.step()

                # Max memory peaks from batch 2 onwards
                if (batch_no == 2):
                    max_memory = torch.cuda.max_memory_allocated() / (1024**2)
                    break
            
            memory.append(max_memory)            
            
            print(f'{dataset}, Architecture: {arch_id}, Accuracy: {test_acc}, Latency: {lat} Memory: {max_memory}')
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
        
        with open(f'profiled_nets/{api_source}_{dataset}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Architectures', 'Accuracy', 'Latency', 'Memory'])
            for arch, acc, la, mem in zip(architecture, accuracy, latency, memory):
                writer.writerow([arch, acc, la, mem])
