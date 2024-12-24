# Attempts a zero shot calculation of memory consumption. Inspired by Sohoni et al.
# Implementation CURRENTLY PAUSED because calc for activation is unsure

import torch
import torchvision.models as models
from torchviz import make_dot
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '' # CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' #GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from cifar10loader import trainloader

model = models.resnet152(weights=None).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, nesterov=True, momentum=0.9)
input_tensor = torch.randn((64, 3, 32, 32), requires_grad=True).to(device)
labels = torch.randint(0, 10, (64,)).to(device)

total_param_memory = 0
for param in model.parameters():
    param_memory = param.numel() * param.element_size()
    total_param_memory += param_memory
network_size = total_param_memory / (1024 ** 2)
print(network_size)

# print(optimizer.state_dict())
optimizer_memory = 0
optimizer_state_dict = optimizer.state_dict()
param_indices = optimizer_state_dict['param_groups'][0]['params']
model_params = list(model.parameters())
for param_idx in param_indices:
    param = model_params[param_idx]
    param_memory = param.numel() * param.element_size()
    optimizer_memory += param_memory

total_opt_size = 0
if isinstance(optimizer, torch.optim.SGD):
    total_opt_size = optimizer_memory * 2
elif isinstance(optimizer, torch.optim.Adam):
    total_opt_size = optimizer_memory * 3

## Add for other optimizers 
total_opt_size /= (1024 ** 2)
print(total_opt_size)

seen = set()
forward_pass_vars = []
backward_pass_vars = []

def tensor_memory(tensor):
    if tensor is not None:
        return (tensor.numel() * tensor.element_size())
    return 0

# the function below has been adapted from the torchviz traversal method (https://github.com/szagoruyko/pytorchviz/blob/master/torchviz/dot.py#L102)
# gradient function can be one of two things: (https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/variable.h)
# /// 1. A `grad_fn`, if the variable is in the interior of the graph. This is the
# ///    gradient of the function that produced the variable.
# /// 2. A `grad_accumulator`, if the variable is a leaf, which accumulates a
# ///    scalar gradient value into its `grad` variable.
# activation_memory = 0

activation_memory = []
def traverse_graph(node, current_node_memory):
    assert not torch.is_tensor(node) # Ensure we're traversing nodes, not tensors
    if node in seen:
        # activation_memory.append(node_memory)
        return
    seen.add(node)

    # Special case for ReLU which requires 1 bit per input element
    if 'Relu' in str(type(node).__name__):
        current_node_memory += (node._saved_result.numel() * node._saved_result.element_size() / 8)

    # gets the activation inputs for the backward pass
    for attr in dir(node):
        try:
            if not attr.startswith("_saved_"): # only attributes with _saved_ prefix get considered
                continue
            # print(attr)
            val = getattr(node, attr)
            seen.add(val)
            # attr = attr[len("_saved_")]
            if torch.is_tensor(val):
                # print(attr)
                current_node_memory += tensor_memory(val)
                # print(current_node_memory)
                # activation_memory.append(node_memory)
                # backward_pass_vars.append(val)
            if isinstance(val, tuple):
                for i, t in enumerate(val):
                    if torch.is_tensor(t):
                        current_node_memory += tensor_memory(t)
                        # activation_memory.append(node_memory)
                        # backward_pass_vars.append(t)
        except Exception as e:
            print(e)
    
    # gets the inputs for the forward pass
    if hasattr(node, 'variable'):
        # if grad_accumulator, add the node for .variable
        var = node.variable
        current_node_memory += tensor_memory(var)
        # activation_memory.append(node_memory)
        # forward_pass_vars.append(var)
        seen.add(var)
    
    # recurse the traversal
    if hasattr(node, 'next_functions'):
        for u in node.next_functions:
            if u[0] is not None:
                traverse_graph(u[0], current_node_memory)
    
    # for forward pass check for activations stored for backprop. 
    # note: this used to show .saved_tensors in pytorch0.2, but stopped
    # working* as it was moved to ATen and Variable-Tensor merged 
    if hasattr(node, 'saved_tensor'):
        for t in node.saved_tensors:
            current_node_memory += tensor_memory(t)
            seen.add(t)
    activation_memory.append(current_node_memory)
    return(max(activation_memory) / (1024**2))

output = model(input_tensor)
print(traverse_graph(output.grad_fn, 0))