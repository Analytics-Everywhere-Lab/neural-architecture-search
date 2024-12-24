# Receives the computational graph of a network and saves it as a string. 
# Temporary code and can be modified. See encode_nats.py for implementation


# the traversal has been adapted from the torchviz traversal method (https://github.com/szagoruyko/pytorchviz/blob/master/torchviz/dot.py#L102)
# gradient function can be one of two things: (https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/variable.h)
# /// 1. A `grad_fn`, if the variable is in the interior of the graph. This is the
# ///    gradient of the function that produced the variable.
# /// 2. A `grad_accumulator`, if the variable is a leaf, which accumulates a
# ///    scalar gradient value into its `grad` variable.

import torch
from torchviz import make_dot
import torchvision.models as models
import gc

input_tensor = torch.randn((64, 3, 32, 32), requires_grad=True)


def get_fn_name(fn):
    name = str(type(fn).__name__)
    if 'AccumulateGrad' not in name and 'Mul' not in name and 'T' not in name and 'View' not in name and 'Mean' not in name:
        if 'Backward' in name:
            name = name.replace('Backward0', '')
            # print(name)
        # if name == 'Add':
        #     return None
        return name

def get_string(node, seen, node_id, conv_params, dense_units, operation_type, node_dict):
    # global node_id, conv_params, dense_units, operation_type
    assert not torch.is_tensor(node) # Ensure we're traversing nodes, not tensors
    if node in seen:  
        return
    seen.add(node)

    if hasattr(node, 'next_functions'):    
        for u in node.next_functions:
            if u[0] is not None:
                get_string(u[0], seen, node_id, conv_params, dense_units, operation_type, node_dict)

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

    # return operation_type



def get_net_string(network):
    # global operation_type, conv_params, dense_units, seen
    seen = set()
    node_dict = {}
    node_id = 0
    operation_type = '' 
    conv_params = ''
    dense_units = ''
    node = None
    output = network(input_tensor)

    # make_dot(output, params=dict(list(network.named_parameters())), show_attrs=True).render('resnet18', format='png')
    if isinstance(output, (torch.Tensor)):
        node =  output.grad_fn
    else:
        node = output[0].grad_fn # NATSBench architectures output a tuple i.e output, label
    
    operation_type = get_string(node, seen, node_id, conv_params, dense_units, operation_type, node_dict)

    print(operation_type)
    
    operation_type, conv_params, dense_units = '', '', '' # Reset vars so they don't accummulate
    seen = set()
    return operation_type

model = models.resnet18(weights=None)

get_net_string(model)
