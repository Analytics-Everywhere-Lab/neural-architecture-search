from typing import Union, Tuple
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from metrics import *
from scipy.stats import mode

from nas_spaces import NASSpaceBase, Exemplar

# ADDITIONAL IMPORTS
import os 
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(root_dir)
from XAutoDL.xautodl.models import get_cell_based_tiny_net

# THIS VERY REDUNDANT PIECE OF CODE
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
    
    return operation_type


"""
    Dictionary mapping metric names to their implementation.
    
    Each metric function is expected to accept 4 arguments:
        - The model (torch.nn.Module)
        - An input batch (torch.Tenor)
        - A target tensor (labels for the input batch (torch.Tensor)
        - The device (torch.device)
    
    If the actual implementation doesn't need some of the args a lambda function can be used
"""
METRIC_NAME_MAP = {
    # log(x)
    'logsynflow': compute_synflow_per_weight,
    # x
    'synflow': lambda n, inputs, targets, dev: compute_synflow_per_weight(n, inputs, targets, dev, remap=None),

    'params': lambda n, _1, _2, _3: count_params(n) / 1e6,
    'macs': lambda n, inp, _1, _2: get_macs_and_params(n, inp.shape)[0],
    'naswot': compute_naswot_score,
}





def get_random_population(api: NASSpaceBase, N, generation=0):
    exemplars = [api.random().set_generation(generation) for _ in range(N)]
    return exemplars


def population_init(api, N, max_latency=float('inf'), max_params=float('inf'), analyzer=None, start=0.0, metrics=None):
    population = []
    while len(population) < N:
        new = get_random_population(api, 1)[0]
        if is_feasible(new, max_latency, max_params):
            population.append(new)
        if analyzer:
            analyzer.update(population, start, metrics)
    return population


def is_feasible(exemplar, max_latency=float('inf'), max_params=float('inf')):
    cost = exemplar.get_cost_info()
    if cost['latency'] <= max_latency and cost['params'] <= max_params:
        return True
    return False


def compute_tfm(exemplar: Exemplar,
                batch_or_loader: Union[Tuple[torch.Tensor, torch.Tensor], DataLoader],
                device: torch.device,
                metrics: Tuple = tuple(METRIC_NAME_MAP.keys()),
                ignore_errors: bool = False):
    if not exemplar.born:
        network = exemplar.space.get_network(exemplar, device=device)
        network = network.to(device)

        metric_trials = {}
        metric_times = {}

        init_time = 0
        for i in range(3):
            if isinstance(batch_or_loader, tuple):
                inputs, targets = batch_or_loader
            elif isinstance(batch_or_loader, DataLoader):
                inputs, targets = next(iter(batch_or_loader))
                inputs, targets = inputs.to(device), targets.to(device)
            else:
                raise ValueError("Invalid argument")

            start_time = time.time()
            network = init_model(network)
            end_time = time.time()
            init_time += (end_time - start_time)

            for metric_name in metrics:
                start_time = time.time()
                try:
                    val = METRIC_NAME_MAP[metric_name](network, inputs, targets, device)
                except RuntimeError as ex:
                    if ignore_errors:
                        # In case of errors set the value to None but keep running!
                        val = None
                    else:
                        raise ex
                end_time = time.time()
                if metric_name not in metric_trials:
                    metric_trials[metric_name] = []
                    metric_times[metric_name] = []
                metric_trials[metric_name].append(val)
                metric_times[metric_name].append(end_time - start_time)

        for metric_name in metrics:
            if None in metric_trials[metric_name]:
                metric_trials[metric_name] = None
                metric_times[metric_name] = None
            else:
                metric_trials[metric_name] = np.mean(metric_trials[metric_name])
                metric_times[metric_name] = np.sum(metric_times[metric_name])

        return metric_trials, metric_times, init_time


def return_top_k(exemplars, K=3, metric_names=[]):
    exemplars = [exemplar for exemplar in exemplars if exemplar.born]

    values_dict = {}
    for metric_n in metric_names:
        if metric_n == 'skip':
            values_dict['skip'] = np.array([exemplar.skip() for exemplar in exemplars])
        else:
            values_dict[metric_n] = np.array([exemplar.get_metric(metric_n) for exemplar in exemplars])

    scores = np.zeros(len(exemplars))
    scores_dict = {}
    for metric_n in metric_names:
        if metric_n == 'ntk':
            values_dict[metric_n] = 1 - values_dict[metric_n] / (np.max(np.abs(values_dict[metric_n])) + 1e-9)
        
        if values_dict[metric_n].size == 0 or np.isnan(values_dict[metric_n]).all():
            # Handle the empty or all-NaN case. Happens with the memory constraint
            # print('NaN')
            values_dict[metric_n] = np.zeros_like(values_dict[metric_n])
        else:
            values_dict[metric_n] = values_dict[metric_n] / (np.max(np.abs(values_dict[metric_n])) + 1e-9)
        
        scores_dict[metric_n] = values_dict[metric_n]
        scores += values_dict[metric_n]

    for idx, (exemplar, rank) in enumerate(zip(exemplars, scores)):
        exemplar.rank = rank

    exemplars.sort(key=lambda x: -x.rank)
    return exemplars[:K]


def get_max_accuracy(api: NASSpaceBase, max_latency: float, max_params: float):
    best, best_accuracy = None, 0.0
    for idx in range(len(api)):
        info = api.get_cost_info(idx)
        if info['latency'] <= max_latency and info['params'] <= max_params:
            accuracy = api.get_accuracy(api[idx])
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best = idx
                print(best, best_accuracy)
    return best_accuracy


def kaiming_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def init_model(model):
    model.apply(kaiming_normal)
    return model


def dictionary_update(exemplars, history, replace=True):
    if replace:
        # Update already seen genotypes
        history.update({exemplar.genotype: exemplar for exemplar in exemplars})
    else:
        # Add new genotypes
        history.update({exemplar.genotype: exemplar for exemplar in exemplars if exemplar.genotype not in history})
        # For already seen exemplars, just update the generation
        for exemplar in exemplars:
            history[exemplar.genotype].generation = exemplar.generation

    # This is also removing exemplars with same genotype from population
    current_genotypes = set([exemplar.genotype for exemplar in exemplars])
    exemplars = [history[genotype] for genotype in current_genotypes]
    return exemplars, history


def evaluate_exemplar(exemplar, scaler, encoder, tokenizer):
    global operation_type, conv_params, dense_units, seen, node_dict, node_id
    network = exemplar.get_network(device=torch.device("cpu"))
    net_str = get_net_string(network)

    # Flush the variables so they don't get accummulated
    operation_type, conv_params, dense_units = '', '', '' 
    seen = set()
    node_dict = {}
    node_id = 0

    tokenized_test_str = tokenizer(net_str, max_length=None, truncation=True, padding='max_length', return_tensors='pt').input_ids
    _, prediction_outputs = encoder(tokenized_test_str)
    prediction_outputs = prediction_outputs.detach().cpu().numpy()
    prediction_outputs = scaler.inverse_transform(prediction_outputs)
    # print(prediction_outputs[:, 1].item())
    return prediction_outputs[:, 1].item()

def clean_history(history, max_params, max_latency, scaler, encoder, tokenizer): # DEFINE THE CONSTRAINTS HERE
    computed_latency = []
    for _, exemplar in history.items():
        pred_latency = evaluate_exemplar(exemplar, scaler, encoder, tokenizer)
        computed_latency.append(pred_latency)
        

    # Original code - modified
    return {genotype: exemplar for i, (genotype, exemplar) in enumerate(history.items())
            if exemplar.get_cost_info()['params'] <= max_params    # max params and max latency are pre-defined values
            and computed_latency[i] <= max_latency}


def get_top_k_accuracies(exemplars, K=3, metrics=[]):
    best_K, acc = return_top_k(exemplars, K, metrics), []
    for exemplar in best_K:
        idx = exemplar.idx
        acc.append((idx, round(exemplar.get_accuracy(), 3)))
    return acc, [exemplar.genotype for exemplar in best_K], [exemplar.rank for exemplar in best_K]


def info(api, exemplars, history, start, metrics, max_latency, max_params, scaler, encoder, tokenizer):
    history_feasible = clean_history(history, max_params, max_latency, scaler, encoder, tokenizer).values()
    acc_history, genotypes_hist, ranks_hist = get_top_k_accuracies(history_feasible, K=3, metrics=metrics)
    exemplars_feasible = [exemplar for exemplar in exemplars if exemplar.born and
                          evaluate_exemplar(exemplar, scaler, encoder, tokenizer) <= max_latency and # Modification here
                          exemplar.get_cost_info()['params'] <= max_params]
    acc_pop, _, _ = get_top_k_accuracies(exemplars_feasible, K=3, metrics=metrics)
    accuracies = [exemplar.get_accuracy() for exemplar in history_feasible]
    # hardware_costs = [evaluate_exemplar(exemplar) for exemplar in history_feasible]

    metrics_time = api.total_metrics_time(history_feasible, metrics)
    search_time = time.time() - start
    total_time = metrics_time + search_time

    # This is not a correct number if we loose the constraints at the beginning
    print(f'\n|||| {len(history_feasible)} different cells explored..')
    # print(f'|||| Max acc: {round(max(accuracies), 2)}, Avg acc: {round(np.mean(accuracies), 2)}, Std acc: {round(np.std(accuracies), 2)}..') 
    
    # This modification below is because of the unstable nature of searching with memory
    if accuracies:
        print(f'|||| Max acc: {round(max(accuracies), 2)}, Avg acc: {round(np.mean(accuracies), 2)}, Std acc: {round(np.std(accuracies), 2)}..')
        # print(f'|||| Max cost: {round(max(hardware_costs), 2)}, Avg cost: {round(np.mean(hardware_costs), 2)}, Std cost: {round(np.std(hardware_costs), 2)}..')
    else:
        print("No accuracies available.")


    print(f'|||| Accuracies of top 3 visited cells: {acc_history}')
    print(f'|||| Ranks of top 3 visited cells: {ranks_hist}')
    print(f'|||| Genotype of top 3 visited cells:')
    for index, genotype in enumerate(genotypes_hist):
        print(f'     |||| {acc_history[index][0]} : {genotype}')
    print(f'|||| Accuracies of top 3 surviving cells: {acc_pop}')
    print(f'|||| Accuracies of top 3 surviving cells: {acc_pop}')
    print(f'|||| Search Time: {round(search_time / 60, 2)} minutes..')
    print(f'|||| Metrics Time: {round(metrics_time / 60, 2)} minutes..')
    print(f'|||| Total Time: {round(total_time / 60, 2)} minutes..')

    # return acc_history[0][1], total_time / 60
    return acc_history[0][1] if acc_history else np.nan, total_time / 60   # In the memory search, some searches find 0 architectures, return None is no architecture


def edit_distance(exemplar1, exemplar2):
    genes1 = genotype_to_gene_list(exemplar1.genotype)
    genes2 = genotype_to_gene_list(exemplar2.genotype)

    count = 0
    for gene1, gene2 in zip(genes1, genes2):
        if gene1 != gene2:
            count += 1
    return count


def genotype_to_gene_list(genotype):
    out = []
    levels = genotype.split('+')
    for level in levels:
        level = level.split('|')[1:-1]
        for i in range(len(level)):
            out.append(level[i])
    return out


def gene_list_to_genotype(gene_list):
    gene_list = gene_list[0][0]
    out = '|' + gene_list[0] + '|+|'
    for idx, gene in enumerate(gene_list[1:]):
        out += gene + '|'
        if idx == 1:
            out += '+|'
    return out


def mean_exemplar(exemplars, K=5, metrics=[]):
    _, genotypes_hist, _ = get_top_k_accuracies(exemplars, K=K, metrics=metrics)

    top_genotypes = [genotype_to_gene_list(genotype) for genotype in genotypes_hist]

    mean_genes = mode(top_genotypes, axis=0)
    mean_genotype = gene_list_to_genotype(mean_genes)
    return mean_genotype


class EarlyStop:
    def __init__(self, patience=20):
        super().__init__()
        self.patience = patience
        self.waiting = 0
        self.n_pop = 0

    def stop(self, history):
        n = len(history)
        if n >= self.n_pop:
            self.n_pop = n
            self.waiting = 0
        else:
            self.waiting += 1
            if self.patience >= self.waiting:
                print('\n   Early Stop...')
                return True
        return False
