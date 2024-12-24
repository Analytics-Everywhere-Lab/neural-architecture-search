# Mofication to pass different constraints to the search

#!/usr/bin/env python3
import os.path

import numpy as np
import pandas as pd

# Additional imports
from sklearn.preprocessing import MinMaxScaler 
import torch
from encoder_utils import Encoder
from transformers import T5Tokenizer

from argparse import ArgumentParser
from search_modified import search
from our_baselines import freeREAMinus
from utils import seed_all, get_nas_archive_path
from nas_spaces import NATSBench, NASBench101
from analyzer import Analyzer

parser = ArgumentParser()
parser.add_argument('--algo', type=str, default='ours', choices=('ours', 'freeREA-'))
parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--max-flops', type=float, default=float('inf'))
parser.add_argument('--max-params', type=float, default=float('inf'))
parser.add_argument('--initial-pop', type=int, default=100)
parser.add_argument('--tournament-size', type=int, default=25)
parser.add_argument('--n-random', type=int, default=0)
parser.add_argument('--max-time', type=float, default=0.75, help="Maximum time in minutes")  # mins
parser.add_argument('--repeat', type=int, default=30, help="Number of repetitions")
parser.add_argument('--space', type=str, default='nats', choices=('nats', 'nasbench101'),
                    help="Search space")
parser.add_argument('--dataset', choices=('cifar10', 'cifar100', 'ImageNet16-120'), default='cifar10',
                    help="Image classification dataset. NasBench101 only supports cifar10")
parser.add_argument('--metrics-root', default='cached_metrics',
                    help="Position of pre-computed metrics")
parser.add_argument('--seed', type=int, default=0,
                    help="Random seed for reproducibility")
parser.add_argument('--metric', type=str, default='Latency', choices=('Latency, Memory'))
args = parser.parse_args()

space = args.space
archive_path = get_nas_archive_path(space)

seed_all(args.seed)

metrics_root = os.path.join(args.metrics_root, args.space)


# We want to scale the value to what the model was trained on and get its median as max constraint  
# ADDITIONAL CODE
df = pd.read_csv(f'../../NATS_Bench_Code/tss_{args.dataset}.csv')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
encoder = Encoder()
encoder.eval()

constraint = 0
constraint_list = [] 
if args.metric == 'Latency':
    encoder.load_state_dict(torch.load(f'../../NATS_Bench_Code/evaluator/saved_models/tss_{args.dataset}_Latency.pth')) # Latency
    constraint_list = list(zip(df['Accuracy'], df['Latency'])) 
    
if args.metric == 'Memory':
    encoder.load_state_dict(torch.load(f'../../NATS_Bench_Code/evaluator/saved_models/tss_{args.dataset}_Memory.pth')) # Memory
    constraint_list = list(zip(df['Accuracy'], df['Memory']))

constraint = np.mean(constraint_list)    
scaler = MinMaxScaler() 
scaled_constraint = scaler.fit_transform(constraint_list)

print(f'Constraint Metric: {constraint:.4f} {'MB' if args.metric == 'Memory' else 's'}')


if space == 'nats':
    api = NATSBench(archive_path, args.dataset, metric_root=metrics_root)
elif space == 'nasbench101':
    api = NASBench101(archive_path, metric_root=metrics_root, progress=True, verbose=True)
else:
    raise ValueError("This should never happen.")



accuracies_hist = []
times_hist = []
analyzer = Analyzer(api, args.dataset, args.algo)
for _ in range(args.repeat):
    analyzer.new_run()
    if args.algo == 'ours':
        top1, total_time = search(api,
                                  N=args.initial_pop,
                                  n=args.tournament_size,
                                #   max_flops=args.max_flops,
                                  max_latency=constraint,   # Should have changed it to max_metric. 
                                  max_params=args.max_params,  # And here
                                  max_time=args.max_time,
                                  n_random=args.n_random,
                                  analyzer=analyzer,
                                  scaler=scaler, # We need to pass the scaler because its range is adapted from the dataset
                                  encoder=encoder,
                                  tokenizer=tokenizer) 
    elif args.algo == 'freeREA-':
        top1, total_time = freeREAMinus(api,
                                        N=args.initial_pop,
                                        max_flops=args.max_flops,
                                        max_params=args.max_params,
                                        max_time=args.max_time,
                                        analyzer=analyzer)

    accuracies_hist.append(top1)
    times_hist.append(total_time)

if args.save:
    analyzer.save()

print("Accuracies:")
accuracies_hist = np.array(accuracies_hist)
print(f"    Mean: {np.nanmean(accuracies_hist)}")
print(f"     Std: {np.nanstd(accuracies_hist)}")

print("Times [min]")
times_hist = np.array(times_hist)
print(f"    Mean: {times_hist.mean()}")
print(f"     Std: {times_hist.std()}")
