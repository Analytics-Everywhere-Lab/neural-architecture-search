#!/bin/bash

# python3 train.py --search_space tss --dataset ImageNet16-120 --metric Latency
# sleep 60
# python3 train.py --search_space tss --dataset ImageNet16-120 --metric Memory
# sleep 60

# python3 train.py --search_space sss --dataset cifar10 --metric Latency
# sleep 60
# python3 train.py --search_space sss --dataset cifar10 --metric Memory
# sleep 60


# python3 train.py --search_space sss --dataset cifar100 --metric Latency
# sleep 60
# python3 train.py --search_space sss --dataset cifar100 --metric Memory
# sleep 60

python3 encode_nats_sss.py

python3 train.py --search_space sss --dataset ImageNet16-120 --metric Latency
sleep 60
python3 train.py --search_space sss --dataset ImageNet16-120 --metric Memory
sleep 60
