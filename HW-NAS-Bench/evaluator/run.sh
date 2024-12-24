#!/bin/bash

python3 train.py --device edgegpu
sleep 30

python3 train.py --device edgetpu
sleep 30

python3 train.py --device eyeriss
sleep 30

python3 train.py --device fpga
sleep 30

python3 train.py --device pixel3
sleep 30

python3 train.py --device raspi4
sleep 30