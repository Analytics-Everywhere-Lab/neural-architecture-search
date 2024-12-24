# Use code on line 5 before launching any NATS code

#!/bin/bash
tar xvf nats_bench/NATS-tss-v1_0-3ffb9-simple.tar
export TORCH_HOME=~/.cache/torch
mv nats_bench/NATS-tss-v1_0-3ffb9-simple $TORCH_HOME

mv nats_bench/NATS-sss-v1_0-50262-simple $TORCH_HOME