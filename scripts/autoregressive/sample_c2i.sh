# !/bin/bash
# set -x

torchrun --nnodes=1 --nproc_per_node=4 --node_rank=0   \
        --master_port=12345 test_net.py         \
        "$@"
