#!/bin/bash
# RunPod Multi-Node Cluster Setup

# For RunPod Instant Clusters:
# 1. Go to RunPod Console > Clusters
# 2. Create new cluster with 5x RTX 4090 nodes
# 3. Use PyTorch 2.0 template
# 4. Enable "Fast Networking" for inter-node communication

# On the master node (node 0):
cd /workspace
git clone <your-repo>
cd adaptive-llm-agents
pip install -r requirements.txt

# Set master node address (RunPod provides this)
export MASTER_ADDR=<master-node-internal-ip>
export MASTER_PORT=29500

# On ALL nodes, run:
torchrun \
    --nproc_per_node=1 \
    --nnodes=5 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_distributed.py

# Or use RunPod's built-in orchestration:
runpodctl distributed-run --nodes=5 "python train_distributed.py"