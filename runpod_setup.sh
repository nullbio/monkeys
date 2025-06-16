#!/bin/bash
# RunPod setup script for A100 training

# 1. First, create a RunPod account and add credits
# 2. Go to https://runpod.io/console/gpu-cloud
# 3. Select "A100 80GB" (~$1.89/hour)
# 4. Choose PyTorch 2.0 template
# 5. Launch the pod

# Once connected via SSH or Web Terminal:

# Clone your repo
git clone https://github.com/nullbio/adaptive-llm-agents.git
cd adaptive-llm-agents

# Or upload via runpod CLI:
# pip install runpodctl
# runpodctl send adaptive-llm-agents/ pod-id:/workspace/

# Install dependencies
pip install -r requirements.txt

# Start training with more data for better results
sed -i 's/train\[:10000\]/train[:50000]/' train_agents_efficient.py

# Run training (will take ~6-8 hours on A100)
python train_agents_efficient.py --num-epochs 5

# Monitor GPU usage
watch -n 1 nvidia-smi

# Download results before stopping pod
# runpodctl receive pod-id:/workspace/adaptive-llm-agents/checkpoints ./checkpoints
