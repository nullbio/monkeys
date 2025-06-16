#!/bin/bash
# Quick setup for distributed training on RunPod
# Run this in each pod's web terminal

# Get node rank from hostname or set manually
if [ -z "$NODE_RANK" ]; then
    # Try to extract from hostname
    NODE_RANK=$(hostname | grep -oE '[0-9]+$' || echo "0")
    echo "Detected NODE_RANK=$NODE_RANK (change if incorrect)"
fi

# Configuration
export MASTER_ADDR=100.65.13.140  # agent-0's IP
export MASTER_PORT=29500
export WORLD_SIZE=5
export RANK=$NODE_RANK

echo "Setting up Node $RANK of $WORLD_SIZE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# Clone your training code
cd /workspace
if [ ! -d "monkeys" ]; then
    # Replace with your actual repo
    git clone https://github.com/yourusername/monkeys.git
fi
cd monkeys

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers accelerate datasets

# Create the distributed training script if it doesn't exist
if [ ! -f "train_distributed.py" ]; then
    echo "Creating placeholder training script..."
    cat > train_distributed.py << 'EOF'
import torch
import torch.distributed as dist
import os

def main():
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"Node {rank} of {world_size} is ready!")
    
    # TODO: Add your actual training code here
    # For now, just verify distributed setup works
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
EOF
fi

echo "Ready to start training!"
echo "Run: python train_distributed.py"