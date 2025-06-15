#!/bin/bash
# Vast.ai setup for A100 training

# 1. Create account at vast.ai
# 2. Add credits ($30-50 recommended)
# 3. Search for: A100 80GB, PyTorch, High reliability
# 4. Rent instance (~$0.80-1.50/hour)

# Connect via SSH (vast.ai will provide command)
# ssh -p PORT root@IP

# Setup environment
cd /workspace
git clone https://github.com/yourusername/adaptive-llm-agents.git
cd adaptive-llm-agents

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Configure for A100 (more aggressive settings)
cat > config_a100.yaml << EOF
model:
  base_model: "Qwen/Qwen2.5-3B"
  num_agents: 5
  shared_layers: 20
  device: "cuda"
  precision: "fp16"
  gradient_checkpointing: false  # A100 has enough memory

frequency_bias:
  bias_type: "emotional"
  bias_strength: 0.35
  temperature_range: [0.7, 1.3]

training:
  batch_size: 16  # Larger batch on A100
  learning_rate: 1e-4
  num_epochs: 10
  warmup_steps: 500
  gradient_accumulation_steps: 1
  mixed_precision: true

compression:
  method: "harmonic"
  compression_ratio: 0.1
  frequency_components: 2000
EOF

# Run training
python train_agents_efficient.py --config config_a100.yaml --num-epochs 10

# Save checkpoints to persistent storage
tar -czf checkpoints_$(date +%Y%m%d_%H%M%S).tar.gz checkpoints/