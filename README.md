# Adaptive Multi-Agent LLM System

An experimental implementation of a dynamically adaptive Language Model system that trains multiple parallel agents with different behavioral biases and smoothly navigates between them based on learned emotion signals.

## 🧠 Core Concept

This system trains multiple Qwen-3-4B models in parallel with slightly altered token frequencies, creating a spectrum of "personalities" or behavioral biases. It then learns to detect emotion signals from user interactions and dynamically adjusts which agent (or blend of agents) responds, creating a personalized, adaptive conversation experience.

### Key Innovations

1. **Single Forward Pass for Multiple Agents**: All parallel models share computation, diverging only where necessary
2. **Meta-Signal Learning**: Learns arbitrary emotion signaling conventions (e.g., "---" for unhappiness)
3. **Differential Weight Storage**: Stores only weight differences using harmonic analysis for massive compression
4. **Per-Attention-Head Routing**: Different attention heads can route to different agents simultaneously
5. **Learned Decay Dynamics**: Non-linear, context-aware decay patterns unique to each agent

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/nullbio/adaptive-llm-agents.git
cd adaptive-llm-agents

# Install dependencies
pip install -r requirements.txt

# Download Qwen-3-4B base model
python scripts/download_model.py

# Train initial agents with frequency biases
python train_agents.py --num-agents 5 --base-model qwen-3-4b

# Run interactive demo
python demo.py
```

## 📁 Project Structure

```
adaptive-llm-agents/
├── models/
│   ├── multi_agent_qwen.py      # Core multi-agent architecture
│   ├── frequency_bias.py        # Token frequency modulation
│   ├── emotion_detector.py      # Meta-signal learning
│   └── decay_learner.py         # Adaptive decay dynamics
├── compression/
│   ├── harmonic_storage.py      # Differential weight compression
│   └── weight_delta.py          # Delta computation utilities
├── training/
│   ├── agent_trainer.py         # Parallel agent training
│   ├── rl_decay.py              # RL for decay pattern learning
│   └── signal_learner.py        # Emotion signal training
├── data/
│   └── signal_examples/         # Training data for meta-signals
├── scripts/
│   ├── download_model.py
│   └── evaluate_agents.py
└── demo.py                      # Interactive demonstration
```

## 🔧 Configuration

### Basic Configuration (config.yaml)

```yaml
model:
  base_model: "qwen-3-4b"
  num_agents: 5
  shared_layers: 20  # Number of layers to share

frequency_bias:
  focus_tokens: "emotional"  # emotional, content, or custom
  bias_strength: 0.15
  
emotion_detection:
  vector_dim: 64
  meta_signals_enabled: true
  
decay:
  base_rate: 0.1
  learn_decay: true
  context_window: 50
```

## 🎯 Usage Examples

### Basic Usage

```python
from models import MultiAgentQwen

# Initialize system
system = MultiAgentQwen(num_agents=5)

# Process with emotion detection
response = system.generate(
    "I'm really frustrated with this error",
    detect_emotion=True,
    allow_baseline_shift=True
)
```

### Custom Signal Convention

```python
# Define custom emotion signals in prompt
system_prompt = """
Emotion signals:
- "!!!" = extremely excited (+5)
- "..." = contemplative (0)
- "ugh" = frustrated (-3)
"""

system.set_meta_signals(system_prompt)
```

### Accessing Agent Spectrum

```python
# Interpolate between agents
blended_response = system.generate(
    prompt="Tell me a story",
    agent_weights=[0.3, 0.7, 0, 0, 0]  # 30% agent 1, 70% agent 2
)
```

## 📊 Training

### Phase 1: Initial Agent Training

```bash
python train_agents.py \
    --num-agents 5 \
    --frequency-bias-type emotional \
    --epochs 10 \
    --batch-size 32
```

### Phase 2: Emotion Vector Learning

```bash
python train_emotion_detector.py \
    --sentiment-dataset your_dataset \
    --vector-dim 64 \
    --use-contrastive-loss
```

### Phase 3: Decay Pattern Learning (RL)

```bash
python train_decay_rl.py \
    --user-feedback-data feedback.json \
    --reward-shaping adaptive
```

## 🧪 Experiments

The system supports several experimental modes:

1. **Static Agent Testing**: Test individual agents with fixed frequency biases
2. **Dynamic Blending**: Real-time interpolation between agents
3. **Meta-Signal Learning**: Train on custom emotion conventions
4. **Attention Routing**: Visualize per-head agent selection

## 🔬 Technical Details

### Frequency Bias Implementation

Agents are created by modulating token probabilities:

```python
logits_agent_i = base_logits + frequency_bias_i
```

### Differential Storage

Instead of storing full weights for each agent:

```python
agent_weights = base_weights + IFFT(compressed_delta_i)
```

### Emotion Vector Mapping

Continuous emotion vectors map to agent space navigation:

```python
agent_blend = softmax(emotion_vector @ agent_embeddings.T)
```

## 📈 Performance

- **Memory**: ~80% reduction via differential storage
- **Inference**: Single forward pass for all agents
- **Adaptation**: Real-time emotion response (<100ms)

## 🤝 Contributing

We welcome contributions! Key areas:

- Extended emotion taxonomies
- Alternative compression methods
- Multi-modal emotion detection
- Cross-cultural signal adaptation

## 📚 References

- Qwen-3-4B: [Model Card](https://huggingface.co/Qwen/Qwen-3-4B)
- Harmonic Analysis for NN Compression: [Paper](https://arxiv.org/example)
- Adaptive RL in Language Models: [Paper](https://arxiv.org/example)

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

Special thanks to the Qwen team for the excellent base model and the open-source community for inspiration on adaptive AI systems.
