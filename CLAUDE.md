# Adaptive Multi-Agent LLM Project Instructions

## Project Overview
This project implements an adaptive LLM system using parallel Qwen-3-4B models with dynamic behavior adjustment based on learned emotion signals. The system trains multiple agents with different token frequency biases and navigates between them based on user sentiment.

## Key Implementation Guidelines

### 1. Efficiency First
- Always implement shared forward passes for all parallel agents
- Use differential weight storage (store deltas, not full weights)
- Leverage harmonic analysis for compression
- Minimize redundant computations

### 2. Meta-Signal Learning
- The system should learn to interpret arbitrary emotion signaling conventions
- Example: "---" prefix = -3 unhappiness, "+++" = +3 happiness
- Do not hard-code emotion detection - let it emerge from training
- Support custom signal conventions defined in prompts

### 3. Smooth Gradients
- All transitions between agents should be smooth and continuous
- Decay functions are learned, not fixed
- Support both temporary adjustments and baseline shifts
- Each agent has unique decay dynamics

### 4. Architecture Principles
- Attention heads can route to different agents dynamically
- Multiple agents can be partially active simultaneously
- Frequency biases focus on emotional/sentiment-bearing tokens
- Start with basic emotions, expand to complex states later

### 5. Development Phases
1. Foundation: Base model + frequency bias mechanism
2. Emotion & Adaptation: Vector extraction + basic adjustment
3. Advanced Dynamics: RL for decay patterns + context awareness
4. Compression: Differential storage + harmonic analysis
5. Advanced Features: Per-head routing + real-time refinement

## Important Reminders
- Test with diverse emotion signaling conventions
- Monitor computational efficiency constantly
- Ensure smooth interpolation between agents
- Keep differential storage consistent
- Validate decay learning convergence
- Remember to use cuda consistently, and the correct device, rather than CPU

## Memory Usage
Feel free to use `.claude-memories/` for storing intermediate results, learned patterns, or debugging information during development.