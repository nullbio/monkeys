import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Tuple, Dict
import numpy as np
from .frequency_bias import FrequencyBias
from .emotion_detector import EmotionDetector
from .decay_learner import DecayLearner


class MultiAgentQwen(nn.Module):
    """
    Multi-agent Qwen model with shared forward pass and dynamic agent blending.
    All agents share most computation, diverging only in final layers.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B",
        num_agents: int = 5,
        shared_layers: int = 20,
        device: str = "cuda",
        config: Optional[Dict] = None
    ):
        super().__init__()
        self.num_agents = num_agents
        self.shared_layers = shared_layers
        self.device = device
        self.config = config or {}
        
        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        
        # Freeze shared layers
        self._freeze_shared_layers()
        
        # Initialize components
        self.frequency_biases = nn.ModuleList([
            FrequencyBias(
                vocab_size=self.tokenizer.vocab_size,
                bias_strength=self.config.get('bias_strength', 0.15),
                agent_id=i
            ) for i in range(num_agents)
        ])
        
        self.emotion_detector = EmotionDetector(
            model_dim=self.base_model.config.hidden_size,
            output_dim=self.config.get('emotion_vector_dim', 64)
        ).to(device)
        
        self.decay_learner = DecayLearner(
            num_agents=num_agents,
            context_dim=self.base_model.config.hidden_size
        ).to(device)
        
        # Agent blending weights (current state)
        self.register_buffer('agent_weights', torch.ones(num_agents) / num_agents)
        self.register_buffer('agent_baselines', torch.ones(num_agents) / num_agents)
        
        # Per-attention-head routing
        self.head_router = nn.Linear(
            self.base_model.config.hidden_size,
            self.base_model.config.num_attention_heads * num_agents
        ).to(device)
        
    def _freeze_shared_layers(self):
        """Freeze layers that are shared across all agents."""
        # Get transformer layers
        if hasattr(self.base_model, 'transformer'):  # Qwen architecture
            layers = self.base_model.transformer.h
        elif hasattr(self.base_model, 'model'):  # Some variants
            layers = self.base_model.model.layers
        else:
            raise ValueError("Unknown model architecture")
            
        # Freeze shared layers
        for i in range(min(self.shared_layers, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = False
                
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_emotion_routing: bool = True,
        return_all_agents: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all agents in parallel.
        
        Returns:
            Dictionary containing:
            - logits: Blended logits if return_all_agents=False, else all agent logits
            - emotion_vector: Current emotion state
            - agent_weights: Current blending weights
            - hidden_states: Final hidden states
        """
        batch_size = input_ids.shape[0]
        
        # Get base model outputs up to shared layers
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        
        hidden_states = outputs.hidden_states[-1]  # Final layer hidden states
        base_logits = outputs.logits
        
        # Detect emotion if enabled
        emotion_vector = None
        if use_emotion_routing:
            emotion_vector = self.emotion_detector(hidden_states, attention_mask)
            self._update_agent_weights(emotion_vector, hidden_states)
        
        # Apply per-agent frequency biases
        agent_logits = []
        for i, bias in enumerate(self.frequency_biases):
            # Apply frequency bias to logits
            biased_logits = bias(base_logits, hidden_states)
            agent_logits.append(biased_logits)
        
        agent_logits = torch.stack(agent_logits, dim=1)  # [batch, agents, seq, vocab]
        
        # Blend agents based on current weights
        if return_all_agents:
            final_logits = agent_logits
        else:
            # Weighted blend of agent logits
            weights = self.agent_weights.view(1, -1, 1, 1)  # [1, agents, 1, 1]
            final_logits = (agent_logits * weights).sum(dim=1)
        
        # Optional: Per-attention-head routing
        if hasattr(self, 'use_head_routing') and self.use_head_routing:
            head_weights = self.head_router(hidden_states)  # [batch, seq, heads*agents]
            head_weights = head_weights.view(batch_size, -1, self.base_model.config.num_attention_heads, self.num_agents)
            head_weights = torch.softmax(head_weights, dim=-1)
            # Apply head-specific routing (implementation depends on model architecture)
        
        return {
            'logits': final_logits,
            'emotion_vector': emotion_vector,
            'agent_weights': self.agent_weights.clone(),
            'hidden_states': hidden_states,
            'all_agent_logits': agent_logits if return_all_agents else None
        }
    
    def _update_agent_weights(self, emotion_vector: torch.Tensor, context: torch.Tensor):
        """Update agent blending weights based on emotion and decay dynamics."""
        # Initialize emotion projection if needed
        if not hasattr(self, '_emotion_to_agent_projection'):
            self._emotion_to_agent_projection = nn.Parameter(
                torch.randn(self.config.get('emotion_detection', {}).get('vector_dim', 64), self.num_agents) * 0.1
            ).to(self.device)
        
        # Map emotion to agent space
        emotion_affinity = torch.matmul(emotion_vector, self._emotion_to_agent_projection)
        emotion_affinity = torch.softmax(emotion_affinity, dim=-1).mean(dim=0)  # Average across batch
        
        # Get decay parameters for current context
        decay_params = self.decay_learner(self.agent_weights, context.mean(dim=1))
        
        # Update weights with decay
        target_weights = emotion_affinity
        decay_rate = decay_params['decay_rate']
        
        # Ensure decay_rate is a scalar and on the same device
        if decay_rate.dim() > 0:
            decay_rate = decay_rate.mean()
        
        # Ensure all tensors are on the same device
        decay_rate = decay_rate.to(self.agent_weights.device)
        target_weights = target_weights.to(self.agent_weights.device)
        
        self.agent_weights = (
            decay_rate * self.agent_weights +
            (1 - decay_rate) * target_weights
        )
        
        # Check for baseline shift
        if decay_params['shift_baseline'] and emotion_affinity.max() > self.config.get('baseline_shift_threshold', 0.8):
            self.agent_baselines = 0.9 * self.agent_baselines + 0.1 * self.agent_weights
        
        # Normalize weights
        self.agent_weights = self.agent_weights / self.agent_weights.sum()
    
    def set_meta_signals(self, signal_prompt: str):
        """Configure meta-signal learning from a prompt describing signal conventions."""
        self.emotion_detector.learn_meta_signals(signal_prompt, self.tokenizer)
    
    def interpolate_agents(self, weights: List[float]) -> None:
        """Manually set agent blending weights."""
        assert len(weights) == self.num_agents
        self.agent_weights = torch.tensor(weights, device=self.device)
        self.agent_weights = self.agent_weights / self.agent_weights.sum()
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        detect_emotion: bool = True,
        return_metadata: bool = False,
        **kwargs
    ) -> str:
        """Generate text with adaptive agent blending."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.forward(
                **inputs,
                use_emotion_routing=detect_emotion,
                return_all_agents=False
            )
            
            # Standard generation with blended logits
            generated = self.base_model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                logits_processor=[self._get_logits_processor(outputs['agent_weights'])],
                **kwargs
            )
        
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        if return_metadata:
            return {
                'text': text,
                'agent_weights': outputs['agent_weights'].cpu().numpy(),
                'emotion_vector': outputs['emotion_vector'].cpu().numpy() if outputs['emotion_vector'] is not None else None
            }
        return text
    
    def _get_logits_processor(self, agent_weights):
        """Create a logits processor that applies agent-specific biases during generation."""
        def processor(input_ids, scores):
            # Apply weighted frequency biases
            for i, (weight, bias) in enumerate(zip(agent_weights, self.frequency_biases)):
                if weight > 0.01:  # Only apply if weight is significant
                    scores = scores + weight * bias.get_bias_vector()
            return scores
        return processor
    
