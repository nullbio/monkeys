import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Tuple
import torch.nn.functional as F


class FrequencyBias(nn.Module):
    """
    Applies learned frequency biases to token distributions for each agent.
    Each agent has a unique bias pattern focusing on emotional/sentiment tokens.
    """
    
    def __init__(
        self,
        vocab_size: int,
        bias_strength: float = 0.15,
        agent_id: int = 0,
        bias_type: str = "emotional",
        temperature_range: Tuple[float, float] = (0.7, 1.3)
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.bias_strength = bias_strength
        self.agent_id = agent_id
        self.bias_type = bias_type
        self.temperature_range = temperature_range
        
        # Don't use fixed vocab size - will adapt to actual logits size
        self.bias_vector = None
        
        # Learned emotional token detector - will be initialized on first use
        self.token_emotion_scores = None
        
        # Agent-specific temperature
        self.temperature = temperature_range[0] + (temperature_range[1] - temperature_range[0]) * (agent_id / 4)
        
    def _initialize_biases(self):
        """Initialize frequency biases based on agent ID and bias type."""
        if self.bias_vector is None:
            return
            
        torch.manual_seed(42 + self.agent_id)  # Reproducible but different per agent
        
        # Initialize with small random values
        self.bias_vector.data = torch.randn_like(self.bias_vector) * 0.01
        
        # Add harmonic variations to create smooth differences between agents
        vocab_size = self.bias_vector.shape[0]
        frequencies = torch.fft.fft(self.bias_vector.data)
        phase_shift = 2 * np.pi * self.agent_id / 5  # 5 agents
        frequencies = frequencies * torch.exp(1j * phase_shift * torch.arange(vocab_size, device=self.bias_vector.device))
        self.bias_vector.data = torch.real(torch.fft.ifft(frequencies)) * self.bias_strength
        
    def _boost_token_set(self, tokens: List[str], strength: float):
        """Boost specific tokens in the bias vector."""
        # Note: In practice, you'd use the actual tokenizer to get token IDs
        # This is a simplified version
        for token in tokens:
            # Hash token to get pseudo-ID (replace with real tokenizer)
            token_id = hash(token) % self.vocab_size
            self.bias_vector.data[token_id] += strength
            
    def forward(self, logits: torch.Tensor, hidden_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply frequency bias to logits.
        
        Args:
            logits: [batch, seq_len, vocab_size]
            hidden_states: Optional hidden states for context-aware biasing
            
        Returns:
            Biased logits
        """
        # Initialize bias vector if needed
        actual_vocab_size = logits.shape[-1]
        if self.bias_vector is None or self.bias_vector.shape[0] != actual_vocab_size:
            self.bias_vector = nn.Parameter(torch.zeros(actual_vocab_size, device=logits.device))
            self._initialize_biases()
            
        # Base frequency bias
        bias = self.bias_vector.unsqueeze(0).unsqueeze(0)  # [1, 1, vocab_size]
        
        # Context-aware adjustment if hidden states provided
        if hidden_states is not None:
            # Use final hidden state to modulate bias strength
            context_vector = hidden_states.mean(dim=1)  # [batch, hidden_dim]
            
            # Learn to adjust bias based on context
            if not hasattr(self, 'context_modulator'):
                self.context_modulator = nn.Linear(context_vector.shape[-1], 1)
            
            # Ensure context_modulator is on the right device
            self.context_modulator = self.context_modulator.to(context_vector.device)
            
            modulation = torch.sigmoid(self.context_modulator(context_vector))  # [batch, 1]
            modulation = modulation.unsqueeze(-1)  # [batch, 1, 1]
            bias = bias.to(modulation.device) * modulation
        
        # Apply bias and temperature
        biased_logits = logits + bias
        biased_logits = biased_logits / self.temperature
        
        return biased_logits
    
    def get_bias_vector(self) -> torch.Tensor:
        """Get the raw bias vector for this agent."""
        if self.bias_vector is None:
            return torch.zeros(1)  # Return dummy tensor if not initialized
        return self.bias_vector
    
    def set_emotional_tokens(self, token_ids: List[int], scores: List[float]):
        """Update emotional scores for specific tokens."""
        if self.token_emotion_scores is not None:
            for token_id, score in zip(token_ids, scores):
                if token_id < self.token_emotion_scores.shape[0]:
                    self.token_emotion_scores.data[token_id] = score
            
    def apply_harmonic_modulation(self, frequency: float, amplitude: float):
        """Apply harmonic modulation to create smooth variations."""
        if self.bias_vector is not None:
            # Create harmonic pattern
            vocab_size = self.bias_vector.shape[0]
            harmonic = torch.sin(2 * np.pi * frequency * torch.arange(vocab_size) / vocab_size)
            self.bias_vector.data += amplitude * harmonic.to(self.bias_vector.device) * self.bias_strength
        
    def get_top_biased_tokens(self, k: int = 10) -> List[int]:
        """Get the top k most positively biased token IDs."""
        if self.bias_vector is None:
            return []
        return torch.topk(self.bias_vector, min(k, self.bias_vector.shape[0])).indices.tolist()
    
    def get_bottom_biased_tokens(self, k: int = 10) -> List[int]:
        """Get the top k most negatively biased token IDs."""
        if self.bias_vector is None:
            return []
        return torch.topk(-self.bias_vector, min(k, self.bias_vector.shape[0])).indices.tolist()