import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import re
from transformers import AutoTokenizer


class EmotionDetector(nn.Module):
    """
    Learns to detect emotion signals from text, including meta-signals.
    Can adapt to arbitrary signaling conventions defined in prompts.
    """
    
    def __init__(
        self,
        model_dim: int,
        output_dim: int = 64,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.model_dim = model_dim
        self.output_dim = output_dim
        
        # Multi-head attention for emotion detection
        self.emotion_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Emotion vector projection
        self.emotion_projection = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, output_dim),
            nn.Tanh()  # Emotion vectors in [-1, 1]
        )
        
        # Meta-signal pattern storage
        self.meta_patterns = {}
        self.pattern_embeddings = nn.ParameterDict()
        
        # Learned signal detector
        self.signal_detector = nn.LSTM(
            input_size=model_dim,
            hidden_size=model_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Pattern matching network
        self.pattern_scorer = nn.Linear(model_dim, 1)
        
        # Emotion history for context
        self.register_buffer('emotion_history', torch.zeros(100, output_dim))
        self.history_pointer = 0
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_scores: bool = False
    ) -> torch.Tensor:
        """
        Detect emotion from hidden states.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: Optional attention mask
            return_scores: Whether to return pattern matching scores
            
        Returns:
            emotion_vector: [batch, emotion_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Apply signal detection LSTM (ensure float32 for LSTM)
        original_dtype = hidden_states.dtype
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.float()
        
        signal_features, _ = self.signal_detector(hidden_states)
        
        # Convert back to original dtype
        if original_dtype == torch.float16:
            signal_features = signal_features.half()
        
        # Self-attention to find emotional regions
        attended_features, attention_weights = self.emotion_attention(
            signal_features, signal_features, signal_features,
            key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )
        
        # Check for meta-signal patterns
        pattern_scores = self._detect_patterns(hidden_states, attention_mask)
        
        # Combine attended features with pattern scores
        if pattern_scores is not None:
            # Weight features by pattern detection
            pattern_weight = torch.sigmoid(pattern_scores).unsqueeze(-1)
            combined_features = attended_features * pattern_weight + signal_features * (1 - pattern_weight)
        else:
            combined_features = attended_features
        
        # Global pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(combined_features)
            pooled = (combined_features * mask_expanded).sum(1) / mask_expanded.sum(1)
        else:
            pooled = combined_features.mean(dim=1)
        
        # Project to emotion vector
        emotion_vector = self.emotion_projection(pooled)
        
        # Update emotion history
        self._update_history(emotion_vector.detach().mean(dim=0))
        
        if return_scores:
            return emotion_vector, pattern_scores
        return emotion_vector
    
    def learn_meta_signals(self, signal_prompt: str, tokenizer: AutoTokenizer):
        """
        Learn emotion signaling patterns from a prompt.
        
        Example prompt:
        "--- means -3 happiness
         +++ means +3 happiness
         !!! means extremely excited
         ... means contemplative"
        """
        # Parse signal patterns from prompt
        patterns = self._parse_signal_patterns(signal_prompt)
        
        # Tokenize patterns and create embeddings
        for pattern_text, (emotion_type, intensity) in patterns.items():
            # Create unique key for this pattern
            pattern_key = f"{pattern_text}_{emotion_type}"
            
            # Initialize learnable embedding for this pattern
            if pattern_key not in self.pattern_embeddings:
                self.pattern_embeddings[pattern_key] = nn.Parameter(
                    torch.randn(self.output_dim) * 0.1
                )
            
            # Store pattern info
            self.meta_patterns[pattern_text] = {
                'tokens': tokenizer.encode(pattern_text, add_special_tokens=False),
                'emotion': emotion_type,
                'intensity': intensity,
                'embedding_key': pattern_key
            }
    
    def _parse_signal_patterns(self, prompt: str) -> Dict[str, Tuple[str, float]]:
        """Parse emotion patterns from a signal definition prompt."""
        patterns = {}
        
        # Common patterns to look for
        # Pattern: "XXX means/indicates Y emotion/happiness/etc"
        pattern_regex = r'["\']?([^"\']+)["\']?\s+(?:means|indicates|represents?)\s+([+-]?\d+\.?\d*)\s*(\w+)'
        
        lines = prompt.strip().split('\n')
        for line in lines:
            match = re.search(pattern_regex, line, re.IGNORECASE)
            if match:
                signal = match.group(1).strip()
                intensity = float(match.group(2))
                emotion = match.group(3).strip().lower()
                patterns[signal] = (emotion, intensity)
        
        # Also look for simpler patterns like "---: -3" or "+++ = +3 happy"
        simple_pattern = r'([^\s:=]+)\s*[:=]\s*([+-]?\d+\.?\d*)\s*(\w*)'
        for line in lines:
            if not any(word in line.lower() for word in ['means', 'indicates', 'represents']):
                match = re.search(simple_pattern, line)
                if match:
                    signal = match.group(1).strip()
                    intensity = float(match.group(2))
                    emotion = match.group(3).strip().lower() if match.group(3) else 'valence'
                    patterns[signal] = (emotion, intensity)
        
        return patterns
    
    def _detect_patterns(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """Detect learned meta-signal patterns in the input."""
        if not self.meta_patterns:
            return None
        
        batch_size, seq_len, _ = hidden_states.shape
        pattern_scores = torch.zeros(batch_size, seq_len, device=hidden_states.device)
        
        # Score each position for pattern matches
        for pattern_text, pattern_info in self.meta_patterns.items():
            if pattern_info['embedding_key'] in self.pattern_embeddings:
                pattern_emb = self.pattern_embeddings[pattern_info['embedding_key']]
                
                # Compute similarity between hidden states and pattern embedding
                similarity = F.cosine_similarity(
                    hidden_states,
                    pattern_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1),
                    dim=-1
                )
                
                # Weight by intensity
                pattern_scores += similarity * abs(pattern_info['intensity'])
        
        # Apply pattern scorer network
        pattern_scores = self.pattern_scorer(hidden_states).squeeze(-1) + pattern_scores
        
        if attention_mask is not None:
            pattern_scores = pattern_scores.masked_fill(~attention_mask.bool(), -1e9)
        
        return pattern_scores
    
    def _update_history(self, emotion_vector: torch.Tensor):
        """Update emotion history buffer."""
        self.emotion_history[self.history_pointer] = emotion_vector
        self.history_pointer = (self.history_pointer + 1) % self.emotion_history.shape[0]
    
    def get_emotion_trajectory(self, window: int = 20) -> torch.Tensor:
        """Get recent emotion trajectory."""
        if window > self.emotion_history.shape[0]:
            window = self.emotion_history.shape[0]
        
        # Get last 'window' emotions in order
        if self.history_pointer >= window:
            trajectory = self.emotion_history[self.history_pointer - window:self.history_pointer]
        else:
            # Wrap around
            trajectory = torch.cat([
                self.emotion_history[-(window - self.history_pointer):],
                self.emotion_history[:self.history_pointer]
            ])
        
        return trajectory
    
    def reset_history(self):
        """Reset emotion history."""
        self.emotion_history.zero_()
        self.history_pointer = 0