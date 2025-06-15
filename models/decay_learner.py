import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class DecayLearner(nn.Module):
    """
    Learns context-aware decay patterns for each agent.
    Non-linear decay with potential for baseline shifts.
    """
    
    def __init__(
        self,
        num_agents: int,
        context_dim: int,
        hidden_dim: int = 256,
        num_decay_components: int = 3
    ):
        super().__init__()
        self.num_agents = num_agents
        self.num_decay_components = num_decay_components
        
        # Per-agent decay networks
        self.agent_decay_nets = nn.ModuleList([
            self._create_decay_network(context_dim, hidden_dim)
            for _ in range(num_agents)
        ])
        
        # Global context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Baseline shift detector
        self.baseline_shift_detector = nn.Sequential(
            nn.Linear(hidden_dim // 2 + num_agents, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Smooth interpolation network for cross-agent decay
        self.interpolation_net = nn.Sequential(
            nn.Linear(num_agents * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_agents * num_decay_components)
        )
        
        # Decay history for learning patterns
        self.register_buffer('decay_history', torch.zeros(100, num_agents))
        self.register_buffer('context_history', torch.zeros(100, hidden_dim // 2))
        self.history_pointer = 0
        
    def _create_decay_network(self, context_dim: int, hidden_dim: int) -> nn.Module:
        """Create a decay network for a single agent."""
        return nn.Sequential(
            nn.Linear(context_dim + self.num_agents, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.num_decay_components + 2)  # +2 for min/max bounds
        )
    
    def forward(
        self,
        current_weights: torch.Tensor,
        context: torch.Tensor,
        emotion_trajectory: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute decay parameters based on context and current state.
        
        Args:
            current_weights: Current agent blending weights [num_agents]
            context: Context representation [batch, context_dim]
            emotion_trajectory: Optional emotion history [trajectory_len, emotion_dim]
            
        Returns:
            Dictionary containing:
            - decay_rate: Effective decay rate for each agent
            - shift_baseline: Whether to shift baseline (strong signal detected)
            - decay_components: Individual decay components
        """
        batch_size = context.shape[0]
        device = context.device
        
        # Ensure current_weights is on the same device
        current_weights = current_weights.to(device)
        
        # Encode context
        encoded_context = self.context_encoder(context.mean(dim=0, keepdim=True))  # [1, hidden//2]
        
        # Prepare input: context + current weights
        decay_input = torch.cat([
            encoded_context.expand(batch_size, -1),
            current_weights.unsqueeze(0).expand(batch_size, -1)
        ], dim=-1)
        
        # Get per-agent decay parameters
        agent_decays = []
        for i, decay_net in enumerate(self.agent_decay_nets):
            agent_input = torch.cat([context, current_weights.unsqueeze(0).expand(batch_size, -1)], dim=-1)
            decay_params = decay_net(agent_input)
            agent_decays.append(decay_params)
        
        agent_decays = torch.stack(agent_decays, dim=1)  # [batch, agents, components+2]
        
        # Extract components
        decay_components = agent_decays[..., :self.num_decay_components]
        min_decay = torch.sigmoid(agent_decays[..., -2]) * 0.1  # Min decay in [0, 0.1]
        max_decay = torch.sigmoid(agent_decays[..., -1]) * 0.5 + 0.5  # Max decay in [0.5, 1.0]
        
        # Compute effective decay rate using learned components
        # Components represent: fast decay, slow decay, oscillating decay
        t = torch.arange(batch_size, device=context.device).float() / batch_size
        
        fast_decay = torch.exp(-decay_components[..., 0] * t.unsqueeze(-1).unsqueeze(-1))
        slow_decay = torch.exp(-decay_components[..., 1] * t.unsqueeze(-1).unsqueeze(-1) * 0.1)
        oscillating = torch.sin(decay_components[..., 2] * t.unsqueeze(-1).unsqueeze(-1) * 2 * np.pi) * 0.1
        
        # Combine decay components
        combined_decay = (fast_decay + slow_decay + oscillating) / 3
        combined_decay = torch.clamp(combined_decay, min_decay, max_decay)
        
        # Smooth interpolation between agents
        interpolation_input = torch.cat([current_weights, current_weights.roll(1)], dim=-1)
        interpolation_weights = self.interpolation_net(interpolation_input)
        interpolation_weights = interpolation_weights.view(self.num_agents, self.num_decay_components)
        interpolation_weights = F.softmax(interpolation_weights, dim=0).unsqueeze(0).unsqueeze(0)
        
        # Apply interpolation for smooth gradient
        # combined_decay shape: [batch, 1, agents]
        # interpolation_weights shape: [1, 1, agents, components]
        # We need to average across agents
        final_decay = combined_decay.mean(dim=2).mean(dim=0)  # Simple average for now
        
        # Detect baseline shift
        shift_input = torch.cat([encoded_context.squeeze(0), current_weights], dim=-1)
        shift_probability = self.baseline_shift_detector(shift_input)
        
        # Update history
        # Get per-agent decay rates from the last timestep
        if combined_decay.dim() == 3:  # [batch, timesteps, agents]
            agent_decay_rates = combined_decay[-1, -1, :]  # Last batch, last timestep
        else:
            agent_decay_rates = combined_decay.squeeze()
        
        # Ensure it's 1D
        if agent_decay_rates.dim() > 1:
            agent_decay_rates = agent_decay_rates.flatten()[:self.num_agents]
        
        self._update_history(agent_decay_rates, encoded_context.squeeze(0))
        
        return {
            'decay_rate': final_decay,
            'shift_baseline': shift_probability > 0.5,
            'shift_probability': shift_probability,
            'decay_components': {
                'fast': fast_decay.mean(dim=0).squeeze(),
                'slow': slow_decay.mean(dim=0).squeeze(),
                'oscillating': oscillating.mean(dim=0).squeeze()
            },
            'min_decay': min_decay.mean(dim=0).squeeze(),
            'max_decay': max_decay.mean(dim=0).squeeze()
        }
    
    def _update_history(self, decay_rates: torch.Tensor, context: torch.Tensor):
        """Update decay and context history."""
        # Ensure decay_rates has the correct shape [num_agents]
        if decay_rates.dim() > 1:
            decay_rates = decay_rates.squeeze()
        if decay_rates.shape[0] != self.num_agents:
            # If it's a scalar, expand to all agents
            if decay_rates.numel() == 1:
                decay_rates = decay_rates.expand(self.num_agents)
            else:
                decay_rates = decay_rates[:self.num_agents]
        
        self.decay_history[self.history_pointer] = decay_rates.detach()
        self.context_history[self.history_pointer] = context.detach()
        self.history_pointer = (self.history_pointer + 1) % self.decay_history.shape[0]
    
    def learn_from_feedback(self, feedback: float, current_state: Dict[str, torch.Tensor]):
        """
        Reinforcement learning update based on user feedback.
        Positive feedback reinforces current decay patterns.
        """
        # This would be implemented with a proper RL algorithm
        # For now, it's a placeholder for the RL component
        pass
    
    def get_decay_trajectory(self, agent_id: int, steps: int = 50) -> torch.Tensor:
        """Visualize decay trajectory for a specific agent."""
        t = torch.linspace(0, 1, steps)
        trajectory = torch.zeros(steps)
        
        # Get recent decay parameters for this agent
        recent_decays = self.decay_history[-10:, agent_id]
        avg_decay = recent_decays.mean()
        
        # Simple exponential decay for visualization
        trajectory = torch.exp(-avg_decay * t)
        
        return trajectory