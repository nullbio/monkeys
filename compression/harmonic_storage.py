import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch.fft


class HarmonicStorage:
    """
    Efficient storage of model weight differences using harmonic analysis.
    Compresses weight deltas by storing only significant frequency components.
    """
    
    def __init__(
        self,
        compression_ratio: float = 0.1,
        num_components: int = 1000,
        device: str = "cuda"
    ):
        self.compression_ratio = compression_ratio
        self.num_components = num_components
        self.device = device
        
        # Storage for compressed deltas
        self.compressed_deltas = {}
        self.delta_metadata = {}
        
        # Base model weights (reference point)
        self.base_weights = None
        
    def set_base_weights(self, model: nn.Module):
        """Store base model weights as reference."""
        self.base_weights = {}
        for name, param in model.named_parameters():
            self.base_weights[name] = param.data.clone()
    
    def compress_agent_weights(self, agent_id: int, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Compress an agent's weights by storing only the delta from base.
        
        Args:
            agent_id: ID of the agent
            model: Model with agent-specific weights
            
        Returns:
            Compressed representation
        """
        if self.base_weights is None:
            raise ValueError("Base weights not set. Call set_base_weights first.")
        
        compressed = {}
        metadata = {}
        
        for name, param in model.named_parameters():
            if name in self.base_weights:
                # Compute delta
                delta = param.data - self.base_weights[name]
                
                # Apply harmonic compression
                compressed_delta, meta = self._harmonic_compress(delta)
                compressed[name] = compressed_delta
                metadata[name] = meta
        
        self.compressed_deltas[agent_id] = compressed
        self.delta_metadata[agent_id] = metadata
        
        return compressed
    
    def _harmonic_compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compress tensor using FFT and keep only significant components.
        
        Args:
            tensor: Weight delta tensor
            
        Returns:
            Compressed tensor and metadata
        """
        original_shape = tensor.shape
        tensor_flat = tensor.flatten()
        
        # Apply FFT
        fft_result = torch.fft.fft(tensor_flat)
        
        # Get magnitude and phase
        magnitude = torch.abs(fft_result)
        phase = torch.angle(fft_result)
        
        # Keep only top k components by magnitude
        k = min(self.num_components, int(len(tensor_flat) * self.compression_ratio))
        topk_indices = torch.topk(magnitude, k).indices
        
        # Store compressed representation
        compressed = {
            'indices': topk_indices,
            'magnitudes': magnitude[topk_indices],
            'phases': phase[topk_indices],
            'original_shape': original_shape,
            'original_length': len(tensor_flat)
        }
        
        metadata = {
            'compression_ratio': k / len(tensor_flat),
            'reconstruction_error': self._compute_reconstruction_error(
                tensor_flat, compressed
            )
        }
        
        return compressed, metadata
    
    def decompress_agent_weights(self, agent_id: int) -> Dict[str, torch.Tensor]:
        """
        Reconstruct agent weights from compressed representation.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Reconstructed weight deltas
        """
        if agent_id not in self.compressed_deltas:
            raise ValueError(f"No compressed weights for agent {agent_id}")
        
        decompressed = {}
        
        for name, compressed in self.compressed_deltas[agent_id].items():
            # Reconstruct from harmonic components
            reconstructed = self._harmonic_decompress(compressed)
            decompressed[name] = reconstructed
        
        return decompressed
    
    def _harmonic_decompress(self, compressed: Dict) -> torch.Tensor:
        """Reconstruct tensor from compressed harmonic representation."""
        # Initialize complex array
        fft_reconstructed = torch.zeros(
            compressed['original_length'],
            dtype=torch.complex64,
            device=self.device
        )
        
        # Place components back
        fft_reconstructed[compressed['indices']] = (
            compressed['magnitudes'] * 
            torch.exp(1j * compressed['phases'])
        )
        
        # Inverse FFT
        reconstructed_flat = torch.real(torch.fft.ifft(fft_reconstructed))
        
        # Reshape to original
        reconstructed = reconstructed_flat.reshape(compressed['original_shape'])
        
        return reconstructed
    
    def interpolate_agents(
        self,
        agent_ids: List[int],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        Create interpolated agent by blending compressed representations.
        
        Args:
            agent_ids: List of agent IDs to blend
            weights: Blending weights (should sum to 1)
            
        Returns:
            Interpolated weight deltas
        """
        if len(agent_ids) != len(weights):
            raise ValueError("Number of agents and weights must match")
        
        weights = torch.tensor(weights, device=self.device)
        weights = weights / weights.sum()  # Normalize
        
        interpolated = {}
        
        # Get all parameter names
        param_names = set()
        for agent_id in agent_ids:
            param_names.update(self.compressed_deltas[agent_id].keys())
        
        for name in param_names:
            # Blend in frequency domain
            blended_fft = torch.zeros(
                self.compressed_deltas[agent_ids[0]][name]['original_length'],
                dtype=torch.complex64,
                device=self.device
            )
            
            for agent_id, weight in zip(agent_ids, weights):
                if name in self.compressed_deltas[agent_id]:
                    compressed = self.compressed_deltas[agent_id][name]
                    
                    # Add weighted frequency components
                    blended_fft[compressed['indices']] += weight * (
                        compressed['magnitudes'] * 
                        torch.exp(1j * compressed['phases'])
                    )
            
            # Inverse FFT
            reconstructed_flat = torch.real(torch.fft.ifft(blended_fft))
            original_shape = self.compressed_deltas[agent_ids[0]][name]['original_shape']
            interpolated[name] = reconstructed_flat.reshape(original_shape)
        
        return interpolated
    
    def _compute_reconstruction_error(
        self,
        original: torch.Tensor,
        compressed: Dict
    ) -> float:
        """Compute reconstruction error for compressed representation."""
        reconstructed = torch.real(torch.fft.ifft(
            self._create_fft_from_compressed(compressed, len(original))
        ))
        
        error = torch.norm(original - reconstructed) / torch.norm(original)
        return error.item()
    
    def _create_fft_from_compressed(
        self,
        compressed: Dict,
        length: int
    ) -> torch.Tensor:
        """Create FFT array from compressed representation."""
        fft_array = torch.zeros(length, dtype=torch.complex64, device=self.device)
        fft_array[compressed['indices']] = (
            compressed['magnitudes'] * 
            torch.exp(1j * compressed['phases'])
        )
        return fft_array
    
    def get_storage_efficiency(self) -> Dict[str, float]:
        """Calculate storage efficiency metrics."""
        total_original = 0
        total_compressed = 0
        
        for agent_id, compressed_weights in self.compressed_deltas.items():
            for name, compressed in compressed_weights.items():
                original_size = compressed['original_length']
                compressed_size = len(compressed['indices']) * 3  # indices + magnitude + phase
                
                total_original += original_size
                total_compressed += compressed_size
        
        return {
            'compression_ratio': total_compressed / total_original if total_original > 0 else 0,
            'space_saved': 1 - (total_compressed / total_original) if total_original > 0 else 0,
            'total_agents': len(self.compressed_deltas),
            'parameters_compressed': len(next(iter(self.compressed_deltas.values()))) if self.compressed_deltas else 0
        }