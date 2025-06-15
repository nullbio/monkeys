import torch
import torch.nn as nn
from typing import Dict, Optional, List


class WeightDelta:
    """
    Utilities for computing and applying weight deltas between models.
    """
    
    @staticmethod
    def compute_delta(
        base_model: nn.Module,
        target_model: nn.Module,
        parameter_filter: Optional[callable] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weight delta between base and target model.
        
        Args:
            base_model: Reference model
            target_model: Model to compute delta from
            parameter_filter: Optional function to filter parameters
            
        Returns:
            Dictionary of weight deltas
        """
        deltas = {}
        
        base_params = dict(base_model.named_parameters())
        target_params = dict(target_model.named_parameters())
        
        for name, target_param in target_params.items():
            if name in base_params:
                if parameter_filter is None or parameter_filter(name):
                    deltas[name] = target_param.data - base_params[name].data
        
        return deltas
    
    @staticmethod
    def apply_delta(
        model: nn.Module,
        deltas: Dict[str, torch.Tensor],
        scale: float = 1.0
    ):
        """
        Apply weight deltas to a model.
        
        Args:
            model: Model to update
            deltas: Weight deltas to apply
            scale: Scaling factor for deltas
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in deltas:
                    param.data += scale * deltas[name]
    
    @staticmethod
    def blend_deltas(
        deltas_list: List[Dict[str, torch.Tensor]],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        Blend multiple weight deltas with given weights.
        
        Args:
            deltas_list: List of delta dictionaries
            weights: Blending weights
            
        Returns:
            Blended deltas
        """
        if len(deltas_list) != len(weights):
            raise ValueError("Number of deltas and weights must match")
        
        # Normalize weights
        weights = torch.tensor(weights)
        weights = weights / weights.sum()
        
        blended = {}
        
        # Get all parameter names
        all_params = set()
        for deltas in deltas_list:
            all_params.update(deltas.keys())
        
        # Blend each parameter
        for param_name in all_params:
            param_sum = None
            
            for deltas, weight in zip(deltas_list, weights):
                if param_name in deltas:
                    if param_sum is None:
                        param_sum = weight * deltas[param_name]
                    else:
                        param_sum += weight * deltas[param_name]
            
            if param_sum is not None:
                blended[param_name] = param_sum
        
        return blended
    
    @staticmethod
    def sparsify_delta(
        deltas: Dict[str, torch.Tensor],
        sparsity: float = 0.9
    ) -> Dict[str, torch.Tensor]:
        """
        Create sparse version of deltas by zeroing small values.
        
        Args:
            deltas: Weight deltas
            sparsity: Fraction of values to zero out
            
        Returns:
            Sparsified deltas
        """
        sparse_deltas = {}
        
        for name, delta in deltas.items():
            # Compute threshold
            flat_delta = delta.flatten()
            k = int(len(flat_delta) * (1 - sparsity))
            threshold = torch.topk(torch.abs(flat_delta), k).values[-1]
            
            # Create mask
            mask = torch.abs(delta) >= threshold
            sparse_deltas[name] = delta * mask
        
        return sparse_deltas
    
    @staticmethod
    def delta_statistics(deltas: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics about weight deltas.
        
        Args:
            deltas: Weight deltas
            
        Returns:
            Statistics dictionary
        """
        stats = {}
        
        for name, delta in deltas.items():
            stats[name] = {
                'mean': delta.mean().item(),
                'std': delta.std().item(),
                'min': delta.min().item(),
                'max': delta.max().item(),
                'norm': torch.norm(delta).item(),
                'sparsity': (delta == 0).float().mean().item(),
                'num_params': delta.numel()
            }
        
        # Global statistics
        total_params = sum(d.numel() for d in deltas.values())
        total_norm = sum(torch.norm(d).item() ** 2 for d in deltas.values()) ** 0.5
        
        stats['global'] = {
            'total_parameters': total_params,
            'total_norm': total_norm,
            'num_layers': len(deltas)
        }
        
        return stats