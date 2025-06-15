import torch
import torch.nn as nn
import numpy as np
import random
from typing import Optional


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    optimizer_type: str = "adamw"
) -> torch.optim.Optimizer:
    """Create optimizer with proper parameter groups."""
    # Separate parameters that should and shouldn't have weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if any(nd in name for nd in ["bias", "norm", "embedding"]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]
    
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=lr)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer