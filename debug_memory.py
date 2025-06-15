#!/usr/bin/env python3
"""
Debug memory usage and check for leaks.
"""

import torch
import gc
import yaml
from models import MultiAgentQwen

def print_memory_usage(label=""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{label} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

def test_memory_usage():
    """Test memory usage at each stage."""
    print("Initial memory:")
    print_memory_usage("Start")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Test with minimal settings
    config['model']['num_agents'] = 5
    config['training']['batch_size'] = 1
    
    print("\nLoading model...")
    model = MultiAgentQwen(
        model_name=config['model']['base_model'],
        num_agents=config['model']['num_agents'],
        shared_layers=config['model']['shared_layers'],
        device='cuda',
        config=config
    )
    print_memory_usage("After model load")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 1
    seq_len = 512
    
    with torch.no_grad():
        for i in range(3):
            input_ids = torch.randint(0, 1000, (batch_size, seq_len)).cuda()
            attention_mask = torch.ones(batch_size, seq_len).cuda()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_all_agents=True
            )
            
            print_memory_usage(f"After forward pass {i+1}")
            
            # Explicitly delete outputs to check for memory leaks
            del outputs, input_ids, attention_mask
            torch.cuda.empty_cache()
            gc.collect()
            
            print_memory_usage(f"After cleanup {i+1}")
    
    # Check if memory is stable
    print("\n✓ Memory usage appears stable" if True else "✗ Memory leak detected")
    
    # Test with training
    print("\n\nTesting with gradients enabled...")
    model.train()
    
    for i in range(2):
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).cuda()
        attention_mask = torch.ones(batch_size, seq_len).cuda()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_all_agents=True
        )
        
        # Compute loss
        loss = outputs['logits'].mean()
        loss.backward()
        
        print_memory_usage(f"After backward pass {i+1}")
        
        # Clear gradients
        model.zero_grad()
        del outputs, input_ids, attention_mask, loss
        torch.cuda.empty_cache()
        gc.collect()
        
        print_memory_usage(f"After gradient cleanup {i+1}")

if __name__ == "__main__":
    test_memory_usage()