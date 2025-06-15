#!/usr/bin/env python3
"""
Quick test script to debug training with minimal data.
"""

import torch
import yaml
from models import MultiAgentQwen

def test_single_batch():
    """Test a single training iteration with minimal data."""
    print("Loading config...")
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Override for minimal testing
    config['model']['device'] = 'cuda'
    config['training']['batch_size'] = 2  # Very small batch
    
    print("Initializing model...")
    model = MultiAgentQwen(
        model_name=config['model']['base_model'],
        num_agents=config['model']['num_agents'],
        shared_layers=config['model']['shared_layers'],
        device='cuda',
        config=config
    )
    
    # Create dummy batch
    batch_size = 2
    seq_len = 128  # Short sequence
    
    print("Creating dummy batch...")
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).cuda()
    attention_mask = torch.ones(batch_size, seq_len).cuda()
    
    print("Running forward pass...")
    try:
        with torch.cuda.amp.autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_all_agents=True
            )
        
        print("✓ Forward pass successful!")
        print(f"Output keys: {outputs.keys()}")
        
        # Test loss computation
        all_agent_logits = outputs['all_agent_logits']
        print(f"Agent logits shape: {all_agent_logits.shape}")
        
        # Test actual training loss computation
        print("\nTesting loss computation...")
        shift_logits = all_agent_logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # Test the exact loss computation used in training
        agent_losses = []
        for i in range(model.num_agents):
            agent_logits = shift_logits[:, i]
            loss = torch.nn.functional.cross_entropy(
                agent_logits.reshape(-1, agent_logits.size(-1)),
                shift_labels.reshape(-1)
            )
            agent_losses.append(loss)
        
        total_loss = torch.stack(agent_losses).mean()
        print(f"✓ Loss computation successful! Loss: {total_loss.item():.4f}")
        
        # Test backward pass
        print("\nTesting backward pass...")
        total_loss.backward()
        print("✓ Backward pass successful!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_single_batch()
    if success:
        print("\n✅ All tests passed! Ready for full training.")
    else:
        print("\n❌ Tests failed. Fix errors before full training.")