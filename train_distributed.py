#!/usr/bin/env python3
"""
Distributed training across multiple GPUs/nodes.
Each agent trains on its own GPU.
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler

from models import MultiAgentQwen
from utils.training_utils import set_seed, create_optimizer


class DistributedAgentTrainer:
    """Train agents across multiple GPUs."""
    
    def __init__(self, rank: int, world_size: int, config: dict):
        self.rank = rank  # GPU/node ID
        self.world_size = world_size  # Total GPUs
        self.config = config
        self.agent_id = rank  # Each GPU trains one agent
        
        # Set device for this process
        self.device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(self.device)
        
        print(f"GPU {rank}: Training Agent {self.agent_id}")
        
        # Initialize model
        self.model = MultiAgentQwen(
            model_name=config['model']['base_model'],
            num_agents=config['model']['num_agents'],
            shared_layers=config['model']['shared_layers'],
            device=self.device,
            config=config
        )
        
        # Set this GPU to focus on its agent
        weights = [0.0] * self.model.num_agents
        weights[self.agent_id] = 1.0
        self.model.interpolate_agents(weights)
        
        # Wrap model in DDP
        self.model = DDP(self.model, device_ids=[rank])
        
    def prepare_data(self):
        """Prepare distributed data loading."""
        dataset = load_dataset("allenai/soda", split="train[:50000]")
        
        def tokenize_function(examples):
            dialogues = []
            for dialogue in examples['dialogue']:
                text = " [SEP] ".join(dialogue)
                dialogues.append(text)
            
            return self.model.module.tokenizer(
                dialogues,
                truncation=True,
                max_length=512,
                padding="max_length"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        
        # Distributed sampler ensures each GPU gets different data
        sampler = DistributedSampler(
            tokenized_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=16,  # Can use larger batches
            sampler=sampler,
            num_workers=4
        )
        
        return dataloader, sampler
    
    def train(self, num_epochs: int):
        """Train this agent on this GPU."""
        dataloader, sampler = self.prepare_data()
        
        # Only train agent-specific parameters
        agent_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and (
                f"frequency_biases.{self.agent_id}" in name or
                (self.agent_id == 0 and ("emotion" in name or "decay" in name))
            ):
                agent_params.append(param)
        
        optimizer = create_optimizer(
            agent_params,
            lr=self.config['training']['learning_rate']
        )
        
        for epoch in range(num_epochs):
            sampler.set_epoch(epoch)  # Shuffle differently each epoch
            
            self.model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_emotion_routing=True,
                    return_all_agents=False
                )
                
                logits = outputs['logits']
                
                # Compute loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=self.model.module.tokenizer.pad_token_id
                )
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(agent_params, 1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    print(f"Agent {self.agent_id}, Epoch {epoch}, "
                          f"Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(dataloader)
            print(f"Agent {self.agent_id}, Epoch {epoch} complete. "
                  f"Avg Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if self.rank == 0:  # Only main process saves
                self.save_checkpoint(epoch)
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pt")


def setup_distributed(rank: int, world_size: int, config: dict):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Train
    trainer = DistributedAgentTrainer(rank, world_size, config)
    trainer.train(num_epochs=10)
    
    # Cleanup
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--num-gpus', type=int, default=torch.cuda.device_count())
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    world_size = min(args.num_gpus, config['model']['num_agents'])
    
    if world_size > 1:
        # Spawn processes for multi-GPU
        torch.multiprocessing.spawn(
            setup_distributed,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU fallback
        setup_distributed(0, 1, config)


if __name__ == "__main__":
    main()