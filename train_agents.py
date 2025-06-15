#!/usr/bin/env python3
"""
Train multiple parallel agents with different frequency biases.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
import wandb
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset

from models import MultiAgentQwen
from compression import HarmonicStorage
from utils.training_utils import set_seed, create_optimizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentTrainer:
    """Trainer for multi-agent system with shared forward pass."""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config['model']['device'])
        
        # Initialize model
        self.model = MultiAgentQwen(
            model_name=config['model']['base_model'],
            num_agents=config['model']['num_agents'],
            shared_layers=config['model']['shared_layers'],
            device=self.device,
            config=config
        )
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['model']['base_model'],
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize harmonic storage
        self.storage = HarmonicStorage(
            compression_ratio=config['compression']['compression_ratio'],
            num_components=config['compression']['frequency_components'],
            device=self.device
        )
        
        # Setup training
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler() if config['training']['mixed_precision'] else None
        
        # Enable gradient checkpointing if specified
        if config.get('model', {}).get('gradient_checkpointing', False):
            self.model.base_model.gradient_checkpointing_enable()
        
    def prepare_data(self):
        """Load and prepare training data."""
        # Load a conversational dataset
        dataset = load_dataset("allenai/soda", split="train[:10000]")  # Start small
        
        def tokenize_function(examples):
            # Concatenate dialogue turns
            dialogues = []
            for dialogue in examples['dialogue']:
                # Join all turns with special tokens
                text = " [SEP] ".join(dialogue)
                dialogues.append(text)
            
            return self.tokenizer(
                dialogues,
                truncation=True,
                max_length=256,  # Reduced to save memory during multi-agent training
                padding="max_length"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Set format to PyTorch tensors
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=0  # Set to 0 to avoid tokenizer parallelism warning
        )
        
        return dataloader
    
    def train_epoch(self, dataloader, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass through all agents
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_all_agents=True
                )
                
                # Compute loss for each agent
                all_agent_logits = outputs['all_agent_logits']  # [batch, agents, seq, vocab]
                
                # Shift for language modeling
                shift_logits = all_agent_logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                # Compute per-agent losses
                agent_losses = []
                for i in range(self.model.num_agents):
                    agent_logits = shift_logits[:, i]  # [batch, seq-1, vocab]
                    loss = nn.functional.cross_entropy(
                        agent_logits.reshape(-1, agent_logits.size(-1)),
                        shift_labels.reshape(-1),
                        ignore_index=self.tokenizer.pad_token_id
                    )
                    agent_losses.append(loss)
                
                # Combined loss with diversity encouragement
                base_loss = torch.stack(agent_losses).mean()
                
                # Diversity loss: encourage different agents to have different outputs
                diversity_loss = self._compute_diversity_loss(all_agent_logits)
                
                total_step_loss = base_loss + 0.1 * diversity_loss
            
            # Scale loss by gradient accumulation steps
            total_step_loss = total_step_loss / self.config['training']['gradient_accumulation_steps']
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(total_step_loss).backward()
            else:
                total_step_loss.backward()
            
            # Update weights only after gradient accumulation
            if (step + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Update progress
            total_loss += total_step_loss.item()
            progress_bar.set_postfix({
                'loss': total_step_loss.item(),
                'base_loss': base_loss.item(),
                'div_loss': diversity_loss.item()
            })
            
            # Log to wandb
            if step % 10 == 0 and wandb.run is not None:
                wandb.log({
                    'train/loss': total_step_loss.item(),
                    'train/base_loss': base_loss.item(),
                    'train/diversity_loss': diversity_loss.item(),
                    'train/learning_rate': self.scheduler.get_last_lr()[0]
                })
        
        return total_loss / len(dataloader)
    
    def _compute_diversity_loss(self, all_agent_logits: torch.Tensor) -> torch.Tensor:
        """Compute diversity loss to encourage different agent behaviors."""
        # Use KL divergence between agent probability distributions
        batch_size, num_agents, seq_len, vocab_size = all_agent_logits.shape
        
        # Convert to probabilities
        probs = torch.softmax(all_agent_logits, dim=-1)
        
        # Compute pairwise KL divergences
        kl_sum = 0
        count = 0
        
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                # KL(P_i || P_j)
                kl = torch.nn.functional.kl_div(
                    torch.log(probs[:, i] + 1e-8),
                    probs[:, j],
                    reduction='batchmean'
                )
                kl_sum += kl
                count += 1
        
        # We want to maximize KL divergence (minimize negative KL)
        diversity_loss = -kl_sum / count if count > 0 else torch.tensor(0.0)
        
        return diversity_loss
    
    def train(self, num_epochs: int):
        """Main training loop."""
        # Prepare data
        dataloader = self.prepare_data()
        
        # Setup optimizer and scheduler
        self.optimizer = create_optimizer(
            self.model,
            lr=float(self.config['training']['learning_rate']),
            weight_decay=0.01
        )
        
        total_steps = len(dataloader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=total_steps
        )
        
        # Training loop
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(dataloader, epoch)
            logger.info(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config['logging']['save_interval'] == 0:
                self.save_checkpoint(epoch)
            
            # Compress and store agent weights
            if (epoch + 1) % 5 == 0:
                self._compress_agents()
    
    def _compress_agents(self):
        """Compress agent weights using harmonic storage."""
        logger.info("Compressing agent weights...")
        
        # Set base weights (first time only)
        if self.storage.base_weights is None:
            self.storage.set_base_weights(self.model.base_model)
        
        # Compress each agent's specific weights
        for i in range(self.model.num_agents):
            # Create temporary model with agent-specific weights
            agent_model = self._extract_agent_weights(i)
            self.storage.compress_agent_weights(i, agent_model)
        
        # Log compression efficiency
        efficiency = self.storage.get_storage_efficiency()
        logger.info(f"Compression efficiency: {efficiency}")
        
        if wandb.run is not None:
            wandb.log({
                'compression/ratio': efficiency['compression_ratio'],
                'compression/space_saved': efficiency['space_saved']
            })
    
    def _extract_agent_weights(self, agent_id: int):
        """Extract weights specific to an agent."""
        # This is a simplified version - in practice, you'd extract
        # only the agent-specific layers
        return self.model
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config['logging']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        torch.save(
            checkpoint,
            checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        )
        logger.info(f"Saved checkpoint for epoch {epoch}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--wandb-project', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize wandb
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            config=config
        )
    
    # Create trainer and train
    trainer = AgentTrainer(config)
    trainer.train(args.num_epochs)
    
    # Final compression
    trainer._compress_agents()
    
    # Close wandb
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()