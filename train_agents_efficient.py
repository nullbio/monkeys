#!/usr/bin/env python3
"""
Efficient training script that trains agents sequentially rather than all at once.
Much faster and more memory efficient.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset

from models import MultiAgentQwen
from compression import HarmonicStorage
from utils.training_utils import set_seed, create_optimizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EfficientAgentTrainer:
    """More efficient trainer that trains one agent at a time."""
    
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
        self.tokenizer = self.model.tokenizer
        
        # Initialize harmonic storage
        self.storage = HarmonicStorage(
            compression_ratio=config['compression']['compression_ratio'],
            num_components=config['compression']['frequency_components'],
            device=self.device
        )
        
        # Training setup
        self.current_agent = 0
        self.agent_optimizers = []
        self.agent_schedulers = []
        
    def prepare_data(self):
        """Load and prepare training data."""
        dataset = load_dataset("allenai/soda", split="train[:10000]")
        
        def tokenize_function(examples):
            dialogues = []
            for dialogue in examples['dialogue']:
                text = " [SEP] ".join(dialogue)
                dialogues.append(text)
            
            return self.tokenizer(
                dialogues,
                truncation=True,
                max_length=512,  # Can use full length when training one agent
                padding="max_length"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=8,  # Larger batch size possible
            shuffle=True,
            num_workers=4
        )
        
        return dataloader
    
    def train_agent(self, agent_id: int, dataloader, num_steps: int):
        """Train a single agent efficiently."""
        logger.info(f"Training Agent {agent_id}")
        
        # Set agent weights to focus on this agent
        weights = [0.0] * self.model.num_agents
        weights[agent_id] = 1.0
        self.model.interpolate_agents(weights)
        
        # Create optimizer for this agent's parameters
        agent_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and (f"frequency_biases.{agent_id}" in name or 
                                       "emotion" in name or "decay" in name):
                agent_params.append(param)
        
        optimizer = create_optimizer(
            agent_params,
            lr=self.config['training']['learning_rate']
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=num_steps
        )
        
        # Training loop for this agent
        self.model.train()
        progress_bar = tqdm(enumerate(dataloader), total=num_steps, 
                           desc=f"Agent {agent_id}")
        
        for step, batch in progress_bar:
            if step >= num_steps:
                break
                
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass - single agent
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_emotion_routing=True,
                    return_all_agents=False  # Only blend, don't compute all
                )
                
                logits = outputs['logits']
                
                # Standard language modeling loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                loss = nn.functional.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=self.tokenizer.pad_token_id
                )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(agent_params, 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Clear cache periodically
            if step % 100 == 0:
                torch.cuda.empty_cache()
        
        return optimizer.state_dict(), scheduler.state_dict()
    
    def train(self, num_epochs: int):
        """Train all agents efficiently."""
        dataloader = self.prepare_data()
        steps_per_agent = len(dataloader) // self.model.num_agents
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train each agent sequentially
            for agent_id in range(self.model.num_agents):
                opt_state, sched_state = self.train_agent(
                    agent_id, dataloader, steps_per_agent
                )
                
                # Save agent checkpoint
                self.save_agent_checkpoint(agent_id, epoch, opt_state, sched_state)
            
            # Compress all agents after epoch
            self._compress_agents()
            
            # Optional: Run a blended evaluation
            self.evaluate_blended(dataloader)
    
    def evaluate_blended(self, dataloader):
        """Evaluate with all agents blended."""
        self.model.eval()
        self.model.interpolate_agents([0.2] * 5)  # Equal blend
        
        total_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= 50:  # Quick evaluation
                    break
                    
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_emotion_routing=True
                )
                
                logits = outputs['logits']
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                loss = nn.functional.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1)
                )
                
                total_loss += loss.item()
        
        avg_loss = total_loss / 50
        logger.info(f"Blended evaluation loss: {avg_loss:.4f}")
    
    def _compress_agents(self):
        """Compress agent weights."""
        if self.storage.base_weights is None:
            self.storage.set_base_weights(self.model.base_model)
        
        for i in range(self.model.num_agents):
            self.storage.compress_agent_weights(i, self.model)
    
    def save_agent_checkpoint(self, agent_id: int, epoch: int, 
                             opt_state: dict, sched_state: dict):
        """Save individual agent checkpoint."""
        checkpoint_dir = Path(self.config['logging']['checkpoint_dir']) / f"agent_{agent_id}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'agent_id': agent_id,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': opt_state,
            'scheduler_state_dict': sched_state,
            'config': self.config
        }
        
        torch.save(
            checkpoint,
            checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(args.seed)
    
    # Create trainer and train
    trainer = EfficientAgentTrainer(config)
    trainer.train(args.num_epochs)


if __name__ == "__main__":
    main()