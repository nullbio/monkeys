#!/usr/bin/env python3
"""
Interactive demo of the adaptive multi-agent LLM system.
"""

import torch
import argparse
import yaml
from pathlib import Path
import logging
from typing import Optional
from colorama import init, Fore, Style

from models import MultiAgentQwen


# Initialize colorama for colored output
init(autoreset=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractiveDemo:
    """Interactive demonstration of the multi-agent system."""
    
    def __init__(self, config_path: str, checkpoint_path: Optional[str] = None):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model
        self.device = torch.device(self.config['model']['device'])
        self.model = MultiAgentQwen(
            model_name=self.config['model']['base_model'],
            num_agents=self.config['model']['num_agents'],
            shared_layers=self.config['model']['shared_layers'],
            device=self.device,
            config=self.config
        )
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
        
        # Define meta-signals
        self.setup_meta_signals()
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def setup_meta_signals(self):
        """Setup default meta-signal conventions."""
        meta_signal_prompt = """
        --- means -3 unhappiness
        -- means -2 mild frustration
        - means -1 slight negativity
        + means +1 slight positivity
        ++ means +2 happiness
        +++ means +3 very happy
        !!! means extremely excited
        ??? means very confused
        ... means contemplative or thinking
        ~~~ means neutral or indifferent
        """
        
        self.model.set_meta_signals(meta_signal_prompt)
        logger.info("Meta-signals configured")
    
    def print_agent_info(self):
        """Print information about the agents."""
        print(f"\n{Fore.CYAN}=== Multi-Agent LLM System ==={Style.RESET_ALL}")
        print(f"Number of agents: {self.model.num_agents}")
        print(f"Base model: {self.config['model']['base_model']}")
        print("\nAgent personalities:")
        print("  Agent 0: Positive/Optimistic bias")
        print("  Agent 1: Negative/Pessimistic bias")
        print("  Agent 2: Analytical/Neutral bias")
        print("  Agent 3: Emphatic/Intense bias")
        print("  Agent 4: Uncertain/Questioning bias")
        print("\nMeta-signals:")
        print("  Use ---, --, -, +, ++, +++ to indicate emotion")
        print("  Use !!!, ???, ..., ~~~ for special states")
        print(f"{Fore.YELLOW}The system will adapt based on your signals!{Style.RESET_ALL}\n")
    
    def visualize_agent_weights(self):
        """Visualize current agent weights."""
        weights = self.model.agent_weights.cpu().numpy()
        print(f"\n{Fore.GREEN}Current Agent Weights:{Style.RESET_ALL}")
        for i, w in enumerate(weights):
            bar_length = int(w * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"  Agent {i}: [{bar}] {w:.3f}")
    
    def interactive_loop(self):
        """Main interactive loop."""
        self.print_agent_info()
        
        print(f"\n{Fore.MAGENTA}Commands:{Style.RESET_ALL}")
        print("  /weights - Show current agent weights")
        print("  /agent N - Force response from agent N")
        print("  /blend w1 w2 w3 w4 w5 - Set custom agent blend")
        print("  /reset - Reset to equal weights")
        print("  /quit - Exit demo")
        print()
        
        while True:
            try:
                # Get user input
                user_input = input(f"{Fore.BLUE}You: {Style.RESET_ALL}")
                
                if user_input.startswith('/'):
                    self.handle_command(user_input)
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    break
                
                # Generate response
                response_data = self.model.generate(
                    user_input,
                    max_length=150,
                    temperature=0.8,
                    detect_emotion=True,
                    return_metadata=True
                )
                
                # Display response
                print(f"{Fore.GREEN}Assistant: {Style.RESET_ALL}{response_data['text']}")
                
                # Show agent weights if they've changed significantly
                if response_data['emotion_vector'] is not None:
                    self.visualize_agent_weights()
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
    
    def handle_command(self, command: str):
        """Handle special commands."""
        parts = command.split()
        cmd = parts[0]
        
        if cmd == '/weights':
            self.visualize_agent_weights()
        
        elif cmd == '/agent' and len(parts) == 2:
            try:
                agent_id = int(parts[1])
                if 0 <= agent_id < self.model.num_agents:
                    weights = [0.0] * self.model.num_agents
                    weights[agent_id] = 1.0
                    self.model.interpolate_agents(weights)
                    print(f"Forced to use Agent {agent_id}")
                else:
                    print(f"Invalid agent ID. Must be 0-{self.model.num_agents-1}")
            except ValueError:
                print("Invalid agent number")
        
        elif cmd == '/blend' and len(parts) == 6:
            try:
                weights = [float(w) for w in parts[1:]]
                self.model.interpolate_agents(weights)
                print("Custom blend applied")
                self.visualize_agent_weights()
            except ValueError:
                print("Invalid weights. Use numbers like: /blend 0.2 0.2 0.2 0.2 0.2")
        
        elif cmd == '/reset':
            weights = [1.0 / self.model.num_agents] * self.model.num_agents
            self.model.interpolate_agents(weights)
            print("Reset to equal weights")
            self.visualize_agent_weights()
        
        elif cmd == '/quit':
            exit(0)
        
        else:
            print(f"Unknown command: {cmd}")


def main():
    parser = argparse.ArgumentParser(description="Interactive demo of multi-agent LLM")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    args = parser.parse_args()
    
    # Create and run demo
    demo = InteractiveDemo(args.config, args.checkpoint)
    demo.interactive_loop()


if __name__ == "__main__":
    main()