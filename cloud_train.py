#!/usr/bin/env python3
"""
One-click cloud training script.
Upload this single file and run it on any cloud GPU.
"""

import os
import subprocess
import sys

def setup_environment():
    """Set up training environment from scratch."""
    print("Setting up environment...")
    
    # Install required packages
    packages = [
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "datasets>=2.15.0",
        "sentencepiece",
        "pyyaml",
        "tqdm",
        "numpy",
        "scipy"
    ]
    
    subprocess.run([sys.executable, "-m", "pip", "install"] + packages)
    
    # Clone repo or download files
    if not os.path.exists("models"):
        print("Downloading model files...")
        # In practice, you'd clone your repo here
        os.system("git clone https://github.com/yourusername/adaptive-llm-agents.git .")
    
def main():
    # Setup
    setup_environment()
    
    # Import after installation
    from train_agents_efficient import main as train_main
    
    # Configure for cloud
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Start training
    print("Starting training on A100...")
    sys.argv = ["train_agents_efficient.py", "--num-epochs", "10"]
    train_main()

if __name__ == "__main__":
    main()