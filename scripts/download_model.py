#!/usr/bin/env python3
"""
Download the Qwen model for the multi-agent system.
"""

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def download_model(model_name: str = "Qwen/Qwen2.5-3B"):
    """Download and cache the model."""
    print(f"Downloading {model_name}...")
    
    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Download model
    print("Downloading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    print(f"Model downloaded successfully!")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-3B",
                        help='Model to download')
    args = parser.parse_args()
    
    download_model(args.model)


if __name__ == "__main__":
    main()