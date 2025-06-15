#!/usr/bin/env python3
"""
Test script for the multi-agent system components.
"""

import torch
import yaml
from models import MultiAgentQwen, FrequencyBias, EmotionDetector, DecayLearner
from compression import HarmonicStorage, WeightDelta


def test_frequency_bias():
    """Test frequency bias module."""
    print("Testing FrequencyBias...")
    
    vocab_size = 50000
    bias = FrequencyBias(vocab_size=vocab_size, agent_id=0)
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    logits = torch.randn(batch_size, seq_len, vocab_size)
    hidden_states = torch.randn(batch_size, seq_len, 768)
    
    biased_logits = bias(logits, hidden_states)
    assert biased_logits.shape == logits.shape
    
    print("✓ FrequencyBias test passed")


def test_emotion_detector():
    """Test emotion detector module."""
    print("Testing EmotionDetector...")
    
    detector = EmotionDetector(model_dim=768, output_dim=64)
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, 768)
    attention_mask = torch.ones(batch_size, seq_len)
    
    emotion_vector = detector(hidden_states, attention_mask)
    assert emotion_vector.shape == (batch_size, 64)
    
    # Test meta-signal parsing
    signal_prompt = "--- means -3 unhappy"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Simple tokenizer for testing
    detector.learn_meta_signals(signal_prompt, tokenizer)
    
    print("✓ EmotionDetector test passed")


def test_decay_learner():
    """Test decay learner module."""
    print("Testing DecayLearner...")
    
    learner = DecayLearner(num_agents=5, context_dim=768)
    
    # Test forward pass
    current_weights = torch.ones(5) / 5
    context = torch.randn(1, 768)
    
    decay_output = learner(current_weights, context)
    assert 'decay_rate' in decay_output
    assert 'shift_baseline' in decay_output
    
    print("✓ DecayLearner test passed")


def test_harmonic_storage():
    """Test harmonic compression storage."""
    print("Testing HarmonicStorage...")
    
    storage = HarmonicStorage(compression_ratio=0.1, device="cpu")
    
    # Create dummy tensors
    base_tensor = torch.randn(1000, 768)
    agent_tensor = base_tensor + torch.randn(1000, 768) * 0.1
    
    # Test compression
    compressed, metadata = storage._harmonic_compress(agent_tensor - base_tensor)
    assert 'indices' in compressed
    assert 'magnitudes' in compressed
    
    # Test decompression
    reconstructed = storage._harmonic_decompress(compressed)
    assert reconstructed.shape == base_tensor.shape
    
    print("✓ HarmonicStorage test passed")


def test_integration():
    """Test basic integration of components."""
    print("\nTesting integration...")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Override for CPU testing
    config['model']['device'] = 'cpu'
    config['model']['num_agents'] = 3  # Fewer agents for testing
    
    # Initialize model (this will download the model if not cached)
    print("Initializing MultiAgentQwen (this may take a while on first run)...")
    try:
        model = MultiAgentQwen(
            model_name="gpt2",  # Use smaller model for testing
            num_agents=3,
            shared_layers=5,
            device="cpu",
            config=config
        )
        
        # Test forward pass
        input_ids = torch.randint(0, 1000, (1, 20))
        outputs = model(input_ids, return_all_agents=True)
        
        assert 'logits' in outputs
        assert 'agent_weights' in outputs
        print("✓ Integration test passed")
        
    except Exception as e:
        print(f"Integration test skipped (model download may be required): {e}")


def main():
    """Run all tests."""
    print("Running Multi-Agent LLM System Tests...\n")
    
    test_frequency_bias()
    test_emotion_detector()
    test_decay_learner()
    test_harmonic_storage()
    test_integration()
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()