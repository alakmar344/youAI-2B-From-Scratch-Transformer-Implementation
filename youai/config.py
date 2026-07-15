"""YouAI Model Configuration"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class YouAIConfig:
    """Configuration for YouAI model.
    
    Args:
        vocab_size: Size of vocabulary (default: 50257 for GPT-2 tokenizer)
        max_position_embeddings: Maximum sequence length (default: 2048)
        hidden_size: Embedding dimension (default: 2048)
        num_hidden_layers: Number of transformer layers (default: 24)
        num_attention_heads: Number of attention heads (default: 16)
        intermediate_size: FFN hidden dimension (default: 8192)
        hidden_dropout_prob: Dropout probability for hidden layers (default: 0.1)
        attention_dropout_prob: Dropout probability for attention (default: 0.1)
        layer_norm_eps: Layer normalization epsilon (default: 1e-5)
    """
    vocab_size: int = 50257
    max_position_embeddings: int = 2048
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 8192
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-5

    @property
    def total_params(self) -> int:
        """Calculate approximate parameter count."""
        token_emb = self.vocab_size * self.hidden_size
        pos_emb = self.max_position_embeddings * self.hidden_size
        per_layer = (
            4 * self.hidden_size * self.hidden_size +
            2 * self.hidden_size * self.intermediate_size +
            4 * self.hidden_size
        )
        all_layers = per_layer * self.num_hidden_layers
        output_head = self.hidden_size * self.vocab_size
        return token_emb + pos_emb + all_layers + output_head

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'max_position_embeddings': self.max_position_embeddings,
            'hidden_size': self.hidden_size,
            'num_hidden_layers': self.num_hidden_layers,
            'num_attention_heads': self.num_attention_heads,
            'intermediate_size': self.intermediate_size,
            'hidden_dropout_prob': self.hidden_dropout_prob,
            'attention_dropout_prob': self.attention_dropout_prob,
            'layer_norm_eps': self.layer_norm_eps,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'YouAIConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


def get_preset_config(preset: str) -> YouAIConfig:
    """Get a preset model configuration.
    
    Args:
        preset: One of '125m', '350m', '750m', '2b'
        
    Returns:
        YouAIConfig for the specified preset
        
    Raises:
        ValueError: If preset is not recognized
    """
    presets = {
        '125m': YouAIConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
        ),
        '350m': YouAIConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
        ),
        '750m': YouAIConfig(
            hidden_size=1536,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=6144,
        ),
        '2b': YouAIConfig(
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=8192,
        ),
    }
    
    preset = preset.lower()
    if preset not in presets:
        raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(presets.keys())}")
    return presets[preset]
