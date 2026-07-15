"""Shared pytest fixtures.

The tests deliberately use a tiny model and a small vocabulary so the whole
suite runs in seconds on CPU with no network access.
"""

import torch
import pytest

from youai.config import YouAIConfig
from youai.model import YouAIModel


@pytest.fixture(autouse=True)
def _threads():
    torch.set_num_threads(2)
    torch.manual_seed(0)


@pytest.fixture
def tiny_config():
    return YouAIConfig(
        vocab_size=128, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=128, max_position_embeddings=64,
        activation="swiglu", norm_type="rmsnorm", position_embedding_type="rotary",
    )


@pytest.fixture
def tiny_model(tiny_config):
    return YouAIModel(tiny_config).eval()
