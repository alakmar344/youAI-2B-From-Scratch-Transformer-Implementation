"""Tests for alignment training (DPO, ORPO, SimPO)."""

import json
import pytest
import torch

from youai.config import YouAIConfig
from youai.model import YouAIModel
from youai.alignment import (
    DPOConfig, DPOTrainer, PreferenceDataset,
    dpo_loss, orpo_loss, simpo_loss,
    _compute_logprobs, create_preference_data,
)


def _model():
    return YouAIModel(YouAIConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, max_position_embeddings=32,
    ))


def _create_test_data(tmp_path, num=10):
    data = []
    for i in range(num):
        data.append({
            "prompt": f"Question {i}?",
            "chosen": f"This is the correct answer to question {i}.",
            "rejected": f"Wrong answer {i}.",
        })
    path = tmp_path / "prefs.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return str(path)


def test_dpo_loss_basic():
    chosen = torch.randn(4)
    rejected = torch.randn(4)
    ref_chosen = torch.randn(4)
    ref_rejected = torch.randn(4)

    loss, metrics = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.1)
    assert loss.item() > 0
    assert "accuracy" in metrics
    assert "reward_margin" in metrics


def test_orpo_loss_basic():
    # Log-probs must be negative (valid log-probabilities).
    chosen = torch.randn(4) - 3  # shift to be negative
    rejected = torch.randn(4) - 3
    nll = torch.tensor(2.0)

    loss, metrics = orpo_loss(chosen, rejected, nll, beta=0.1)
    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert "nll" in metrics


def test_simpo_loss_basic():
    # Log-probs must be negative.
    chosen = torch.randn(4) - 3
    rejected = torch.randn(4) - 3

    loss, metrics = simpo_loss(chosen, rejected, beta=0.1, gamma=1.0)
    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert "accuracy" in metrics


def test_compute_logprobs():
    model = _model().eval()
    ids = torch.randint(0, 64, (2, 10))
    mask = torch.ones(2, 10)
    resp_mask = torch.zeros(2, 10)
    resp_mask[:, 5:] = 1  # response starts at position 5

    logps = _compute_logprobs(model, ids, mask, resp_mask)
    assert logps.shape == (2,)
    assert torch.isfinite(logps).all()


def test_preference_dataset_creation(tmp_path):
    path = _create_test_data(tmp_path, num=5)

    # Create a mock tokenizer.
    class MockTokenizer:
        eos_token_id = 0
        def encode(self, text, **kwargs):
            return list(range(len(text.split())))
        def __call__(self, text, **kwargs):
            ids = list(range(min(len(text.split()), kwargs.get("max_length", 10))))
            pad_len = kwargs.get("max_length", 10)
            ids = ids + [0] * (pad_len - len(ids))
            mask = [1] * min(len(text.split()), pad_len) + [0] * max(0, pad_len - len(text.split()))
            return {
                "input_ids": torch.tensor([ids[:pad_len]]),
                "attention_mask": torch.tensor([mask[:pad_len]]),
            }

    ds = PreferenceDataset(path, tokenizer=MockTokenizer(), max_length=10)
    assert len(ds) == 5
    item = ds[0]
    assert "chosen_input_ids" in item
    assert "rejected_input_ids" in item
    assert "chosen_response_mask" in item


def test_create_preference_data(tmp_path):
    path = str(tmp_path / "test_prefs.json")
    create_preference_data(path, num_examples=20)
    with open(path) as f:
        data = json.load(f)
    assert len(data) == 20
    assert "prompt" in data[0]
    assert "chosen" in data[0]
    assert "rejected" in data[0]


def test_dpo_config_defaults():
    config = DPOConfig()
    assert config.algorithm == "dpo"
    assert config.beta == 0.1
    assert config.learning_rate == 5e-7


def test_dpo_config_to_dict():
    config = DPOConfig(algorithm="orpo", beta=0.2)
    d = config.to_dict()
    assert d["algorithm"] == "orpo"
    assert d["beta"] == 0.2
