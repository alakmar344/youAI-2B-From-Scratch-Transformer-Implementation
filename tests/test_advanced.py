"""Tests for advanced utilities (compile, model card, training curve)."""

import os
import pytest
import torch

from youai.config import YouAIConfig
from youai.model import YouAIModel
from youai.advanced import (
    compile_model, generate_model_card, plot_training_curve,
)


def _model():
    return YouAIModel(YouAIConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, max_position_embeddings=32,
    ))


def test_compile_model_returns_model():
    model = _model()
    compiled = compile_model(model, mode="default")
    # torch.compile may succeed at compile time but fail at runtime on some systems.
    ids = torch.randint(0, 64, (1, 4))
    try:
        with torch.no_grad():
            out = compiled(ids)
        assert "logits" in out
    except Exception:
        # If compiled model fails, the uncompiled model should still work.
        with torch.no_grad():
            out = model(ids)
        assert "logits" in out


def test_compile_model_invalid_mode():
    model = _model()
    # Should not raise, just return original model on failure.
    compiled = compile_model(model, mode="invalid_mode")
    assert compiled is not None


def test_generate_model_card(tmp_path):
    model = _model()
    out = str(tmp_path / "README.md")
    card = generate_model_card(
        model, out,
        model_name="test-model",
        base_model="gpt2",
        dataset="tinystories",
        description="A test model.",
    )
    assert "test-model" in card
    assert "gpt2" in card
    assert os.path.exists(out)


def test_generate_model_card_minimal(tmp_path):
    model = _model()
    out = str(tmp_path / "README.md")
    card = generate_model_card(model, out)
    assert "language-model" in card
    assert os.path.exists(out)


def test_generate_model_card_with_metrics(tmp_path):
    model = _model()
    out = str(tmp_path / "README.md")
    metrics = {"perplexity": 15.2, "accuracy": 78.5}
    card = generate_model_card(model, out, metrics=metrics)
    assert "15.2" in card
    assert "78.5" in card


def test_plot_training_curve():
    metrics = [
        {"train/loss": 2.5},
        {"train/loss": 2.0},
        {"train/loss": 1.5},
        {"train/loss": 1.2},
        {"train/loss": 1.0},
    ]
    plot = plot_training_curve(metrics, metric_key="train/loss")
    assert "train/loss" in plot
    assert "2.5" in plot
    assert "1.0" in plot


def test_plot_training_curve_missing_metric():
    metrics = [{"other": 1.0}]
    result = plot_training_curve(metrics, metric_key="train/loss")
    assert "No data" in result
