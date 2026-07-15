"""Tests for model merging utilities."""

import torch
import pytest

from youai.config import YouAIConfig
from youai.model import YouAIModel
from youai.merging import (
    linear_merge, slerp_merge, dare_merge, ties_merge,
    merge_models, merge_lora_into_base, model_soup,
    list_merge_methods, MERGE_METHODS,
)


def _model():
    return YouAIModel(YouAIConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, max_position_embeddings=32,
    ))


def test_linear_merge_alpha_zero():
    a = _model().state_dict()
    b = _model().state_dict()
    merged = linear_merge(a, b, alpha=0.0)
    for key in a:
        assert torch.equal(merged[key], a[key])


def test_linear_merge_alpha_one():
    a = _model().state_dict()
    b = _model().state_dict()
    merged = linear_merge(a, b, alpha=1.0)
    for key in b:
        assert torch.equal(merged[key], b[key])


def test_linear_merge_half():
    a = _model().state_dict()
    b = _model().state_dict()
    merged = linear_merge(a, b, alpha=0.5)
    for key in a:
        expected = (a[key].float() + b[key].float()) / 2
        assert torch.allclose(merged[key], expected, atol=1e-5)


def test_slerp_merge_parallel():
    a = _model().state_dict()
    b = {k: v * 2 for k, v in a.items()}  # parallel vectors
    merged = slerp_merge(a, b, alpha=0.5)
    for key in a:
        expected = a[key].float() * 1.5
        assert torch.allclose(merged[key], expected, atol=1e-3)


def test_dare_merge_preserves_base():
    a = _model().state_dict()
    b = _model().state_dict()
    # With alpha=0, should return base model.
    merged = dare_merge(a, b, alpha=0.0, density=0.5)
    for key in a:
        assert torch.allclose(merged[key], a[key].float(), atol=1e-5)


def test_dare_merge_density_one():
    a = _model().state_dict()
    b = _model().state_dict()
    merged = dare_merge(a, b, alpha=0.5, density=1.0)
    expected = linear_merge(a, b, alpha=0.5)
    for key in a:
        assert torch.allclose(merged[key], expected[key], atol=1e-3)


def test_ties_merge():
    a = _model().state_dict()
    b = _model().state_dict()
    merged = ties_merge(a, b, alpha=0.5, density=0.5)
    assert len(merged) == len(a)


def test_merge_models_linear():
    a = _model().eval()
    b = _model().eval()
    merged = merge_models(a, b, method="linear", alpha=0.5)
    assert merged is not a
    assert merged is not b


def test_merge_models_slerp():
    a = _model().eval()
    b = _model().eval()
    merged = merge_models(a, b, method="slerp", alpha=0.5)
    assert merged is not a


def test_merge_models_dare():
    a = _model().eval()
    b = _model().eval()
    merged = merge_models(a, b, method="dare", alpha=0.5, density=0.3)
    assert merged is not a


def test_merge_models_ties():
    a = _model().eval()
    b = _model().eval()
    merged = merge_models(a, b, method="ties", alpha=0.5, density=0.3)
    assert merged is not a


def test_merge_models_unknown_raises():
    a = _model()
    b = _model()
    with pytest.raises(ValueError, match="Unknown merge method"):
        merge_models(a, b, method="nonexistent")


def test_merge_models_save(tmp_path):
    a = _model().eval()
    b = _model().eval()
    out = str(tmp_path / "merged")
    merge_models(a, b, method="linear", alpha=0.5, output_path=out)
    import os
    assert os.path.exists(os.path.join(out, "config.json"))


def test_model_soup_uniform():
    models = [_model().eval() for _ in range(3)]
    soup = model_soup(models)
    assert soup is not models[0]


def test_model_soup_weighted():
    models = [_model().eval() for _ in range(2)]
    soup = model_soup(models, weights=[0.7, 0.3])
    assert soup is not models[0]


def test_model_soup_empty_raises():
    with pytest.raises(ValueError):
        model_soup([])


def test_list_merge_methods(capsys):
    list_merge_methods()
    out = capsys.readouterr().out
    assert "linear" in out
    assert "slerp" in out
    assert "dare" in out
    assert "ties" in out


def test_merge_methods_registry():
    assert len(MERGE_METHODS) == 4
    for name, info in MERGE_METHODS.items():
        assert "description" in info
        assert "best_for" in info
        assert "params" in info


def test_merge_lora_into_base():
    from youai.lora import apply_lora
    model = _model()
    apply_lora(model, r=4)
    merged = merge_lora_into_base(model)
    # Should have no LoRA modules after merge.
    from youai.lora import LoRALinear
    for m in merged.modules():
        assert not isinstance(m, LoRALinear)
