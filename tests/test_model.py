import itertools

import torch
import pytest

from youai.config import YouAIConfig
from youai.model import YouAIModel
from youai.generation import GenerationConfig


def _make(**kw):
    base = dict(
        vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, max_position_embeddings=32,
    )
    base.update(kw)
    return YouAIModel(YouAIConfig(**base)).eval()


def test_forward_shapes(tiny_model):
    ids = torch.randint(0, 128, (2, 8))
    out = tiny_model(ids, labels=ids)
    assert out["logits"].shape == (2, 8, 128)
    assert out["loss"].item() > 0


@pytest.mark.parametrize(
    "pos,act,norm",
    list(itertools.product(["rotary", "learned", "alibi"], ["gelu", "swiglu"], ["rmsnorm", "layernorm"])),
)
def test_architecture_variants(pos, act, norm):
    model = _make(position_embedding_type=pos, activation=act, norm_type=norm)
    ids = torch.randint(0, 64, (2, 8))
    out = model(ids, labels=ids)
    assert torch.isfinite(out["loss"])
    gen = model.generate(ids, max_new_tokens=4, do_sample=False, eos_token_id=None)
    assert gen.shape == (2, 12)


def test_grouped_query_attention():
    model = _make(num_attention_heads=4, num_key_value_heads=1)
    ids = torch.randint(0, 64, (1, 6))
    assert torch.isfinite(model(ids, labels=ids)["loss"])


def test_kv_cache_matches_no_cache(tiny_model):
    ids = torch.randint(0, 128, (2, 6))
    a = tiny_model.generate(ids, max_new_tokens=10, do_sample=False, use_cache=True, eos_token_id=None)
    b = tiny_model.generate(ids, max_new_tokens=10, do_sample=False, use_cache=False, eos_token_id=None)
    assert torch.equal(a, b)


def test_flash_and_manual_attention_match():
    ids = torch.randint(0, 64, (2, 8))
    flash = _make(use_flash_attention=True)
    manual = _make(use_flash_attention=False)
    manual.load_state_dict(flash.state_dict())
    with torch.no_grad():
        assert torch.allclose(flash(ids)["logits"], manual(ids)["logits"], atol=1e-4)


def test_weight_tying(tiny_model):
    assert tiny_model.lm_head.weight.data_ptr() == tiny_model.token_embeddings.weight.data_ptr()


def test_context_overflow_raises(tiny_model):
    ids = torch.randint(0, 128, (1, tiny_model.config.max_position_embeddings + 1))
    with pytest.raises(ValueError):
        tiny_model(ids)


def test_generation_respects_eos():
    model = _make()
    ids = torch.randint(0, 64, (1, 4))
    cfg = GenerationConfig(max_new_tokens=5, do_sample=False, eos_token_id=None)
    out = model.generate(ids, cfg)
    assert out.shape[1] == 9


def test_save_and_load(tiny_model, tmp_path):
    ids = torch.randint(0, 128, (1, 6))
    tiny_model.save_pretrained(str(tmp_path))
    reloaded = YouAIModel.from_pretrained(str(tmp_path)).eval()
    with torch.no_grad():
        assert torch.allclose(tiny_model(ids)["logits"], reloaded(ids)["logits"], atol=1e-5)


def test_gradient_checkpointing_runs():
    model = _make(gradient_checkpointing=True)
    model.train()
    ids = torch.randint(0, 64, (2, 8))
    loss = model(ids, labels=ids)["loss"]
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
