import torch

from youai.config import YouAIConfig
from youai.model import YouAIModel
from youai.lora import (
    LoRALinear, apply_lora, merge_lora, lora_state_dict, save_lora, load_lora,
    trainable_parameter_summary,
)


def _model():
    return YouAIModel(YouAIConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, max_position_embeddings=32,
    ))


def test_lora_linear_starts_as_noop():
    base = torch.nn.Linear(16, 24)
    lora = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(2, 16)
    # B is zero-initialised, so the adapter output equals the base output.
    assert torch.allclose(lora(x), base(x), atol=1e-6)


def test_apply_lora_freezes_base_and_adds_adapters():
    model = _model()
    apply_lora(model, r=4, target_modules=("q_proj", "v_proj"))
    summary = trainable_parameter_summary(model)
    assert 0 < summary["trainable"] < summary["total"]
    # Only lora_ parameters should require grad.
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert "lora_" in name


def test_lora_state_dict_is_small():
    model = _model()
    apply_lora(model, r=4)
    sd = lora_state_dict(model)
    assert sd and all("lora_" in k for k in sd)


def test_lora_trains_only_adapters():
    torch.manual_seed(0)
    model = _model()
    apply_lora(model, r=4)
    base_before = model.layers[0].attention.q_proj.base.weight.clone()
    adapter_before = model.layers[0].attention.q_proj.lora_B.weight.clone()

    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1.0)
    ids = torch.randint(0, 64, (2, 8))
    for _ in range(3):
        opt.zero_grad()
        model(ids, labels=ids)["loss"].backward()
        opt.step()

    q = model.layers[0].attention.q_proj
    assert torch.equal(q.base.weight, base_before)          # base frozen
    assert not torch.equal(q.lora_B.weight, adapter_before)  # adapter learned


def test_merge_lora_preserves_output():
    torch.manual_seed(0)
    model = _model().eval()
    apply_lora(model, r=4)
    # Give the adapter a non-zero effect.
    for m in model.modules():
        if isinstance(m, LoRALinear):
            torch.nn.init.normal_(m.lora_B.weight, std=0.02)
    ids = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        before = model(ids)["logits"].clone()
    merge_lora(model)
    with torch.no_grad():
        after = model(ids)["logits"]
    assert torch.allclose(before, after, atol=1e-5)


def test_save_and_load_adapter(tmp_path):
    # Adapters must sit on the *same* base weights, so build both models from an
    # identical seed before applying LoRA.
    torch.manual_seed(0)
    model = _model()
    torch.manual_seed(0)
    fresh = _model()

    apply_lora(model, r=4)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            torch.nn.init.normal_(m.lora_A.weight, std=0.02)
            torch.nn.init.normal_(m.lora_B.weight, std=0.02)
    ids = torch.randint(0, 64, (1, 6))
    model.eval()
    with torch.no_grad():
        expected = model(ids)["logits"].clone()

    save_lora(model, str(tmp_path))

    apply_lora(fresh, r=4)
    load_lora(fresh, str(tmp_path))
    fresh.eval()
    with torch.no_grad():
        assert torch.allclose(fresh(ids)["logits"], expected, atol=1e-5)
