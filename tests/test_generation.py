import torch

from youai.generation import (
    GenerationConfig,
    apply_repetition_penalty,
    sample_next_token,
    top_k_top_p_filter,
)


def test_top_k_filter_keeps_k_tokens():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    filtered = top_k_top_p_filter(logits.clone(), top_k=2)
    assert torch.isfinite(filtered).sum().item() == 2


def test_top_p_is_batch_correct():
    # Two different rows must be filtered independently (the old bug mixed them).
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 10.0]])
    filtered = top_k_top_p_filter(logits.clone(), top_p=0.5)
    assert torch.isfinite(filtered[0, 0]) and not torch.isfinite(filtered[0, 3])
    assert torch.isfinite(filtered[1, 3]) and not torch.isfinite(filtered[1, 0])


def test_repetition_penalty_lowers_seen_tokens():
    logits = torch.tensor([[2.0, 2.0, 2.0]])
    generated = torch.tensor([[0]])
    out = apply_repetition_penalty(logits.clone(), generated, penalty=2.0)
    assert out[0, 0] < out[0, 1]


def test_greedy_is_deterministic():
    logits = torch.randn(3, 50)
    gen = torch.zeros(3, 1, dtype=torch.long)
    cfg = GenerationConfig(do_sample=False)
    a = sample_next_token(logits.clone(), gen, cfg, step=0)
    b = sample_next_token(logits.clone(), gen, cfg, step=0)
    assert torch.equal(a, b)
    assert torch.equal(a.squeeze(1), logits.argmax(-1))


def test_min_new_tokens_blocks_eos():
    logits = torch.full((1, 10), -10.0)
    logits[0, 5] = 100.0  # would pick eos=5
    cfg = GenerationConfig(do_sample=False, eos_token_id=5, min_new_tokens=3)
    tok = sample_next_token(logits.clone(), torch.zeros(1, 1, dtype=torch.long), cfg, step=0)
    assert tok.item() != 5
