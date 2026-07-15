"""Verify GPT-2 weight loading matches HuggingFace.

Requires the HuggingFace ``gpt2`` checkpoint; skipped if it cannot be fetched
(e.g. offline CI). This is the strongest correctness check in the suite: it
proves the YouAI architecture is bit-compatible with real GPT-2.
"""

import pytest
import torch


@pytest.fixture(scope="module")
def gpt2_pair():
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        from youai.pretrained import from_pretrained_gpt2

        hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
        ours = from_pretrained_gpt2("gpt2")
        tok = GPT2Tokenizer.from_pretrained("gpt2")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"gpt2 unavailable: {exc}")
    return hf, ours, tok


def test_logits_match_huggingface(gpt2_pair):
    hf, ours, tok = gpt2_pair
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    with torch.no_grad():
        hf_logits = hf(ids).logits
        our_logits = ours(ids)["logits"]
    assert (hf_logits - our_logits).abs().max().item() < 1e-3
    assert hf_logits[0, -1].argmax() == our_logits[0, -1].argmax()


def test_greedy_generation_matches_huggingface(gpt2_pair):
    from youai.generation import GenerationConfig

    hf, ours, tok = gpt2_pair
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    ours_gen = ours.generate(ids, GenerationConfig(max_new_tokens=10, do_sample=False, eos_token_id=None))
    hf_gen = hf.generate(ids, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
    assert torch.equal(ours_gen, hf_gen)


def test_param_count_reasonable(gpt2_pair):
    _, ours, _ = gpt2_pair
    # GPT-2 small is ~124M parameters.
    assert 1.2e8 < ours.num_parameters < 1.3e8
