"""Tests for evaluation tools."""

import pytest
import torch

from youai.config import YouAIConfig
from youai.model import YouAIModel
from youai.evaluation import (
    evaluate_perplexity, evaluate_generation,
    compute_bleu, compute_rouge_l,
    evaluate_multiple_choice, run_benchmark,
    _distinct_n, _repetition_rate, _lexical_diversity,
)


def _model():
    return YouAIModel(YouAIConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, max_position_embeddings=32,
    ))


class MockBatchEncoding(dict):
    """Mimics HuggingFace BatchEncoding with attribute access."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


class MockTokenizer:
    eos_token_id = 0
    def encode(self, text, **kwargs):
        ids = list(range(min(len(text.split()), 20)))
        if kwargs.get("return_tensors") == "pt":
            return torch.tensor([ids])
        return ids
    def decode(self, ids, **kwargs):
        return " ".join(f"tok{i}" for i in ids)
    def __call__(self, text, return_tensors=None, **kwargs):
        ids = self.encode(text)
        return MockBatchEncoding({
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids)),
        })


def test_distinct_n():
    tokens = ["a", "b", "c", "a", "b", "c"]
    assert _distinct_n(tokens, 1) == 0.5  # 3 unique out of 6
    assert _distinct_n(tokens, 2) > 0


def test_distinct_n_empty():
    assert _distinct_n([], 1) == 0.0


def test_repetition_rate():
    tokens = ["a", "b", "a", "b", "a", "b"]
    rate = _repetition_rate(tokens, n=2)
    assert rate > 0  # "a b" repeats


def test_repetition_rate_no_repeat():
    tokens = ["a", "b", "c", "d", "e", "f"]
    rate = _repetition_rate(tokens, n=2)
    assert rate == 0.0


def test_lexical_diversity():
    assert _lexical_diversity(["a", "b", "c"]) == 1.0
    assert _lexical_diversity(["a", "a", "a"]) == 1 / 3
    assert _lexical_diversity([]) == 0.0


def test_compute_bleu_perfect():
    refs = ["the cat sat on the mat"]
    hyps = ["the cat sat on the mat"]
    result = compute_bleu(refs, hyps)
    assert result["bleu"] > 95  # near-perfect


def test_compute_bleu_bad():
    refs = ["the cat sat on the mat"]
    hyps = ["dog dog dog dog"]
    result = compute_bleu(refs, hyps)
    assert result["bleu"] < 20


def test_compute_bleu_multiple():
    refs = ["hello world", "foo bar baz"]
    hyps = ["hello world", "foo bar baz"]
    result = compute_bleu(refs, hyps)
    # BLEU with perfect matches should be high.
    assert result["precision_1"] > 0.9
    assert result["bleu"] > 0


def test_compute_rouge_l_perfect():
    refs = ["the cat sat on the mat"]
    hyps = ["the cat sat on the mat"]
    result = compute_rouge_l(refs, hyps)
    assert result["rouge_l_f1"] > 95


def test_compute_rouge_l_partial():
    refs = ["the cat sat on the mat"]
    hyps = ["the dog sat"]
    result = compute_rouge_l(refs, hyps)
    assert 0 < result["rouge_l_f1"] < 100


def test_compute_rouge_l_empty():
    refs = [""]
    hyps = [""]
    result = compute_rouge_l(refs, hyps)
    assert result["rouge_l_f1"] == 0.0


def test_evaluate_generation():
    model = _model().eval()
    tok = MockTokenizer()
    results = evaluate_generation(
        model, tok, prompts=["Hello world", "Test prompt"],
        max_new_tokens=5, temperature=1.0, device="cpu",
    )
    assert "avg_length_tokens" in results
    assert "distinct_1" in results
    assert "repetition_rate" in results
    assert results["num_generations"] == 2


def test_evaluate_multiple_choice():
    model = _model().eval()
    tok = MockTokenizer()
    questions = [
        {"question": "What is 2+2?", "choices": ["3", "4", "5", "6"], "answer": 1},
        {"question": "Capital of France?", "choices": ["London", "Paris", "Berlin", "Rome"], "answer": 1},
    ]
    # This will use the mock tokenizer which returns short sequences.
    results = evaluate_multiple_choice(model, tok, questions, device="cpu")
    assert "accuracy" in results
    assert results["num_questions"] == 2


def test_run_benchmark():
    model = _model().eval()
    tok = MockTokenizer()
    results = run_benchmark(
        model, tok,
        prompts=["Hello", "World"],
        device="cpu",
    )
    assert "generation" in results
    assert "model_info" in results
    assert results["model_info"]["parameters"] > 0


def test_perplexity_on_file(tmp_path):
    model = _model().eval()
    tok = MockTokenizer()
    corpus = tmp_path / "test.txt"
    corpus.write_text("hello world " * 50)
    results = evaluate_perplexity(model, tok, str(corpus), max_length=16, stride=8, device="cpu")
    assert "perplexity" in results
    assert results["perplexity"] > 0
