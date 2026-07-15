"""Integration tests for inference / streaming.

These require the GPT-2 tokenizer, which is fetched once and cached. If the
tokenizer cannot be loaded (e.g. no network in CI), the tests are skipped.
"""

import pytest
import torch

from youai.config import get_preset_config
from youai.model import YouAIModel


@pytest.fixture(scope="module")
def tokenizer():
    try:
        from youai.tokenizer import get_tokenizer

        return get_tokenizer()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"tokenizer unavailable: {exc}")


@pytest.fixture(scope="module")
def small_model():
    torch.manual_seed(0)
    return YouAIModel(get_preset_config("nano")).eval()


def test_inference_generate(small_model, tokenizer):
    from youai.inference import YouAIInference

    inf = YouAIInference.from_model(small_model, device="cpu")
    out = inf.generate("Hello world", max_new_tokens=8, do_sample=False)
    assert isinstance(out, list) and isinstance(out[0], str)


def test_inference_skip_prompt(small_model):
    from youai.inference import YouAIInference

    inf = YouAIInference.from_model(small_model, device="cpu")
    full = inf.generate("Hello world", max_new_tokens=6, do_sample=False, skip_prompt=False)[0]
    only_new = inf.generate("Hello world", max_new_tokens=6, do_sample=False, skip_prompt=True)[0]
    assert len(full) >= len(only_new)


def test_streaming(small_model, tokenizer):
    from youai.streaming import StreamingGenerator

    gen = StreamingGenerator(small_model, tokenizer, device="cpu")
    fragments = list(gen.stream("Once upon a time", max_new_tokens=6, do_sample=False))
    assert "".join(fragments) == gen.generate_stream(
        "Once upon a time", max_new_tokens=6, do_sample=False
    )


def test_chat_session(small_model, tokenizer):
    from youai.streaming import ChatSession

    session = ChatSession(small_model, tokenizer, device="cpu")
    session.chat("Hi there", max_new_tokens=6)
    assert len(session.get_history()) == 1
    session.clear_history()
    assert session.get_history() == []


def test_perplexity(small_model, tokenizer):
    from youai.inference import YouAIInference

    inf = YouAIInference.from_model(small_model, device="cpu")
    assert inf.perplexity("The quick brown fox") > 0
