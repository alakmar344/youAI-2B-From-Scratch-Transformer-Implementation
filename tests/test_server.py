"""Offline tests for the FastAPI server using Starlette's TestClient.

A tiny model is saved to disk and served; the GPT-2 tokenizer (cached) is the
only external dependency. If FastAPI or the tokenizer is unavailable the module
is skipped.
"""

import pytest
import torch

pytest.importorskip("fastapi")

from youai.config import get_preset_config
from youai.model import YouAIModel


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    try:
        from fastapi.testclient import TestClient
        from youai.server import create_app
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"server deps unavailable: {exc}")

    torch.manual_seed(0)
    ckpt = tmp_path_factory.mktemp("ckpt")
    YouAIModel(get_preset_config("nano")).save_pretrained(str(ckpt))
    try:
        app = create_app(checkpoint=str(ckpt), device="cpu", max_batch=4)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"could not build app (tokenizer?): {exc}")
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["parameters"] > 0


def test_generate(client):
    r = client.post("/generate", json={"prompt": "Hello", "max_new_tokens": 5, "do_sample": False})
    assert r.status_code == 200 and "completion" in r.json()


def test_chat(client):
    r = client.post("/chat", json={"message": "Hi", "max_new_tokens": 5})
    assert r.status_code == 200 and "reply" in r.json()


def test_openai_shim(client):
    r = client.post("/v1/completions", json={"prompt": "AI", "max_tokens": 4, "temperature": 0})
    body = r.json()
    assert body["object"] == "text_completion" and len(body["choices"]) == 1


def test_stream(client):
    with client.stream("POST", "/generate/stream",
                       json={"prompt": "Once", "max_new_tokens": 4, "do_sample": False}) as s:
        lines = [l for l in s.iter_lines() if l.startswith("data:")]
    assert lines and lines[-1].strip() == "data: [DONE]"


def test_batched_generate_offline():
    """generate_batch produces one completion per prompt."""
    from youai.inference import YouAIInference

    inf = YouAIInference.from_model(YouAIModel(get_preset_config("nano")), device="cpu")
    outs = inf.generate_batch(["Hello", "The world is", "A"], max_new_tokens=4, do_sample=False)
    assert len(outs) == 3 and all(isinstance(o, str) for o in outs)
