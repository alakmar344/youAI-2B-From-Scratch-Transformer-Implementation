"""A production-style FastAPI inference server for YouAI models.

Features:

* ``GET  /health``           — liveness / model info.
* ``POST /generate``         — batched text completion (see the micro-batcher).
* ``POST /chat``             — single-turn chat with history.
* ``POST /generate/stream``  — Server-Sent-Events token streaming.
* An OpenAI-style ``POST /v1/completions`` shim for drop-in tooling.

Concurrent ``/generate`` requests are grouped by a small dynamic **micro-batcher**
so the GPU processes several prompts in one forward pass instead of one-at-a-time,
which is the main throughput win for a real endpoint.

Run it with::

    youai serve --checkpoint ./checkpoints/final --port 8000
    # or:  python -m youai.server --checkpoint ... --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .utils import get_logger

logger = get_logger()


# ----------------------------------------------------------------------
# Request/response schemas (pydantic). Defined at module level so FastAPI can
# resolve them as request bodies (locally-defined models get mis-read as query
# params). The import is guarded so importing this module never hard-requires
# pydantic until the server is actually built.
# ----------------------------------------------------------------------
try:
    from pydantic import BaseModel, Field

    class GenerateRequest(BaseModel):
        prompt: str
        max_new_tokens: int = Field(64, ge=1, le=2048)
        temperature: float = Field(0.8, ge=0.0, le=5.0)
        top_k: int = 50
        top_p: float = 0.9
        repetition_penalty: float = 1.1
        do_sample: bool = True

    class ChatRequest(BaseModel):
        message: str
        history: List[dict] = Field(default_factory=list)
        system_prompt: str = "You are a helpful AI assistant."
        max_new_tokens: int = 150
        temperature: float = 0.7

except ImportError:  # pragma: no cover - server extra not installed
    GenerateRequest = ChatRequest = None


# ----------------------------------------------------------------------
# Dynamic micro-batcher
# ----------------------------------------------------------------------
@dataclass
class _PendingRequest:
    prompt: str
    params: dict
    future: "asyncio.Future"


class MicroBatcher:
    """Collects concurrent generation requests and runs them as one batch.

    Requests that arrive within ``max_wait`` seconds (up to ``max_batch``) are
    generated together. Sampling params are taken from the first request in a
    batch; requests are grouped by params so behaviour stays predictable.
    """

    def __init__(self, inferencer, max_batch: int = 8, max_wait: float = 0.02):
        self.inferencer = inferencer
        self.max_batch = max_batch
        self.max_wait = max_wait
        self.queue: "asyncio.Queue[_PendingRequest]" = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None

    def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def submit(self, prompt: str, params: dict) -> str:
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        await self.queue.put(_PendingRequest(prompt, params, fut))
        return await fut

    async def _run(self):
        while True:
            first = await self.queue.get()
            batch = [first]
            deadline = time.monotonic() + self.max_wait
            # Group requests that share the same sampling params.
            while len(batch) < self.max_batch:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
                if nxt.params == first.params:
                    batch.append(nxt)
                else:
                    # Different params → handle it on its own next loop.
                    await self.queue.put(nxt)
                    break
            await self._process_batch(batch)

    async def _process_batch(self, batch: List[_PendingRequest]):
        prompts = [r.prompt for r in batch]
        params = batch[0].params
        try:
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None, lambda: self.inferencer.generate_batch(prompts, **params)
            )
            for req, text in zip(batch, outputs):
                if not req.future.done():
                    req.future.set_result(text)
        except Exception as exc:  # pragma: no cover - defensive
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(exc)


# ----------------------------------------------------------------------
# App factory
# ----------------------------------------------------------------------
def create_app(checkpoint: Optional[str] = None, device: str = "auto",
               pretrained: Optional[str] = None, max_batch: int = 8):
    """Build the FastAPI app around a loaded model."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install the server extra: pip install 'youai[server]'") from exc

    from .inference import YouAIInference
    from .streaming import StreamingGenerator

    if pretrained:
        from .pretrained import from_pretrained_gpt2

        inferencer = YouAIInference.from_model(from_pretrained_gpt2(pretrained), device=device)
    elif checkpoint:
        inferencer = YouAIInference(checkpoint, device=device)
    else:
        raise ValueError("Provide either checkpoint= or pretrained=.")

    streamer = StreamingGenerator(inferencer.model, inferencer.tokenizer, device=device)

    app = FastAPI(title="YouAI Inference Server", version="1.0.0")
    batcher = MicroBatcher(inferencer, max_batch=max_batch)

    @app.on_event("startup")
    async def _startup():
        batcher.start()

    @app.get("/health")
    async def health():
        cfg = inferencer.model.config
        return {
            "status": "ok",
            "device": str(inferencer.device),
            "parameters": inferencer.model.num_parameters,
            "context_length": cfg.max_position_embeddings,
        }

    @app.post("/generate")
    async def generate(req: GenerateRequest):
        params = req.model_dump(exclude={"prompt"})
        text = await batcher.submit(req.prompt, params)
        return {"prompt": req.prompt, "completion": text}

    @app.post("/chat")
    async def chat(req: ChatRequest):
        loop = asyncio.get_event_loop()
        reply = await loop.run_in_executor(
            None,
            lambda: inferencer.chat(req.message, history=req.history,
                                    system_prompt=req.system_prompt,
                                    max_new_tokens=req.max_new_tokens,
                                    temperature=req.temperature),
        )
        return {"reply": reply}

    @app.post("/generate/stream")
    async def generate_stream(req: GenerateRequest):
        params = req.model_dump(exclude={"prompt"})

        async def event_source():
            for fragment in streamer.stream(req.prompt, **params):
                yield f"data: {json.dumps({'token': fragment})}\n\n"
                await asyncio.sleep(0)
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")

    @app.post("/v1/completions")
    async def openai_completions(body: dict):
        prompt = body.get("prompt", "")
        params = {
            "max_new_tokens": body.get("max_tokens", 64),
            "temperature": body.get("temperature", 0.8),
            "top_p": body.get("top_p", 0.9),
            "top_k": body.get("top_k", 50),
            "repetition_penalty": body.get("repetition_penalty", 1.1),
            "do_sample": body.get("temperature", 0.8) > 0,
        }
        text = await batcher.submit(prompt, params)
        return {
            "object": "text_completion",
            "model": "youai",
            "choices": [{"index": 0, "text": text, "finish_reason": "length"}],
        }

    return app


def serve(checkpoint: Optional[str] = None, host: str = "0.0.0.0", port: int = 8000,
          device: str = "auto", pretrained: Optional[str] = None, max_batch: int = 8):
    """Load a model and run the HTTP server (blocking)."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install the server extra: pip install 'youai[server]'") from exc

    app = create_app(checkpoint=checkpoint, device=device, pretrained=pretrained, max_batch=max_batch)
    logger.info("Serving YouAI on http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port)


def main(argv=None):
    parser = argparse.ArgumentParser(description="YouAI inference server")
    parser.add_argument("--checkpoint", help="Path to a trained checkpoint directory")
    parser.add_argument("--pretrained", help="Serve pretrained GPT-2 weights instead (e.g. gpt2)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batch", type=int, default=8)
    args = parser.parse_args(argv)
    if not args.checkpoint and not args.pretrained:
        parser.error("provide --checkpoint or --pretrained")
    serve(checkpoint=args.checkpoint, host=args.host, port=args.port,
          device=args.device, pretrained=args.pretrained, max_batch=args.max_batch)


if __name__ == "__main__":
    main()
