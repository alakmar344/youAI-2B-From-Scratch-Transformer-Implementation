"""Token-by-token streaming generation for interactive apps."""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Callable, Generator, List, Optional

import torch
import torch.nn as nn

from .generation import GenerationConfig, normalize_stop_sequences, sample_next_token
from .tokenizer import get_tokenizer
from .utils import resolve_device


class StreamingGenerator:
    """Stream generated tokens one at a time, using a KV cache for speed.

    Usage::

        gen = StreamingGenerator(model, tokenizer)
        for token in gen.stream("Once upon a time"):
            print(token, end="", flush=True)
    """

    def __init__(self, model: nn.Module, tokenizer=None, device: str = "auto"):
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer or get_tokenizer()

    def _prepare(self, prompt: str, max_new_tokens, temperature, top_k, top_p,
                 repetition_penalty, do_sample):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        cfg = GenerationConfig(
            max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
            top_p=top_p, repetition_penalty=repetition_penalty, do_sample=do_sample,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return input_ids, cfg

    @torch.no_grad()
    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        stop_sequences: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        """Yield decoded text fragments as they are generated."""
        input_ids, cfg = self._prepare(
            prompt, max_new_tokens, temperature, top_k, top_p, repetition_penalty, do_sample
        )
        stops = normalize_stop_sequences(stop_sequences)
        generated = input_ids
        past = None
        cur = input_ids
        prev_text = ""
        new_tokens: List[int] = []
        max_ctx = self.model.config.max_position_embeddings

        for step in range(cfg.max_new_tokens):
            model_input = cur if past is not None else generated[:, -max_ctx:]
            out = self.model(model_input, past_key_values=past, use_cache=True)
            past = out["past_key_values"]
            next_token = sample_next_token(out["logits"][:, -1, :], generated, cfg, step)
            tid = int(next_token.item())
            if tid == cfg.eos_token_id:
                break

            new_tokens.append(tid)
            generated = torch.cat([generated, next_token], dim=1)
            cur = next_token

            # Decode incrementally so multi-token characters render correctly.
            full = self.tokenizer.decode(new_tokens)
            fragment = full[len(prev_text):]
            prev_text = full

            if stops and any(s in full for s in stops):
                # Emit up to the stop sequence, then finish.
                cut = min(full.find(s) for s in stops if s in full)
                remaining = full[len(prev_text) - len(fragment):cut]
                if remaining:
                    yield remaining
                break

            if fragment:
                yield fragment

    def generate_stream(self, prompt: str, callback: Optional[Callable[[str], None]] = None,
                        **kwargs) -> str:
        """Stream generation while accumulating and returning the full text."""
        full = ""
        for fragment in self.stream(prompt, **kwargs):
            full += fragment
            if callback:
                callback(fragment)
        return full

    async def async_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Asynchronous streaming for web servers / async frameworks."""
        for fragment in self.stream(prompt, **kwargs):
            yield fragment
            await asyncio.sleep(0)


class ChatSession:
    """Stateful chat session with streaming support and history management."""

    def __init__(self, model: nn.Module, tokenizer=None,
                 system_prompt: str = "You are a helpful AI assistant.",
                 max_history: int = 10, device: str = "auto"):
        self.generator = StreamingGenerator(model, tokenizer, device)
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.history: List[dict] = []

    def _build_prompt(self, user_message: str) -> str:
        prompt = f"{self.system_prompt}\n\n" if self.system_prompt else ""
        for turn in self.history[-self.max_history:]:
            prompt += f"Human: {turn['user']}\nAssistant: {turn['assistant']}\n\n"
        prompt += f"Human: {user_message}\nAssistant:"
        return prompt

    def chat(self, message: str, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
        prompt = self._build_prompt(message)
        response = self.generator.generate_stream(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature,
            stop_sequences=["Human:", "\nHuman"],
        ).strip()
        self.history.append({"user": message, "assistant": response})
        return response

    def chat_stream(self, message: str, max_new_tokens: int = 200,
                    temperature: float = 0.7) -> Generator[str, None, None]:
        prompt = self._build_prompt(message)
        full = ""
        for fragment in self.generator.stream(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature,
            stop_sequences=["Human:", "\nHuman"],
        ):
            full += fragment
            yield fragment
        self.history.append({"user": message, "assistant": full.strip()})

    def clear_history(self) -> None:
        self.history = []

    def get_history(self) -> List[dict]:
        return list(self.history)

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt
