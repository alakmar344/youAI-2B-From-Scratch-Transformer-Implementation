"""Inference: load a trained model and generate / chat with it."""

from __future__ import annotations

import os
from typing import List, Optional

import torch

from .model import YouAIModel
from .generation import GenerationConfig, sample_next_token
from .tokenizer import get_tokenizer
from .utils import resolve_device, get_logger

logger = get_logger()


class YouAIInference:
    """Load a checkpoint and run text generation or chat.

    Args:
        checkpoint_path: Directory containing ``config.json`` and
            ``pytorch_model.bin`` (as produced by the trainer).
        device: Device string (``"auto"`` by default).
        tokenizer_name: Tokenizer to use for encode/decode.
    """

    def __init__(self, checkpoint_path: str, device: str = "auto",
                 tokenizer_name: str = "gpt2"):
        self.device = resolve_device(device)
        self.model = YouAIModel.from_pretrained(checkpoint_path, map_location=self.device)
        self.model.to(self.device).eval()
        self.tokenizer = get_tokenizer(tokenizer_name)
        logger.info("Loaded model (%s params) on %s",
                    f"{self.model.num_parameters/1e6:.1f}M", self.device)

    # ------------------------------------------------------------------
    @classmethod
    def from_model(cls, model: YouAIModel, device: str = "auto",
                   tokenizer_name: str = "gpt2") -> "YouAIInference":
        """Wrap an in-memory model (skips loading from disk)."""
        self = cls.__new__(cls)
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = get_tokenizer(tokenizer_name)
        return self

    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        skip_prompt: bool = False,
        **kwargs,
    ) -> List[str]:
        """Generate one or more completions for ``prompt``.

        Args:
            skip_prompt: If ``True``, return only the newly generated text.
            Additional keyword arguments are forwarded to ``GenerationConfig``
            (e.g. ``max_length`` as an alias for ``max_new_tokens``).
        """
        if "max_length" in kwargs and max_new_tokens == 100:
            max_new_tokens = kwargs.pop("max_length")
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        cfg = GenerationConfig(
            max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
            top_p=top_p, repetition_penalty=repetition_penalty, do_sample=do_sample,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        prompt_len = input_ids.size(1)
        results = []
        for _ in range(num_return_sequences):
            output = self.model.generate(input_ids, cfg)
            tokens = output[0, prompt_len:] if skip_prompt else output[0]
            results.append(self.tokenizer.decode(tokens, skip_special_tokens=True))
        return results

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        skip_prompt: bool = True,
    ) -> List[str]:
        """Generate completions for several prompts in a single batched pass.

        Prompts are left-padded so a shared attention mask keeps padding out of
        the computation. This is what the server's micro-batcher uses to raise
        throughput under concurrent load.
        """
        if not prompts:
            return []
        tok = self.tokenizer
        old_side = tok.padding_side
        tok.padding_side = "left"
        try:
            enc = tok(prompts, return_tensors="pt", padding=True)
        finally:
            tok.padding_side = old_side

        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        cfg = GenerationConfig(
            max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
            top_p=top_p, repetition_penalty=repetition_penalty, do_sample=do_sample,
            eos_token_id=tok.eos_token_id,
        )
        max_ctx = self.model.config.max_position_embeddings
        prompt_len = input_ids.size(1)
        generated = input_ids
        finished = torch.zeros(len(prompts), dtype=torch.bool, device=self.device)

        for step in range(max_new_tokens):
            out = self.model(generated[:, -max_ctx:], attention_mask=attn[:, -max_ctx:])
            next_token = sample_next_token(out["logits"][:, -1, :], generated, cfg, step)
            next_token = torch.where(
                finished.unsqueeze(1), torch.full_like(next_token, cfg.eos_token_id), next_token
            )
            generated = torch.cat([generated, next_token], dim=1)
            attn = torch.cat([attn, torch.ones_like(next_token)], dim=1)
            finished = finished | (next_token.squeeze(1) == cfg.eos_token_id)
            if bool(finished.all()) or generated.size(1) >= max_ctx:
                break

        results = []
        for row in generated:
            tokens = row[prompt_len:] if skip_prompt else row
            results.append(self.tokenizer.decode(tokens, skip_special_tokens=True))
        return results

    # ------------------------------------------------------------------
    def chat(
        self,
        message: str,
        history: Optional[List[dict]] = None,
        system_prompt: str = "You are a helpful AI assistant.",
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate a single chat reply, using an optional conversation history."""
        prompt = f"{system_prompt}\n\n" if system_prompt else ""
        for msg in (history or [])[-8:]:
            role = "Human" if msg.get("role") == "user" else "Assistant"
            prompt += f"{role}: {msg.get('content', '')}\n"
        prompt += f"Human: {message}\nAssistant:"

        completion = self.generate(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature,
            skip_prompt=True, **kwargs,
        )[0]

        # Trim at the next turn boundary.
        for stop in ("\nHuman:", "\nAssistant:", "Human:"):
            if stop in completion:
                completion = completion.split(stop)[0]
        return completion.strip()

    # ------------------------------------------------------------------
    def perplexity(self, text: str) -> float:
        """Compute the model's perplexity on a piece of text (lower is better)."""
        import math

        ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        ids = ids[:, : self.model.config.max_position_embeddings]
        with torch.no_grad():
            loss = self.model(input_ids=ids, labels=ids)["loss"]
        return math.exp(loss.item())
