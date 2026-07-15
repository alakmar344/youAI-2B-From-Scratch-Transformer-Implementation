"""Evaluation tools for language models.

Provides:

* **Perplexity** — standard language modelling metric (lower is better).
* **Generation quality** — diversity, repetition, coherence metrics.
* **Text classification accuracy** — evaluate on multiple-choice benchmarks.
* **BLEU / ROUGE** — reference-based generation quality.
* **Embedding similarity** — semantic similarity between generated and reference.

Usage::

    from youai.evaluation import evaluate_perplexity, evaluate_generation

    # Perplexity on a text file
    results = evaluate_perplexity(model, tokenizer, "test.txt")

    # Generation quality metrics
    results = evaluate_generation(model, tokenizer, prompts=["What is AI?"])
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_logger, resolve_device

logger = get_logger()


# ======================================================================
# Perplexity
# ======================================================================
@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    tokenizer,
    data_path: str,
    max_length: int = 1024,
    stride: int = 512,
    device: str = "cpu",
    batch_size: int = 1,
) -> Dict[str, float]:
    """Compute perplexity on a text file using sliding window.

    Args:
        model: The language model.
        tokenizer: Tokenizer.
        data_path: Path to a text file.
        max_length: Context window size.
        stride: Sliding window stride.
        device: Device to run on.
        batch_size: Batch size for evaluation.

    Returns:
        Dict with ``perplexity``, ``num_tokens``, ``num_chunks``.
    """
    model = model.to(device).eval()
    dev = resolve_device(device) if isinstance(device, str) else device

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]
    total_len = input_ids.size(0)

    nlls = []
    num_tokens = 0

    for begin_loc in range(0, total_len, stride):
        end_loc = min(begin_loc + max_length, total_len)
        trg_len = end_loc - begin_loc
        if begin_loc > 0:
            trg_len = min(stride, end_loc - begin_loc)

        input_chunk = input_ids[None, begin_loc:end_loc].to(dev)
        target_chunk = input_chunk.clone()
        if trg_len < input_chunk.size(1):
            target_chunk[:, :-trg_len] = -100

        with torch.amp.autocast(dev.type, enabled=False):
            out = model(input_ids=input_chunk, labels=target_chunk)
            loss = out["loss"]

        # Only count the loss on the target tokens.
        nlls.append(loss.item() * trg_len)
        num_tokens += trg_len

        if end_loc >= total_len:
            break

    avg_nll = sum(nlls) / max(1, num_tokens)
    ppl = math.exp(min(avg_nll, 20))  # cap to avoid overflow

    return {
        "perplexity": round(ppl, 2),
        "avg_nll": round(avg_nll, 4),
        "num_tokens": num_tokens,
        "num_chunks": len(nlls),
    }


# ======================================================================
# Generation quality
# ======================================================================
def _distinct_n(tokens: List[str], n: int) -> float:
    """Fraction of unique n-grams (diversity metric)."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / max(1, len(ngrams))


def _repetition_rate(tokens: List[str], n: int = 4) -> float:
    """Fraction of tokens that are part of a repeated n-gram."""
    if len(tokens) < n:
        return 0.0
    seen = set()
    repeated = 0
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        if ngram in seen:
            repeated += 1
        seen.add(ngram)
    return repeated / max(1, len(tokens) - n + 1)


def _lexical_diversity(tokens: List[str]) -> float:
    """Type-token ratio (unique tokens / total tokens)."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


@torch.no_grad()
def evaluate_generation(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    num_samples: int = 1,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate generation quality metrics.

    Args:
        model: The language model.
        tokenizer: Tokenizer.
        prompts: List of prompts to generate from.
        max_new_tokens: Max tokens to generate per prompt.
        temperature: Sampling temperature.
        top_k: Top-k sampling.
        top_p: Nucleus sampling.
        num_samples: Number of samples per prompt.
        device: Device to run on.

    Returns:
        Dict with generation quality metrics.
    """
    from .generation import GenerationConfig

    model = model.to(device).eval()
    cfg = GenerationConfig(
        max_new_tokens=max_new_tokens, temperature=temperature,
        top_k=top_k, top_p=top_p, do_sample=temperature > 0,
        eos_token_id=tokenizer.eos_token_id,
    )

    all_generations = []
    all_lengths = []
    total_prompt_tokens = 0

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        total_prompt_tokens += input_ids.size(1)

        for _ in range(num_samples):
            output = model.generate(input_ids, cfg)
            gen_tokens = output[0, input_ids.size(1):].cpu().tolist()
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            all_generations.append(gen_text)
            all_lengths.append(len(gen_tokens))

    # Tokenize all generations for n-gram analysis.
    all_token_lists = []
    for text in all_generations:
        tokens = text.lower().split()
        all_token_lists.append(tokens)

    # Compute metrics.
    avg_length = sum(all_lengths) / max(1, len(all_lengths))

    # Distinct-n (diversity).
    all_flat = [t for tl in all_token_lists for t in tl]
    distinct_1 = _distinct_n(all_flat, 1)
    distinct_2 = _distinct_n(all_flat, 2)
    distinct_3 = _distinct_n(all_flat, 3)

    # Repetition rate (per generation).
    rep_rates = [_repetition_rate(tl) for tl in all_token_lists if len(tl) > 4]
    avg_rep = sum(rep_rates) / max(1, len(rep_rates)) if rep_rates else 0.0

    # Lexical diversity (per generation).
    diversities = [_lexical_diversity(tl) for tl in all_token_lists if tl]
    avg_diversity = sum(diversities) / max(1, len(diversities))

    return {
        "num_generations": len(all_generations),
        "avg_length_tokens": round(avg_length, 1),
        "distinct_1": round(distinct_1, 4),
        "distinct_2": round(distinct_2, 4),
        "distinct_3": round(distinct_3, 4),
        "repetition_rate": round(avg_rep, 4),
        "lexical_diversity": round(avg_diversity, 4),
        "prompts_evaluated": len(prompts),
    }


# ======================================================================
# BLEU / ROUGE (simple implementations, no external deps)
# ======================================================================
def compute_bleu(
    references: List[str],
    hypotheses: List[str],
    max_n: int = 4,
) -> Dict[str, float]:
    """Compute BLEU score (corpus-level).

    Args:
        references: List of reference texts.
        hypotheses: List of hypothesis texts.
        max_n: Maximum n-gram order.

    Returns:
        Dict with ``bleu`` (0-100) and per-n-gram precisions.
    """
    assert len(references) == len(hypotheses)

    def _ngrams(text: str, n: int) -> Counter:
        tokens = text.lower().split()
        return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

    precisions = []
    for n in range(1, max_n + 1):
        total_matches = 0
        total_count = 0
        for ref, hyp in zip(references, hypotheses):
            ref_ngrams = _ngrams(ref, n)
            hyp_ngrams = _ngrams(hyp, n)
            for ngram, count in hyp_ngrams.items():
                total_matches += min(count, ref_ngrams.get(ngram, 0))
            total_count += sum(hyp_ngrams.values())
        precisions.append(total_matches / max(1, total_count))

    # Brevity penalty.
    ref_len = sum(len(r.split()) for r in references)
    hyp_len = sum(len(h.split()) for h in hypotheses)
    bp = min(1.0, math.exp(1 - ref_len / max(1, hyp_len)))

    # Geometric mean of precisions.
    log_avg = sum(math.log(max(p, 1e-10)) for p in precisions) / max_n
    bleu = bp * math.exp(log_avg) * 100

    return {
        "bleu": round(bleu, 2),
        "bp": round(bp, 4),
        **{f"precision_{n}": round(p, 4) for n, p in enumerate(precisions, 1)},
    }


def compute_rouge_l(
    references: List[str],
    hypotheses: List[str],
) -> Dict[str, float]:
    """Compute ROUGE-L (longest common subsequence) F1 score.

    Args:
        references: List of reference texts.
        hypotheses: List of hypothesis texts.

    Returns:
        Dict with ``rouge_l_f1``, ``rouge_l_precision``, ``rouge_l_recall``.
    """
    def _lcs_len(x: List[str], y: List[str]) -> int:
        m, n = len(x), len(y)
        if m > 500 or n > 500:
            # For very long texts, use a faster approximation.
            return len(set(x) & set(y))
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    total_f1, total_prec, total_rec = 0.0, 0.0, 0.0
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()
        lcs = _lcs_len(ref_tokens, hyp_tokens)
        prec = lcs / max(1, len(hyp_tokens))
        rec = lcs / max(1, len(ref_tokens))
        f1 = 2 * prec * rec / max(1e-10, prec + rec)
        total_f1 += f1
        total_prec += prec
        total_rec += rec

    n = max(1, len(references))
    return {
        "rouge_l_f1": round(total_f1 / n * 100, 2),
        "rouge_l_precision": round(total_prec / n * 100, 2),
        "rouge_l_recall": round(total_rec / n * 100, 2),
    }


# ======================================================================
# Multiple-choice evaluation (MMLU-style)
# ======================================================================
@torch.no_grad()
def evaluate_multiple_choice(
    model: nn.Module,
    tokenizer,
    questions: List[Dict],
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate on multiple-choice questions (MMLU-style).

    Each question is a dict with:
        - ``question``: str
        - ``choices``: list of str (typically 4 options)
        - ``answer``: int (index of correct choice)

    Uses per-choice log-probability scoring (the choice with highest
    log-prob for its continuation is selected).

    Args:
        model: The language model.
        tokenizer: Tokenizer.
        questions: List of question dicts.
        device: Device.

    Returns:
        Dict with ``accuracy``, ``num_questions``, ``random_baseline``.
    """
    model = model.to(device).eval()
    correct = 0
    total = 0

    for q in questions:
        prompt = q["question"]
        choices = q["choices"]
        answer_idx = q["answer"]

        best_score = float("-inf")
        best_idx = 0

        for i, choice in enumerate(choices):
            text = f"{prompt}\n{choice}"
            input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

            out = model(input_ids=input_ids)
            logits = out["logits"]

            # Score only the choice tokens.
            choice_start = prompt_ids.size(1) - 1
            choice_logits = logits[0, choice_start:-1]
            choice_ids = input_ids[0, choice_start + 1:]

            log_probs = F.log_softmax(choice_logits, dim=-1)
            score = log_probs.gather(1, choice_ids.unsqueeze(1)).sum().item()

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx == answer_idx:
            correct += 1
        total += 1

    return {
        "accuracy": round(correct / max(1, total) * 100, 2),
        "correct": correct,
        "num_questions": total,
        "random_baseline": round(100 / max(1, len(questions[0]["choices"]) if questions else 4), 2),
    }


# ======================================================================
# Benchmark suite
# ======================================================================
def run_benchmark(
    model: nn.Module,
    tokenizer,
    data_path: Optional[str] = None,
    prompts: Optional[List[str]] = None,
    device: str = "cpu",
) -> Dict[str, object]:
    """Run a comprehensive benchmark suite on a model.

    Args:
        model: The model to evaluate.
        tokenizer: Tokenizer.
        data_path: Path to a text file for perplexity evaluation.
        prompts: Prompts for generation quality evaluation.
        device: Device.

    Returns:
        Dict with all benchmark results.
    """
    results = {}

    if data_path and os.path.exists(data_path):
        logger.info("Computing perplexity on %s...", data_path)
        results["perplexity"] = evaluate_perplexity(
            model, tokenizer, data_path, device=device,
        )

    if prompts:
        logger.info("Evaluating generation quality on %d prompts...", len(prompts))
        results["generation"] = evaluate_generation(
            model, tokenizer, prompts, device=device,
        )

    # Model info.
    results["model_info"] = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "parameters_formatted": f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M",
    }

    return results
