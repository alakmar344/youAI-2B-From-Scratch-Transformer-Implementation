"""Export and optimise YouAI models for deployment.

* ONNX / TorchScript export (via a thin wrapper that returns a plain logits
  tensor — the model's dict output cannot be traced directly).
* Dynamic int8 / fp16 quantization for smaller, faster CPU inference.
* A benchmarking helper and GGUF metadata export.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict

import torch
import torch.nn as nn

from .utils import get_logger, human_bytes, model_size_bytes

logger = get_logger()


class _LogitsWrapper(nn.Module):
    """Wrap a YouAIModel so it returns a bare logits tensor (traceable)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)["logits"]


class ModelExporter:
    """Export a model to ONNX or TorchScript."""

    def __init__(self, model: nn.Module, tokenizer=None):
        self.model = model.cpu().eval()
        self.tokenizer = tokenizer
        self.wrapper = _LogitsWrapper(self.model).eval()

    def export_onnx(self, output_path: str, opset_version: int = 17,
                    dynamic_axes: bool = True, seq_len: int = 32) -> str:
        try:
            import onnx  # noqa: F401
        except ImportError as exc:
            raise ImportError("Install ONNX: pip install onnx onnxruntime") from exc

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        dummy = torch.randint(0, self.model.config.vocab_size, (1, seq_len))
        axes = None
        if dynamic_axes:
            axes = {"input_ids": {0: "batch", 1: "sequence"},
                    "logits": {0: "batch", 1: "sequence"}}
        torch.onnx.export(
            self.wrapper, dummy, output_path, opset_version=opset_version,
            input_names=["input_ids"], output_names=["logits"],
            dynamic_axes=axes, do_constant_folding=True,
        )
        import onnx

        onnx.checker.check_model(onnx.load(output_path))
        logger.info("Exported ONNX model to %s", output_path)
        return output_path

    def export_torchscript(self, output_path: str, seq_len: int = 32) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        dummy = torch.randint(0, self.model.config.vocab_size, (1, seq_len))
        traced = torch.jit.trace(self.wrapper, dummy, check_trace=False)
        traced.save(output_path)
        logger.info("Exported TorchScript model to %s", output_path)
        return output_path

    def export_config(self, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(self.model.config.to_dict(), f, indent=2)
        info = {
            "model_type": "youai",
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "size": human_bytes(model_size_bytes(self.model)),
        }
        with open(os.path.join(output_dir, "model_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        return output_dir


class ModelQuantizer:
    """Reduce model size / speed up CPU inference via quantization."""

    def __init__(self, model: nn.Module):
        self.model = model.cpu().eval()

    def dynamic_quantize(self, dtype: str = "qint8") -> nn.Module:
        original = model_size_bytes(self.model)
        if dtype == "qint8":
            quantized = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear}, dtype=torch.qint8
            )
        elif dtype in ("float16", "fp16"):
            quantized = self.model.half()
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        new = model_size_bytes(quantized)
        logger.info("Quantized (%s): %s -> %s (%.1f%% smaller)", dtype,
                    human_bytes(original), human_bytes(new), (1 - new / original) * 100)
        return quantized

    def save(self, model: nn.Module, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        if hasattr(model, "config"):
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump(model.config.to_dict(), f, indent=2)
        logger.info("Saved quantized model to %s", output_dir)
        return output_dir


def create_gguf_metadata(model: nn.Module, tokenizer, output_path: str) -> str:
    """Write GGUF-compatible metadata for llama.cpp conversion tooling."""
    config = model.config
    metadata = {
        "general": {"architecture": "youai", "name": "YouAI Model", "file_type": "F32"},
        "llama": {
            "vocab_size": config.vocab_size,
            "embedding_length": config.hidden_size,
            "block_count": config.num_hidden_layers,
            "feed_forward_length": config.intermediate_size,
            "attention.head_count": config.num_attention_heads,
            "attention.head_count_kv": config.num_key_value_heads,
            "rope.freq_base": config.rope_theta,
        },
        "tokenizer": {"ggml.model": "gpt2"},
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Wrote GGUF metadata to %s", output_path)
    return output_path


def benchmark_model(model: nn.Module, device: str = "cpu", num_runs: int = 50,
                    sequence_length: int = 64, warmup: int = 5) -> Dict[str, float]:
    """Benchmark forward-pass latency and throughput."""
    from .utils import resolve_device

    dev = resolve_device(device)
    model = model.to(dev).eval()
    dummy = torch.randint(0, model.config.vocab_size, (1, sequence_length)).to(dev)

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            model(dummy)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        total = time.perf_counter() - start

    return {
        "total_time_seconds": round(total, 3),
        "avg_inference_ms": round(total / num_runs * 1000, 2),
        "tokens_per_second": round(sequence_length * num_runs / total, 1),
        "sequence_length": sequence_length,
        "num_runs": num_runs,
        "device": str(dev),
    }
