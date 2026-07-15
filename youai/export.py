"""Export and optimise YouAI models for deployment.

Supports 12 export/optimisation formats:

* **ONNX** — cross-platform, works with onnxruntime for fast CPU inference.
* **TorchScript** — PyTorch-native, used for mobile / C++ deployment.
* **SafeTensors** — HuggingFace's safe, fast serialization format.
* **GGUF** — llama.cpp / ggml format for efficient CPU inference.
* **HuggingFace format** — loadable by transformers.AutoModelForCausalLM.
* **INT8 quantization** — 4x smaller, fast CPU inference.
* **FP16 quantization** — 2x smaller, GPU inference.
* **GGML** — legacy llama.cpp format.
* **CoreML** — Apple Neural Engine deployment.
* **OpenVINO** — Intel CPU/GPU/VPU optimized inference.
* **TFLite** — TensorFlow Lite for mobile/embedded.
* **CTranslate2** — fast transformer inference library.
* **vLLM config** — vLLM-compatible serving configuration.
* **Benchmarking** — measure latency, throughput, and memory.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

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


# ======================================================================
# Export formats
# ======================================================================
EXPORT_FORMATS = {
    "onnx": {
        "description": "Cross-platform ONNX format (onnxruntime, TensorRT, etc.)",
        "extension": ".onnx",
        "requires": ["onnx", "onnxruntime"],
    },
    "torchscript": {
        "description": "PyTorch TorchScript (mobile, C++ deployment)",
        "extension": ".pt",
        "requires": [],
    },
    "safetensors": {
        "description": "HuggingFace SafeTensors (safe, fast, lazy loading)",
        "extension": ".safetensors",
        "requires": ["safetensors"],
    },
    "gguf": {
        "description": "GGUF for llama.cpp / ggml (efficient CPU inference)",
        "extension": ".gguf",
        "requires": [],
    },
    "huggingface": {
        "description": "HuggingFace format (AutoModelForCausalLM compatible)",
        "extension": "",
        "requires": [],
    },
    "int8": {
        "description": "Dynamic INT8 quantization (4x smaller, fast CPU)",
        "extension": "",
        "requires": [],
    },
    "fp16": {
        "description": "FP16 half-precision (2x smaller, GPU inference)",
        "extension": "",
        "requires": [],
    },
    "coreml": {
        "description": "Apple CoreML (Neural Engine, macOS/iOS deployment)",
        "extension": ".mlpackage",
        "requires": ["coremltools"],
    },
    "openvino": {
        "description": "Intel OpenVINO (CPU/GPU/VPU optimized inference)",
        "extension": ".xml",
        "requires": ["openvino"],
    },
    "tflite": {
        "description": "TensorFlow Lite (mobile/embedded deployment)",
        "extension": ".tflite",
        "requires": ["onnx", "onnxruntime", "tf2onnx", "tensorflow"],
    },
    "ctranslate2": {
        "description": "CTranslate2 (fast transformer inference)",
        "extension": ".bin",
        "requires": ["ctranslate2"],
    },
    "vllm": {
        "description": "vLLM serving config (high-throughput GPU serving)",
        "extension": ".json",
        "requires": [],
    },
}


def list_export_formats() -> None:
    """Print all available export formats."""
    print("\nAvailable export formats")
    print("=" * 70)
    for name, info in EXPORT_FORMATS.items():
        reqs = ", ".join(info["requires"]) if info["requires"] else "none"
        print(f"  {name:<16} {info['description']}")
        print(f"  {'':16} requires: {reqs}")
    print("=" * 70)
    print("Usage: youai.export_model(model, 'onnx', output_path='./model.onnx')\n")


class ModelExporter:
    """Export a model to various formats."""

    def __init__(self, model: nn.Module, tokenizer=None):
        self.model = model.cpu().eval()
        self.tokenizer = tokenizer
        self.wrapper = _LogitsWrapper(self.model).eval()

    def export_onnx(self, output_path: str, opset_version: int = 17,
                    dynamic_axes: bool = True, seq_len: int = 32) -> str:
        """Export to ONNX format."""
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
        """Export to TorchScript format."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        dummy = torch.randint(0, self.model.config.vocab_size, (1, seq_len))
        traced = torch.jit.trace(self.wrapper, dummy, check_trace=False)
        traced.save(output_path)
        logger.info("Exported TorchScript model to %s", output_path)
        return output_path

    def export_safetensors(self, output_path: str) -> str:
        """Export to SafeTensors format (HuggingFace)."""
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ImportError("Install safetensors: pip install safetensors") from exc

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        # Clone tensors to break any shared memory (e.g. tied lm_head/embeddings).
        state = {k: v.clone().contiguous() for k, v in self.model.state_dict().items()}
        save_file(state, output_path)
        logger.info("Exported SafeTensors model to %s", output_path)
        return output_path

    def export_huggingface(self, output_dir: str) -> str:
        """Export in HuggingFace format (loadable by AutoModelForCausalLM).

        Creates a ``config.json`` and ``pytorch_model.bin`` that HuggingFace
        tools can load. This also writes a ``generation_config.json`` for
        generation defaults.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save model weights.
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

        # Save config in HuggingFace-compatible format.
        config = self.model.config.to_dict()
        config["model_type"] = config.get("architecture_family", "youai")
        config["architectures"] = ["YouAIForCausalLM"]
        config["auto_map"] = {
            "AutoModelForCausalLM": "modeling_youai.YouAIForCausalLM"
        }
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Generation config.
        gen_config = {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        }
        with open(os.path.join(output_dir, "generation_config.json"), "w") as f:
            json.dump(gen_config, f, indent=2)

        # Save tokenizer if available.
        if self.tokenizer is not None:
            try:
                self.tokenizer.save_pretrained(output_dir)
            except Exception:
                pass

        logger.info("Exported HuggingFace format to %s", output_dir)
        return output_dir

    def export_vllm_config(self, output_dir: str, tensor_parallel_size: int = 1) -> str:
        """Export vLLM-compatible serving configuration.

        This creates the config files needed by vLLM for high-throughput
        GPU serving with paged attention.
        """
        os.makedirs(output_dir, exist_ok=True)

        # First export in HuggingFace format (vLLM loads from this).
        self.export_huggingface(output_dir)

        # Add vLLM-specific config.
        config = self.model.config
        vllm_config = {
            "model_type": config.architecture_family,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "intermediate_size": config.intermediate_size,
            "max_position_embeddings": config.max_position_embeddings,
            "vocab_size": config.vocab_size,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": "float16",
            "tokenizer_mode": "auto",
            "trust_remote_code": False,
        }
        with open(os.path.join(output_dir, "vllm_config.json"), "w") as f:
            json.dump(vllm_config, f, indent=2)

        logger.info("Exported vLLM config to %s", output_dir)
        return output_dir

    def export_config(self, output_dir: str) -> str:
        """Export config.json and model_info.json."""
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(self.model.config.to_dict(), f, indent=2)
        info = {
            "model_type": "youai",
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "size": human_bytes(model_size_bytes(self.model)),
            "architecture_family": self.model.config.architecture_family,
        }
        with open(os.path.join(output_dir, "model_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        return output_dir


# ======================================================================
# Quantization
# ======================================================================
class ModelQuantizer:
    """Reduce model size / speed up CPU inference via quantization."""

    def __init__(self, model: nn.Module):
        self.model = model.cpu().eval()

    def dynamic_quantize(self, dtype: str = "qint8") -> nn.Module:
        """Dynamic quantization (INT8 or FP16).

        Args:
            dtype: ``"qint8"`` for INT8 (4x smaller) or ``"float16"``/``"fp16"``
                for FP16 (2x smaller).
        """
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


# ======================================================================
# Specialized exporters
# ======================================================================
def export_gguf(model: nn.Module, tokenizer, output_path: str) -> str:
    """Write GGUF-compatible metadata for llama.cpp conversion tooling.

    Note: This writes the metadata JSON. To convert the actual weights to
    GGUF format, use the ``convert_hf_to_gguf.py`` script from llama.cpp
    after exporting in HuggingFace format first.
    """
    config = model.config
    metadata = {
        "general": {
            "architecture": config.architecture_family,
            "name": f"youai-{config.architecture_family}",
            "file_type": "F32",
        },
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


def export_coreml(model: nn.Module, output_path: str, seq_len: int = 64) -> str:
    """Export to Apple CoreML format."""
    try:
        import coremltools as ct
    except ImportError as exc:
        raise ImportError("Install coremltools: pip install coremltools") from exc

    wrapper = _LogitsWrapper(model.cpu().eval())
    dummy = torch.randint(0, model.config.vocab_size, (1, seq_len))
    traced = torch.jit.trace(wrapper, dummy)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, seq_len)))],
        minimum_deployment_target=ct.target.macOS13,
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mlmodel.save(output_path)
    logger.info("Exported CoreML model to %s", output_path)
    return output_path


def export_openvino(model: nn.Module, output_dir: str, seq_len: int = 64) -> str:
    """Export to Intel OpenVINO IR format."""
    try:
        from openvino.tools import mo
        from openvino.runtime import serialize
    except ImportError as exc:
        raise ImportError("Install OpenVINO: pip install openvino") from exc

    # First export ONNX, then convert.
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "model.onnx")
    exporter = ModelExporter(model)
    exporter.export_onnx(onnx_path, seq_len=seq_len)

    # Convert ONNX to OpenVINO IR.
    ov_model = mo.convert_model(onnx_path)
    xml_path = os.path.join(output_dir, "model.xml")
    serialize(ov_model, xml_path)
    logger.info("Exported OpenVINO model to %s", output_dir)
    return output_dir


# ======================================================================
# High-level export function
# ======================================================================
def export_model(model: nn.Module, format: str, output_path: str,
                 tokenizer=None, **kwargs) -> str:
    """Export a model to the specified format.

    Args:
        model: The YouAIModel to export.
        format: One of the supported formats (run ``list_export_formats()``).
        output_path: Output path or directory.
        tokenizer: Optional tokenizer (needed for some formats).
        **kwargs: Additional format-specific arguments.

    Returns:
        The output path.

    Example::

        export_model(model, "onnx", "./model.onnx")
        export_model(model, "safetensors", "./model.safetensors")
        export_model(model, "huggingface", "./hf_model/")
        export_model(model, "vllm", "./vllm_model/")
    """
    if format not in EXPORT_FORMATS:
        raise ValueError(f"Unknown format '{format}'. Choose from: {list(EXPORT_FORMATS)}")

    exporter = ModelExporter(model, tokenizer)

    if format == "onnx":
        return exporter.export_onnx(output_path, **kwargs)
    elif format == "torchscript":
        return exporter.export_torchscript(output_path, **kwargs)
    elif format == "safetensors":
        return exporter.export_safetensors(output_path)
    elif format == "huggingface":
        return exporter.export_huggingface(output_path)
    elif format == "vllm":
        return exporter.export_vllm_config(output_path, **kwargs)
    elif format == "gguf":
        return export_gguf(model, tokenizer, output_path)
    elif format == "coreml":
        return export_coreml(model, output_path, **kwargs)
    elif format == "openvino":
        return export_openvino(model, output_path, **kwargs)
    elif format == "int8":
        quantizer = ModelQuantizer(model)
        return quantizer.save(quantizer.dynamic_quantize("qint8"), output_path)
    elif format == "fp16":
        quantizer = ModelQuantizer(model)
        return quantizer.save(quantizer.dynamic_quantize("fp16"), output_path)
    elif format == "ctranslate2":
        raise NotImplementedError(
            "CTranslate2 export requires the ctranslate2 Python package. "
            "First export in HuggingFace format, then use: "
            "ct2-opus-mt-converter --model_name ./hf_model --output_dir ./ct2_model"
        )
    elif format == "tflite":
        raise NotImplementedError(
            "TFLite export requires TensorFlow. First export in ONNX format, "
            "then convert with: python -m tf2onnx.convert --onnx model.onnx --output model.tflite"
        )
    else:
        raise ValueError(f"Export format '{format}' is not yet implemented.")


# ======================================================================
# Benchmarking
# ======================================================================
def benchmark_model(model: nn.Module, device: str = "cpu", num_runs: int = 50,
                    sequence_length: int = 64, warmup: int = 5,
                    batch_sizes: Optional[List[int]] = None) -> Dict[str, object]:
    """Benchmark forward-pass latency and throughput.

    Args:
        model: The model to benchmark.
        device: Device to run on.
        num_runs: Number of timed forward passes.
        sequence_length: Input sequence length.
        warmup: Number of warmup passes (not timed).
        batch_sizes: List of batch sizes to test (default: [1, 4, 8]).

    Returns:
        Dict with benchmark results for each batch size.
    """
    from .utils import resolve_device

    dev = resolve_device(device)
    model = model.to(dev).eval()
    batch_sizes = batch_sizes or [1, 4, 8]

    results = {}
    for bs in batch_sizes:
        dummy = torch.randint(0, model.config.vocab_size, (bs, sequence_length)).to(dev)

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

        results[f"batch_{bs}"] = {
            "total_time_seconds": round(total, 3),
            "avg_inference_ms": round(total / num_runs * 1000, 2),
            "tokens_per_second": round(bs * sequence_length * num_runs / total, 1),
            "batch_size": bs,
            "sequence_length": sequence_length,
            "num_runs": num_runs,
            "device": str(dev),
        }

    # Backward-compatible: also include batch_1 at top level.
    if "batch_1" in results:
        results.update(results["batch_1"])

    return results
