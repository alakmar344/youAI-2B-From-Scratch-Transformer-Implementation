"""YouAI Model Export and Quantization

Features:
- Export to ONNX format for cross-platform inference
- Dynamic quantization for smaller model size
- Model pruning for deployment
- GGUF export preparation (for llama.cpp compatibility)
"""

import os
import json
from typing import Optional, Dict, Any, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelExporter:
    """Export YouAI models to various formats.
    
    Supported formats:
    - PyTorch (native)
    - ONNX (cross-platform)
    - TorchScript (production deployment)
    - Quantized (smaller size, faster inference)
    
    Usage:
        from youai.export import ModelExporter
        
        exporter = ModelExporter(model, tokenizer)
        exporter.export_onnx("model.onnx")
        exporter.export_torchscript("model.pt")
    """
    
    def __init__(self, model: nn.Module, tokenizer=None):
        """
        Args:
            model: YouAIModel instance
            tokenizer: Tokenizer for export verification
        """
        self.model = model.cpu().eval()
        self.tokenizer = tokenizer
    
    def export_onnx(
        self,
        output_path: str,
        opset_version: int = 14,
        dynamic_axes: bool = True,
    ) -> str:
        """Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
            dynamic_axes: Support dynamic sequence lengths
            
        Returns:
            Path to exported model
        """
        try:
            import onnx
        except ImportError:
            raise ImportError("ONNX not installed. Run: pip install onnx onnxruntime")
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Create dummy input
        dummy_input = torch.randint(0, 1000, (1, 32))
        
        # Dynamic axes for variable length sequences
        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size', 1: 'sequence_length'}
            }
        
        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            opset_version=opset_version,
            input_names=['input_ids'],
            output_names=['output'],
            dynamic_axes=dynamic_axes_dict,
            do_constant_folding=True,
        )
        
        # Verify
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        print(f"ONNX model exported to: {output_path}")
        return output_path
    
    def export_torchscript(self, output_path: str) -> str:
        """Export model to TorchScript format.
        
        Args:
            output_path: Path to save TorchScript model
            
        Returns:
            Path to exported model
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Trace the model
        dummy_input = torch.randint(0, 1000, (1, 32))
        traced_model = torch.jit.trace(self.model, dummy_input)
        
        # Save
        traced_model.save(output_path)
        
        print(f"TorchScript model exported to: {output_path}")
        return output_path
    
    def export_config(self, output_dir: str) -> str:
        """Export model configuration and metadata.
        
        Args:
            output_dir: Directory to save config files
            
        Returns:
            Path to config directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save config
        if hasattr(self.model, 'config'):
            config_path = os.path.join(output_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(self.model.config.to_dict(), f, indent=2)
        
        # Save model info
        info = {
            "model_type": "youai",
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "architecture": {
                "hidden_size": getattr(self.model.config, 'hidden_size', None),
                "num_layers": getattr(self.model.config, 'num_hidden_layers', None),
                "num_heads": getattr(self.model.config, 'num_attention_heads', None),
            }
        }
        
        info_path = os.path.join(output_dir, "model_info.json")
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"Model config exported to: {output_dir}")
        return output_dir


class ModelQuantizer:
    """Quantize YouAI models for faster inference and smaller size.
    
    Quantization reduces model size by 2-4x with minimal quality loss.
    
    Supported methods:
    - Dynamic quantization (recommended for CPU inference)
    - Static quantization (requires calibration data)
    
    Usage:
        from youai.export import ModelQuantizer
        
        quantizer = ModelQuantizer(model)
        quantized_model = quantizer.dynamic_quantize()
        quantizer.save(quantized_model, "quantized_model")
    """
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: YouAIModel instance to quantize
        """
        self.model = model
    
    def dynamic_quantize(
        self,
        dtype: str = "qint8",
        inplace: bool = False,
    ) -> nn.Module:
        """Apply dynamic quantization.
        
        Dynamic quantization quantizes weights to int8 but keeps
        activations in float. Best for CPU inference.
        
        Args:
            dtype: Quantization dtype ('qint8' or 'float16')
            inplace: Modify model in place
            
        Returns:
            Quantized model
        """
        if dtype == "qint8":
            quantized = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear},  # Quantize linear layers
                dtype=torch.qint8
            )
        elif dtype == "float16":
            quantized = self.model.half()
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        # Calculate size reduction
        original_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        quantized_size = sum(p.numel() * p.element_size() for p in quantized.parameters())
        reduction = (1 - quantized_size / original_size) * 100
        
        print(f"Quantization complete:")
        print(f"  Original size: {original_size / 1024**2:.1f} MB")
        print(f"  Quantized size: {quantized_size / 1024**2:.1f} MB")
        print(f"  Reduction: {reduction:.1f}%")
        
        return quantized
    
    def save(self, model: nn.Module, output_dir: str):
        """Save quantized model.
        
        Args:
            model: Quantized model
            output_dir: Directory to save model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        
        # Save config if available
        if hasattr(model, 'config'):
            with open(os.path.join(output_dir, "config.json"), 'w') as f:
                json.dump(model.config.to_dict(), f, indent=2)
        
        print(f"Quantized model saved to: {output_dir}")


def create_gguf_metadata(
    model: nn.Module,
    tokenizer,
    output_path: str,
) -> str:
    """Create GGUF-compatible metadata for llama.cpp conversion.
    
    This creates the metadata needed to convert a YouAI model
    to GGUF format for use with llama.cpp.
    
    Args:
        model: YouAIModel instance
        tokenizer: Tokenizer
        output_path: Path to save metadata
        
    Returns:
        Path to metadata file
    """
    if not hasattr(model, 'config'):
        raise ValueError("Model must have a config attribute")
    
    config = model.config
    
    metadata = {
        "general": {
            "architecture": "youai",
            "name": "YouAI Model",
            "file_type": "F32",
        },
        "llama": {
            "vocab_size": config.vocab_size,
            "embedding_length": config.hidden_size,
            "block_count": config.num_hidden_layers,
            "feed_forward_length": config.intermediate_size,
            "attention.head_count": config.num_attention_heads,
            "rope.freq_base": 10000.0,
        },
        "tokenizer": {
            "ggml.model": "gpt2",
            "ggml.tokens": list(tokenizer.get_vocab().keys()),
        }
    }
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"GGUF metadata saved to: {output_path}")
    print("To convert to GGUF, use: python convert-hf-to-gguf.py")
    return output_path


def benchmark_model(
    model: nn.Module,
    device: str = "cpu",
    num_runs: int = 100,
    sequence_length: int = 64,
) -> Dict[str, float]:
    """Benchmark model inference speed.
    
    Args:
        model: Model to benchmark
        device: Device to run on
        num_runs: Number of inference runs
        sequence_length: Input sequence length
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    model = model.to(device).eval()
    dummy_input = torch.randint(0, 1000, (1, sequence_length)).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(dummy_input)
    
    # Benchmark
    if device == "cuda":
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            model(dummy_input)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    end = time.perf_counter()
    
    total_time = end - start
    avg_time_ms = (total_time / num_runs) * 1000
    tokens_per_second = (sequence_length * num_runs) / total_time
    
    return {
        "total_time_seconds": round(total_time, 2),
        "avg_inference_ms": round(avg_time_ms, 2),
        "tokens_per_second": round(tokens_per_second, 1),
        "sequence_length": sequence_length,
        "num_runs": num_runs,
        "device": device,
    }
