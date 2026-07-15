"""YouAI Command Line Interface

Provides a unified CLI for all YouAI operations.

Usage:
    # Create and train a model
    youai train --preset 125m --data ./data/train.txt --epochs 3
    
    # Generate text
    youai generate --checkpoint ./checkpoints/final --prompt "Hello"
    
    # Interactive chat
    youai chat --checkpoint ./checkpoints/final
    
    # Export model
    youai export --checkpoint ./checkpoints/final --format onnx
    
    # Benchmark model
    youai benchmark --checkpoint ./checkpoints/final
"""

import argparse
import sys
import os
from typing import Optional


def cmd_train(args):
    """Train a model."""
    from . import create_model, train, download_dataset, create_sample_data
    from .training_advanced import MixedPrecisionTrainer, TrainingConfig, estimate_training_time
    
    # Create model
    print(f"Creating {args.preset} model...")
    model = create_model(args.preset)
    
    # Prepare data
    if args.dataset:
        print(f"Downloading {args.dataset} dataset...")
        train_file, val_file = download_dataset(args.dataset, num_examples=args.num_examples)
    elif args.data:
        train_file = args.data
        val_file = args.val_data
    else:
        print("Creating sample dataset...")
        train_file, val_file = create_sample_data(num_examples=args.num_examples or 1000)
    
    # Estimate training
    if args.estimate:
        estimates = estimate_training_time(
            model, 
            dataset_size=10000,  # Rough estimate
            batch_size=args.batch_size,
            num_epochs=args.epochs,
        )
        print("\nTraining Estimates:")
        print(f"  Parameters: {estimates['parameters_formatted']}")
        print(f"  GPU Memory: {estimates['gpu_memory_gb']} GB")
        print(f"  Estimated Time: {estimates['estimated_time_hours']} hours")
        print(f"  Estimated Cost: ${estimates['estimated_cost_usd']}")
        if not args.yes:
            response = input("\nContinue? (y/n): ")
            if response.lower() != 'y':
                return
    
    # Train
    config = TrainingConfig(
        mixed_precision=args.mixed_precision,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    
    trainer = MixedPrecisionTrainer(
        model=model,
        train_dataloader=None,  # Will be created by train()
        config=config,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        output_dir=args.output_dir,
    )
    
    from . import create_dataloaders
    train_loader, val_loader = create_dataloaders(
        train_file=train_file,
        val_file=val_file,
        batch_size=args.batch_size,
    )
    
    trainer.train_dataloader = train_loader
    trainer.val_dataloader = val_loader
    
    if args.resume:
        print(f"Resuming from {args.resume}...")
        trainer.load_checkpoint(args.resume)
    
    trainer.train()


def cmd_generate(args):
    """Generate text."""
    from . import generate
    
    results = generate(
        prompt=args.prompt,
        checkpoint_path=args.checkpoint,
        max_length=args.max_length,
        temperature=args.temperature,
        num_return_sequences=args.num_sequences,
    )
    
    for i, result in enumerate(results, 1):
        if len(results) > 1:
            print(f"\n--- Response {i} ---")
        print(result)


def cmd_chat(args):
    """Interactive chat."""
    from . import load_model
    
    print("Loading model...")
    model = load_model(args.checkpoint)
    
    print("\n" + "=" * 50)
    print("YouAI Chat")
    print("Type 'quit' to exit, 'clear' to clear history")
    print("=" * 50 + "\n")
    
    history = []
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            history = []
            print("History cleared!")
            continue
        
        if not user_input:
            continue
        
        # Generate response
        response = model.chat(user_input, history=history)
        print(f"AI: {response}\n")
        
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})


def cmd_export(args):
    """Export model."""
    from . import load_model
    from .export import ModelExporter, ModelQuantizer, create_gguf_metadata
    
    print(f"Loading model from {args.checkpoint}...")
    
    if args.format == "onnx":
        from .model import YouAIModel
        from .config import YouAIConfig
        import json
        import torch
        
        config_path = os.path.join(args.checkpoint, "config.json")
        with open(config_path) as f:
            config = YouAIConfig.from_dict(json.load(f))
        
        model = YouAIModel(config)
        weights = torch.load(os.path.join(args.checkpoint, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(weights['model_state_dict'])
        
        exporter = ModelExporter(model)
        output_path = os.path.join(args.output or args.checkpoint, "model.onnx")
        exporter.export_onnx(output_path)
        
    elif args.format == "quantized":
        from .model import YouAIModel
        from .config import YouAIConfig
        import json
        import torch
        
        config_path = os.path.join(args.checkpoint, "config.json")
        with open(config_path) as f:
            config = YouAIConfig.from_dict(json.load(f))
        
        model = YouAIModel(config)
        weights = torch.load(os.path.join(args.checkpoint, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(weights['model_state_dict'])
        
        quantizer = ModelQuantizer(model)
        quantized = quantizer.dynamic_quantize()
        output_dir = args.output or args.checkpoint + "-quantized"
        quantizer.save(quantized, output_dir)
        
    else:
        print(f"Unknown format: {args.format}")


def cmd_benchmark(args):
    """Benchmark model."""
    from .model import YouAIModel
    from .config import YouAIConfig
    from .export import benchmark_model
    import json
    import torch
    
    print(f"Loading model from {args.checkpoint}...")
    
    config_path = os.path.join(args.checkpoint, "config.json")
    with open(config_path) as f:
        config = YouAIConfig.from_dict(json.load(f))
    
    model = YouAIModel(config)
    weights = torch.load(os.path.join(args.checkpoint, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(weights['model_state_dict'])
    
    print("\nRunning benchmark...")
    results = benchmark_model(
        model,
        device=args.device,
        num_runs=args.runs,
        sequence_length=args.sequence_length,
    )
    
    print("\nBenchmark Results:")
    print(f"  Device: {results['device']}")
    print(f"  Sequence Length: {results['sequence_length']}")
    print(f"  Avg Inference: {results['avg_inference_ms']} ms")
    print(f"  Tokens/Second: {results['tokens_per_second']}")


def cmd_info(args):
    """Show model info."""
    from .config import get_preset_config
    
    if args.preset:
        config = get_preset_config(args.preset)
        print(f"\n{args.preset.upper()} Model Configuration:")
        print(f"  Hidden Size: {config.hidden_size}")
        print(f"  Layers: {config.num_hidden_layers}")
        print(f"  Attention Heads: {config.num_attention_heads}")
        print(f"  Intermediate Size: {config.intermediate_size}")
        print(f"  Vocab Size: {config.vocab_size}")
        print(f"  Est. Parameters: {config.total_params:,}")
    else:
        print("\nAvailable presets: 125m, 350m, 750m, 2b")
        print("Use: youai info --preset 125m")


def cmd_datasets(args):
    """List available datasets."""
    from .data import list_datasets
    list_datasets()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="youai",
        description="YouAI - Train Your Own Language Model",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--preset", default="125m", help="Model preset (125m, 350m, 750m, 2b)")
    train_parser.add_argument("--data", help="Training data file")
    train_parser.add_argument("--val-data", help="Validation data file")
    train_parser.add_argument("--dataset", help="HuggingFace dataset name")
    train_parser.add_argument("--num-examples", type=int, help="Number of examples")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    train_parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    train_parser.add_argument("--output-dir", default="./checkpoints", help="Output directory")
    train_parser.add_argument("--mixed-precision", choices=["fp16", "bf16"], help="Mixed precision")
    train_parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    train_parser.add_argument("--resume", help="Resume from checkpoint")
    train_parser.add_argument("--estimate", action="store_true", help="Show training estimates")
    train_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text")
    gen_parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    gen_parser.add_argument("--prompt", required=True, help="Input prompt")
    gen_parser.add_argument("--max-length", type=int, default=100, help="Max tokens")
    gen_parser.add_argument("--temperature", type=float, default=0.8, help="Temperature")
    gen_parser.add_argument("--num-sequences", type=int, default=1, help="Number of responses")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    chat_parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export model")
    export_parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    export_parser.add_argument("--format", required=True, choices=["onnx", "quantized"], help="Export format")
    export_parser.add_argument("--output", help="Output path")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark model")
    bench_parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    bench_parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    bench_parser.add_argument("--runs", type=int, default=100, help="Number of runs")
    bench_parser.add_argument("--sequence-length", type=int, default=64, help="Sequence length")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show model info")
    info_parser.add_argument("--preset", help="Model preset")
    
    # Datasets command
    subparsers.add_parser("datasets", help="List available datasets")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    commands = {
        "train": cmd_train,
        "generate": cmd_generate,
        "chat": cmd_chat,
        "export": cmd_export,
        "benchmark": cmd_benchmark,
        "info": cmd_info,
        "datasets": cmd_datasets,
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()
