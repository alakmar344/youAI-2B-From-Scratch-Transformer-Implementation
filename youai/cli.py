"""Command-line interface for YouAI.

Examples::

    youai train --preset 125m --dataset tinystories --epochs 3 --mixed-precision bf16
    youai generate --checkpoint ./checkpoints/final --prompt "Hello"
    youai chat --checkpoint ./checkpoints/final
    youai export --checkpoint ./checkpoints/final --format onnx
    youai benchmark --checkpoint ./checkpoints/final
    youai info --preset 125m
    youai datasets
"""

from __future__ import annotations

import argparse
import sys


def cmd_train(args) -> None:
    import youai
    from youai.trainer import estimate_training_time

    youai.set_seed(args.seed)
    if args.pretrained:
        print(f"Loading pretrained '{args.pretrained}'...")
        model = youai.from_pretrained_gpt2(args.pretrained)
    else:
        print(f"Creating '{args.preset}' model...")
        model = youai.create_model(args.preset)

    if args.dataset:
        train_file, val_file = youai.download_dataset(args.dataset, num_examples=args.num_examples)
    elif args.data:
        train_file, val_file = args.data, args.val_data
    else:
        print("No data provided — generating a sample dataset.")
        train_file, val_file = youai.create_sample_data(args.num_examples or 2000)

    if args.estimate:
        est = estimate_training_time(model, args.num_examples or 10000, args.batch_size, args.epochs)
        print("\nEstimated: "
              f"{est['parameters_formatted']} params | ~{est['gpu_memory_gb']} GB | "
              f"~{est['estimated_time_hours']}h | ~${est['estimated_cost_usd']}")
        if not args.yes and input("Continue? [y/N] ").lower() != "y":
            return

    metrics = youai.train(
        model, train_file, val_file, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, max_length=args.max_length,
        output_dir=args.output_dir, device=args.device,
        mixed_precision=args.mixed_precision, gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.grad_accum, packing=not args.no_packing,
        resume_from=args.resume, use_accelerate=args.accelerate,
        lora=args.lora, qlora=args.qlora, lora_r=args.lora_r,
    )
    print(f"\nDone. Steps: {metrics['global_step']} | "
          f"best val loss: {metrics['best_val_loss']}")


def cmd_generate(args) -> None:
    import youai

    results = youai.generate(
        prompt=args.prompt, checkpoint_path=args.checkpoint, device=args.device,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty,
        num_return_sequences=args.num_sequences,
    )
    for i, text in enumerate(results, 1):
        if len(results) > 1:
            print(f"\n--- {i} ---")
        print(text)


def cmd_chat(args) -> None:
    import youai

    print("Loading model...")
    model = youai.load_model(args.checkpoint, device=args.device)
    print("\nYouAI chat — type 'quit' to exit, 'clear' to reset history.\n")
    history = []
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if user.lower() in ("quit", "exit"):
            break
        if user.lower() == "clear":
            history = []
            print("(history cleared)")
            continue
        if not user:
            continue
        response = model.chat(user, history=history, temperature=args.temperature)
        print(f"AI: {response}\n")
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": response})


def _load_raw_model(checkpoint: str):
    from youai.model import YouAIModel

    return YouAIModel.from_pretrained(checkpoint)


def cmd_export(args) -> None:
    from youai.export import ModelExporter, ModelQuantizer

    model = _load_raw_model(args.checkpoint)
    if args.format == "onnx":
        out = args.output or f"{args.checkpoint}/model.onnx"
        ModelExporter(model).export_onnx(out)
    elif args.format == "torchscript":
        out = args.output or f"{args.checkpoint}/model.pt"
        ModelExporter(model).export_torchscript(out)
    elif args.format == "quantized":
        out = args.output or f"{args.checkpoint}-quantized"
        q = ModelQuantizer(model)
        q.save(q.dynamic_quantize(args.dtype), out)


def cmd_benchmark(args) -> None:
    from youai.export import benchmark_model

    model = _load_raw_model(args.checkpoint)
    results = benchmark_model(model, device=args.device, num_runs=args.runs,
                              sequence_length=args.sequence_length)
    print("\nBenchmark:")
    for key, value in results.items():
        print(f"  {key:20s}: {value}")


def cmd_info(args) -> None:
    from youai.config import get_preset_config, list_presets

    if args.preset:
        config = get_preset_config(args.preset, modern=not args.classic)
        print(f"\n{args.preset.upper()} ({config.total_params_formatted} params)")
        for key, value in config.to_dict().items():
            print(f"  {key:26s}: {value}")
    else:
        print("\nAvailable presets:")
        for name, params in list_presets().items():
            print(f"  {name:8s} ~{params}")


def cmd_datasets(args) -> None:
    from youai.data import list_datasets

    list_datasets()


def cmd_serve(args) -> None:
    from youai.server import serve

    serve(checkpoint=args.checkpoint, pretrained=args.pretrained,
          host=args.host, port=args.port, device=args.device, max_batch=args.max_batch)


def build_parser() -> argparse.ArgumentParser:
    import youai

    parser = argparse.ArgumentParser(prog="youai", description="Train your own language model.")
    parser.add_argument("--version", action="version", version=f"youai {youai.__version__}")
    sub = parser.add_subparsers(dest="command")

    t = sub.add_parser("train", help="Train a model")
    t.add_argument("--preset", default="125m")
    t.add_argument("--pretrained", help="Load pretrained GPT-2 weights (gpt2, gpt2-medium, ...)")
    t.add_argument("--lora", action="store_true", help="LoRA fine-tuning (train <1%% of params)")
    t.add_argument("--qlora", action="store_true", help="QLoRA fine-tuning (4-bit base on CUDA)")
    t.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    t.add_argument("--accelerate", action="store_true", help="Multi-GPU/distributed via accelerate")
    t.add_argument("--data")
    t.add_argument("--val-data")
    t.add_argument("--dataset", help="HuggingFace dataset preset")
    t.add_argument("--num-examples", type=int)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--grad-accum", type=int, default=1)
    t.add_argument("--learning-rate", type=float, default=3e-4)
    t.add_argument("--max-length", type=int, default=512)
    t.add_argument("--output-dir", default="./checkpoints")
    t.add_argument("--device", default="auto")
    t.add_argument("--mixed-precision", choices=["fp16", "bf16"])
    t.add_argument("--gradient-checkpointing", action="store_true")
    t.add_argument("--no-packing", action="store_true", help="Disable token packing")
    t.add_argument("--resume")
    t.add_argument("--estimate", action="store_true")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("-y", "--yes", action="store_true")
    t.set_defaults(func=cmd_train)

    g = sub.add_parser("generate", help="Generate text")
    g.add_argument("--checkpoint", required=True)
    g.add_argument("--prompt", required=True)
    g.add_argument("--device", default="auto")
    g.add_argument("--max-new-tokens", type=int, default=100)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=50)
    g.add_argument("--top-p", type=float, default=0.9)
    g.add_argument("--repetition-penalty", type=float, default=1.1)
    g.add_argument("--num-sequences", type=int, default=1)
    g.set_defaults(func=cmd_generate)

    c = sub.add_parser("chat", help="Interactive chat")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--device", default="auto")
    c.add_argument("--temperature", type=float, default=0.7)
    c.set_defaults(func=cmd_chat)

    e = sub.add_parser("export", help="Export a model")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--format", required=True, choices=["onnx", "torchscript", "quantized"])
    e.add_argument("--output")
    e.add_argument("--dtype", default="qint8", choices=["qint8", "float16"])
    e.set_defaults(func=cmd_export)

    b = sub.add_parser("benchmark", help="Benchmark inference")
    b.add_argument("--checkpoint", required=True)
    b.add_argument("--device", default="cpu")
    b.add_argument("--runs", type=int, default=50)
    b.add_argument("--sequence-length", type=int, default=64)
    b.set_defaults(func=cmd_benchmark)

    i = sub.add_parser("info", help="Show model / preset info")
    i.add_argument("--preset")
    i.add_argument("--classic", action="store_true", help="Show classic (GPT-2 style) config")
    i.set_defaults(func=cmd_info)

    d = sub.add_parser("datasets", help="List dataset presets")
    d.set_defaults(func=cmd_datasets)

    s = sub.add_parser("serve", help="Run the FastAPI inference server")
    s.add_argument("--checkpoint", help="Checkpoint directory to serve")
    s.add_argument("--pretrained", help="Serve pretrained GPT-2 weights (e.g. gpt2)")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--device", default="auto")
    s.add_argument("--max-batch", type=int, default=8)
    s.set_defaults(func=cmd_serve)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
