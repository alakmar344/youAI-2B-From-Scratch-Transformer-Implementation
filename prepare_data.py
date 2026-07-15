"""Legacy data-preparation entry point.

Forwards to the modern helpers in :mod:`youai.data`. Examples::

    python prepare_data.py --sample --num-examples 5000
    python prepare_data.py --dataset tinystories --num-examples 50000
    python prepare_data.py --files a.txt b.txt
"""

import argparse

from youai.data import create_sample_dataset, download_dataset, prepare_custom_text, list_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare training data for YouAI.")
    parser.add_argument("--sample", action="store_true", help="Generate synthetic sample data")
    parser.add_argument("--dataset", help="HuggingFace dataset preset (see --list)")
    parser.add_argument("--files", nargs="+", help="Custom text files to split into train/val")
    parser.add_argument("--num-examples", type=int, default=10000)
    parser.add_argument("--output-dir", default="./data")
    parser.add_argument("--list", action="store_true", help="List dataset presets")
    args = parser.parse_args()

    if args.list:
        list_datasets()
        return
    if args.dataset:
        train, val = download_dataset(args.dataset, args.output_dir, num_examples=args.num_examples)
    elif args.files:
        train, val = prepare_custom_text(args.files, args.output_dir)
    else:
        train, val = create_sample_dataset(args.output_dir, args.num_examples)
    print(f"Train: {train}\nVal:   {val}")


if __name__ == "__main__":
    main()
