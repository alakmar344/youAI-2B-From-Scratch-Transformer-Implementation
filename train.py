"""Legacy training entry point.

Kept for convenience — it simply forwards to the modern CLI, so
``python train.py --preset 125m --dataset tinystories`` works just like
``youai train ...``. See DOCUMENTATION.md for the full library API.
"""

import sys

from youai.cli import main

if __name__ == "__main__":
    # Default to the ``train`` sub-command when none is supplied.
    argv = sys.argv[1:]
    if not argv or argv[0].startswith("-"):
        argv = ["train", *argv]
    sys.exit(main(argv))
