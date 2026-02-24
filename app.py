#!/usr/bin/env python3
"""Application entrypoint."""

from pathlib import Path
import sys

# Keep local package import simple without requiring installation.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sentiment.cli import run


if __name__ == "__main__":
    run()
