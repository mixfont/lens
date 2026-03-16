#!/usr/bin/env python3
"""
Run Lens font inference for a single image URL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lens_inference import (
  DEFAULT_TOP_K,
  resolve_model_path,
  run_inference_from_url,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = REPO_ROOT / "model"
DEFAULT_JSON_INDENT = 2
DEBUG_DIR = REPO_ROOT / "debug"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Download an image, run OCR-assisted Lens font inference, and print JSON."
    )
  )
  parser.add_argument(
    "image_url",
    help="HTTP(S) URL for the input image.",
  )
  parser.add_argument(
    "--top-k",
    type=int,
    default=DEFAULT_TOP_K,
    help="Number of top font predictions to return.",
  )
  parser.add_argument(
    "--debug",
    action="store_true",
    help="Clear and rewrite debug images in the debug/ directory.",
  )
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  if args.top_k < 1:
    raise SystemExit("--top-k must be >= 1")

  model_path = resolve_model_path(model_root=DEFAULT_MODEL_DIR)

  result = run_inference_from_url(
    image_url=args.image_url,
    model_path=model_path,
    top_k=args.top_k,
    debug_dir=DEBUG_DIR if args.debug else None,
  )
  print(json.dumps(result, indent=DEFAULT_JSON_INDENT))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
