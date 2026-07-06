#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from scripts import serve_policy

DEFAULT_PROMPT = (
    "Pick up the white box, place it in the center of the green rectangular tray, "
    "and orient the box so the barcode side faces upward and is clearly visible"
)
DEFAULT_CONFIG = "pi05_piper_white_box_barcode_green_tray_v1_start_trimmed_lora"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Piper white-box barcode OpenPI policy.")
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Checkpoint step directory containing params/ and assets/, for example .../29999.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--record", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if not (checkpoint_dir / "params").exists():
        raise FileNotFoundError(f"Missing params/ under checkpoint directory: {checkpoint_dir}")
    if not (checkpoint_dir / "assets").exists():
        raise FileNotFoundError(f"Missing assets/ under checkpoint directory: {checkpoint_dir}")

    serve_policy.main(
        serve_policy.Args(
            default_prompt=args.prompt,
            port=args.port,
            record=args.record,
            policy=serve_policy.Checkpoint(config=args.config, dir=str(checkpoint_dir)),
        )
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
