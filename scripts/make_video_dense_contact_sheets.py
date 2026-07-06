#!/usr/bin/env python

"""Create dense contact sheets for exported episode videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for video_path in sorted(args.input_dir.glob("*.mp4")):
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        frames = [frame.to_image() for frame in container.decode(stream)]
        container.close()
        if not frames:
            continue

        indices = np.linspace(0, len(frames) - 1, args.samples).round().astype(int)
        columns = 3
        rows = int(np.ceil(args.samples / columns))
        width = 640
        height = 240
        label_height = 24
        sheet = Image.new("RGB", (width * columns, (height + label_height) * rows), "white")
        draw = ImageDraw.Draw(sheet)
        fps = float(stream.average_rate) if stream.average_rate else 30.0

        for slot, frame_index in enumerate(indices):
            frame = frames[int(frame_index)].resize((width, height))
            x = (slot % columns) * width
            y = (slot // columns) * (height + label_height)
            sheet.paste(frame, (x, y + label_height))
            draw.text((x + 5, y + 4), f"{frame_index / fps:.1f}s", fill="black")

        output_path = args.output_dir / f"{video_path.stem}.jpg"
        sheet.save(output_path, quality=92)
        print(output_path)


if __name__ == "__main__":
    main()
