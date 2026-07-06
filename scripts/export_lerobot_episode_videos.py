#!/usr/bin/env python

"""Export selected LeRobot episodes as side-by-side front and wrist videos."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


def parse_indices(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def video_path(root: Path, row: pd.Series, key: str) -> Path:
    chunk_index = int(row[f"videos/{key}/chunk_index"])
    file_index = int(row[f"videos/{key}/file_index"])
    return root / "videos" / key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode-indices", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = pd.concat(
        [pd.read_parquet(path) for path in sorted((root / "meta" / "episodes").rglob("*.parquet"))],
        ignore_index=True,
    ).set_index("episode_index")

    for episode_index in parse_indices(args.episode_indices):
        row = episodes.loc[episode_index]
        front_key = "observation.images.front"
        wrist_key = "observation.images.wrist"
        front_path = video_path(root, row, front_key)
        wrist_path = video_path(root, row, wrist_key)
        front_start = float(row[f"videos/{front_key}/from_timestamp"])
        wrist_start = float(row[f"videos/{wrist_key}/from_timestamp"])
        front_duration = float(row[f"videos/{front_key}/to_timestamp"]) - front_start
        wrist_duration = float(row[f"videos/{wrist_key}/to_timestamp"]) - wrist_start
        duration = min(front_duration, wrist_duration)
        output_path = args.output_dir / f"episode_{episode_index:03d}_front_wrist.mp4"

        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{front_start:.6f}",
            "-t",
            f"{duration:.6f}",
            "-i",
            str(front_path),
            "-ss",
            f"{wrist_start:.6f}",
            "-t",
            f"{duration:.6f}",
            "-i",
            str(wrist_path),
            "-filter_complex",
            (
                "[0:v]setpts=PTS-STARTPTS,scale=640:480,"
                "drawtext=text='front':x=10:y=10:fontsize=24:fontcolor=red:box=1:boxcolor=white@0.6[f];"
                "[1:v]setpts=PTS-STARTPTS,scale=640:480,"
                "drawtext=text='wrist':x=10:y=10:fontsize=24:fontcolor=red:box=1:boxcolor=white@0.6[w];"
                "[f][w]hstack=inputs=2[v]"
            ),
            "-map",
            "[v]",
            "-an",
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(output_path)


if __name__ == "__main__":
    main()
