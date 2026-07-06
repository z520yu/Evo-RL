#!/usr/bin/env python

"""Create contact sheets for selected LeRobot dataset episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_indices(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--episode-indices", required=True)
    parser.add_argument("--video-key", default="observation.images.front")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, video_backend="pyav")
    dataset.tolerance_s = 0.02
    task_df = pd.read_parquet(args.dataset_root / "meta" / "tasks.parquet")
    task_names = {int(row.task_index): str(index) for index, row in task_df.iterrows()}
    indices = parse_indices(args.episode_indices)

    thumb_width = 320
    thumb_height = 240
    label_height = 42
    samples_per_episode = 4
    rows: list[Image.Image] = []

    for episode_index in indices:
        episode = dataset.meta.episodes[episode_index]
        duration = float(episode[f"videos/{args.video_key}/to_timestamp"]) - float(
            episode[f"videos/{args.video_key}/from_timestamp"]
        )
        timestamps = [duration * fraction for fraction in (0.05, 0.35, 0.65, 0.95)]
        frames = dataset._query_videos({args.video_key: timestamps}, episode_index)[args.video_key]

        row = Image.new("RGB", (thumb_width * samples_per_episode, thumb_height + label_height), "white")
        draw = ImageDraw.Draw(row)
        task_index = int(dataset.hf_dataset[dataset.meta.episodes[episode_index]["dataset_from_index"]]["task_index"])
        task = task_names.get(task_index, "unknown")
        draw.text(
            (6, 4),
            f"episode {episode_index} | task {task_index} | {duration:.1f}s | {task[:80]}",
            fill="black",
        )
        for sample_index, frame in enumerate(frames):
            image = Image.fromarray((frame.permute(1, 2, 0).numpy() * 255).astype("uint8"))
            image = image.resize((thumb_width, thumb_height))
            row.paste(image, (sample_index * thumb_width, label_height))
            draw.text((sample_index * thumb_width + 5, label_height + 5), f"{timestamps[sample_index]:.1f}s", fill="red")
        rows.append(row)

    sheet = Image.new("RGB", (thumb_width * samples_per_episode, (thumb_height + label_height) * len(rows)), "white")
    for row_index, row in enumerate(rows):
        sheet.paste(row, (0, row_index * (thumb_height + label_height)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
