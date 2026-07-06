#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


VIDEO_KEYS = ("observation.images.front", "observation.images.wrist")
ARRAY_STAT_COLUMNS = ("action", "observation.state", "complementary_info.policy_action")
SCALAR_STAT_COLUMNS = (
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "complementary_info.is_intervention",
    "complementary_info.state",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trim static/preparation frames from LeRobot v3 episode starts.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--joint-threshold-deg", type=float, default=10.0)
    parser.add_argument("--buffer-frames", type=int, default=5)
    parser.add_argument("--review-seconds", type=float, default=4.0)
    parser.add_argument("--review-max-episodes", type=int, default=6)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_data(root: Path) -> pd.DataFrame:
    paths = sorted((root / "data").glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No data parquet found under {root / 'data'}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True).sort_values("index").reset_index(drop=True)


def load_episodes(root: Path) -> pd.DataFrame:
    paths = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata parquet found under {root / 'meta' / 'episodes'}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True).sort_values("episode_index").reset_index(drop=True)


def detect_trim_start(ep_rows: pd.DataFrame, threshold: float, buffer_frames: int) -> tuple[int, int, float]:
    states = np.stack(ep_rows["observation.state"].to_numpy()).astype(np.float32)
    actions = np.stack(ep_rows["action"].to_numpy()).astype(np.float32)
    score = np.max(np.abs(actions[:, :6] - states[0, :6]), axis=1)
    hits = np.flatnonzero(score > threshold)
    start_move = int(hits[0]) if len(hits) else 0
    trim_start = max(0, start_move - buffer_frames)
    return start_move, trim_start, float(score[start_move])


def load_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))


def write_info(input_root: Path, output_root: Path, total_frames: int, total_episodes: int) -> None:
    info = load_info(input_root)
    info["total_frames"] = int(total_frames)
    info["total_episodes"] = int(total_episodes)
    info["splits"] = {"train": f"0:{total_episodes}"}
    for key in VIDEO_KEYS:
        if key in info["features"]:
            info["features"][key]["info"]["video.codec"] = "h264"
            info["features"][key]["info"]["video.pix_fmt"] = "yuv420p"
    (output_root / "meta").mkdir(parents=True, exist_ok=True)
    (output_root / "meta" / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=4), encoding="utf-8")


def as_array(values: pd.Series) -> np.ndarray:
    first = values.iloc[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        return np.stack(values.to_numpy()).astype(np.float64)
    return values.to_numpy(dtype=np.float64)[:, None]


def stats_for_array(array: np.ndarray) -> dict:
    return {
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
        "count": [int(array.shape[0])],
        "q01": np.quantile(array, 0.01, axis=0).tolist(),
        "q10": np.quantile(array, 0.10, axis=0).tolist(),
        "q50": np.quantile(array, 0.50, axis=0).tolist(),
        "q90": np.quantile(array, 0.90, axis=0).tolist(),
        "q99": np.quantile(array, 0.99, axis=0).tolist(),
    }


def write_stats(input_root: Path, output_root: Path, data: pd.DataFrame) -> None:
    old_stats = json.loads((input_root / "meta" / "stats.json").read_text(encoding="utf-8"))
    new_stats = dict(old_stats)
    for column in ARRAY_STAT_COLUMNS + SCALAR_STAT_COLUMNS:
        if column in data.columns:
            new_stats[column] = stats_for_array(as_array(data[column]))
    (output_root / "meta" / "stats.json").write_text(json.dumps(new_stats, ensure_ascii=False, indent=4), encoding="utf-8")


def build_trimmed_tables(
    *,
    input_root: Path,
    output_root: Path,
    threshold: float,
    buffer_frames: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    info = load_info(input_root)
    fps = int(info["fps"])
    data = load_data(input_root)
    episodes = load_episodes(input_root)
    episode_meta = {int(row["episode_index"]): row.to_dict() for _, row in episodes.iterrows()}

    trimmed_parts: list[pd.DataFrame] = []
    new_episode_rows: list[dict] = []
    report: list[dict] = []
    next_index = 0
    next_video_time = 0.0

    for ep_idx, ep_rows in data.groupby("episode_index", sort=True):
        ep_idx = int(ep_idx)
        ep_rows = ep_rows.sort_values("frame_index").reset_index(drop=True)
        old_len = int(len(ep_rows))
        start_move, trim_start, score = detect_trim_start(ep_rows, threshold, buffer_frames)
        trimmed = ep_rows.iloc[trim_start:].copy().reset_index(drop=True)
        new_len = int(len(trimmed))
        if new_len <= 0:
            raise ValueError(f"Episode {ep_idx} became empty after trim_start={trim_start}")

        trimmed["frame_index"] = np.arange(new_len, dtype=np.int64)
        trimmed["timestamp"] = trimmed["frame_index"].to_numpy(dtype=np.float32) / float(fps)
        trimmed["index"] = np.arange(next_index, next_index + new_len, dtype=np.int64)
        trimmed_parts.append(trimmed)

        old_ep = dict(episode_meta[ep_idx])
        old_data_file_index = int(old_ep["data/file_index"])
        old_segment_duration = new_len / float(fps)
        old_video = {}
        for video_key in VIDEO_KEYS:
            old_video_file_index = int(old_ep[f"videos/{video_key}/file_index"])
            old_video_from = float(old_ep[f"videos/{video_key}/from_timestamp"])
            old_video[video_key] = {
                "file_index": old_video_file_index,
                "from_timestamp": old_video_from,
                "segment_start": old_video_from + trim_start / float(fps),
                "segment_duration": old_segment_duration,
            }

        new_ep = dict(old_ep)
        new_ep["length"] = new_len
        new_ep["data/chunk_index"] = 0
        new_ep["data/file_index"] = 0
        new_ep["dataset_from_index"] = next_index
        new_ep["dataset_to_index"] = next_index + new_len
        new_ep["meta/episodes/chunk_index"] = 0
        new_ep["meta/episodes/file_index"] = 0
        for video_key in VIDEO_KEYS:
            new_ep[f"videos/{video_key}/chunk_index"] = 0
            new_ep[f"videos/{video_key}/file_index"] = 0
            new_ep[f"videos/{video_key}/from_timestamp"] = next_video_time
            new_ep[f"videos/{video_key}/to_timestamp"] = next_video_time + old_segment_duration
        new_episode_rows.append(new_ep)

        report.append(
            {
                "episode_index": ep_idx,
                "old_len": old_len,
                "new_len": new_len,
                "trim_start": trim_start,
                "start_move": start_move,
                "score_at_start_move": score,
                "old_data_file_index": old_data_file_index,
                "old_segment_duration": old_segment_duration,
                "old_video": old_video,
                "new_video_from_timestamp": next_video_time,
            }
        )

        next_index += new_len
        next_video_time += old_segment_duration

    new_data = pd.concat(trimmed_parts, ignore_index=True)
    new_episodes = pd.DataFrame(new_episode_rows)

    (output_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    new_data.to_parquet(output_root / "data" / "chunk-000" / "file-000.parquet", index=False)
    new_episodes.to_parquet(output_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)
    shutil.copy2(input_root / "meta" / "tasks.parquet", output_root / "meta" / "tasks.parquet")
    write_info(input_root, output_root, len(new_data), len(new_episodes))
    write_stats(input_root, output_root, new_data)

    return new_data, new_episodes, report


def encode_video(
    *,
    input_root: Path,
    output_path: Path,
    video_key: str,
    report: list[dict],
    crf: int,
    preset: str,
    fps: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_indices = sorted({int(row["old_video"][video_key]["file_index"]) for row in report})
    input_paths = [
        input_root / "videos" / video_key / "chunk-000" / f"file-{file_index:03d}.mp4" for file_index in file_indices
    ]
    input_pos = {file_index: idx for idx, file_index in enumerate(file_indices)}
    cmd = ["ffmpeg", "-y"]
    for path in input_paths:
        cmd.extend(["-i", str(path)])

    filters: list[str] = []
    labels: list[str] = []
    for seg_idx, row in enumerate(report):
        old_video = row["old_video"][video_key]
        inp = input_pos[int(old_video["file_index"])]
        label = f"v{seg_idx}"
        filters.append(
            f"[{inp}:v]trim=start={old_video['segment_start']:.6f}:duration={old_video['segment_duration']:.6f},"
            f"setpts=PTS-STARTPTS[{label}]"
        )
        labels.append(f"[{label}]")
    filter_complex = ";".join(filters + [f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0,fps={fps}[outv]"])
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[outv]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
    )
    run(cmd)


def choose_review_episodes(report: list[dict], max_episodes: int) -> list[dict]:
    selected: dict[int, dict] = {}
    for ep_idx in (0, 23):
        for row in report:
            if row["episode_index"] == ep_idx:
                selected[ep_idx] = row
                break
    for row in sorted(report, key=lambda item: item["trim_start"], reverse=True):
        selected.setdefault(int(row["episode_index"]), row)
        if len(selected) >= max_episodes:
            break
    return list(selected.values())


def encode_review_video(
    *,
    input_root: Path,
    output_path: Path,
    report_rows: list[dict],
    mode: str,
    seconds: float,
    crf: int,
    preset: str,
    fps: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_indices_by_key = {
        video_key: sorted({int(row["old_video"][video_key]["file_index"]) for row in report_rows})
        for video_key in VIDEO_KEYS
    }
    inputs = []
    for video_key in VIDEO_KEYS:
        for file_index in file_indices_by_key[video_key]:
            inputs.append(input_root / "videos" / video_key / "chunk-000" / f"file-{file_index:03d}.mp4")
    cmd = ["ffmpeg", "-y"]
    for path in inputs:
        cmd.extend(["-i", str(path)])

    input_pos = {}
    next_input = 0
    for video_key in VIDEO_KEYS:
        for file_index in file_indices_by_key[video_key]:
            input_pos[(video_key, file_index)] = next_input
            next_input += 1

    filters: list[str] = []
    labels: list[str] = []
    for seg_idx, row in enumerate(report_rows):
        if mode == "before":
            duration = min(seconds, float(row["old_len"]) / float(fps))
            title = f"ep {row['episode_index']} before trim={row['trim_start']}"
        elif mode == "after":
            duration = min(seconds, float(row["new_len"]) / float(fps))
            title = f"ep {row['episode_index']} after trim={row['trim_start']}"
        else:
            raise ValueError(f"Unknown review mode: {mode}")

        starts = {}
        inputs_for_row = {}
        for video_key in VIDEO_KEYS:
            old_video = row["old_video"][video_key]
            starts[video_key] = (
                float(old_video["from_timestamp"]) if mode == "before" else float(old_video["segment_start"])
            )
            inputs_for_row[video_key] = input_pos[(video_key, int(old_video["file_index"]))]

        front_in = inputs_for_row[VIDEO_KEYS[0]]
        wrist_in = inputs_for_row[VIDEO_KEYS[1]]
        filters.extend(
            [
                f"[{front_in}:v]trim=start={starts[VIDEO_KEYS[0]]:.6f}:duration={duration:.6f},"
                f"setpts=PTS-STARTPTS,scale=640:480[f{seg_idx}]",
                f"[{wrist_in}:v]trim=start={starts[VIDEO_KEYS[1]]:.6f}:duration={duration:.6f},"
                f"setpts=PTS-STARTPTS,scale=640:480[w{seg_idx}]",
                f"[f{seg_idx}][w{seg_idx}]hstack=inputs=2,"
                f"drawtext=text='{title}':x=20:y=20:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.55[v{seg_idx}]",
            ]
        )
        labels.append(f"[v{seg_idx}]")
    filter_complex = ";".join(filters + [f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0,fps={fps}[outv]"])
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[outv]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
    )
    run(cmd)


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root exists: {output_root}. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    info = load_info(input_root)
    fps = int(info["fps"])
    new_data, _new_episodes, report = build_trimmed_tables(
        input_root=input_root,
        output_root=output_root,
        threshold=args.joint_threshold_deg,
        buffer_frames=args.buffer_frames,
    )
    report_path = output_root / "trim_report.json"
    report_payload = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "joint_threshold_deg": args.joint_threshold_deg,
        "buffer_frames": args.buffer_frames,
        "old_total_frames": int(load_data(input_root).shape[0]),
        "new_total_frames": int(new_data.shape[0]),
        "trimmed_frames": int(sum(row["trim_start"] for row in report)),
        "episodes": report,
    }
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote trim report: {report_path}", flush=True)

    for video_key in VIDEO_KEYS:
        encode_video(
            input_root=input_root,
            output_path=output_root / "videos" / video_key / "chunk-000" / "file-000.mp4",
            video_key=video_key,
            report=report,
            crf=args.crf,
            preset=args.preset,
            fps=fps,
        )

    review_rows = choose_review_episodes(report, args.review_max_episodes)
    review_dir = output_root / "review"
    encode_review_video(
        input_root=input_root,
        output_path=review_dir / "before_front_wrist_review.mp4",
        report_rows=review_rows,
        mode="before",
        seconds=args.review_seconds,
        crf=args.crf,
        preset=args.preset,
        fps=fps,
    )
    encode_review_video(
        input_root=input_root,
        output_path=review_dir / "after_front_wrist_review.mp4",
        report_rows=review_rows,
        mode="after",
        seconds=args.review_seconds,
        crf=args.crf,
        preset=args.preset,
        fps=fps,
    )
    print(f"Output dataset: {output_root}", flush=True)
    print(f"Review videos: {review_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
