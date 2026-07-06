#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_CLIENT_SRC = EVO_ROOT / "openpi-lotus-v2-workspace" / "packages" / "openpi-client" / "src"
LEROBOT_SRC = EVO_ROOT / "src"
for path in (OPENPI_CLIENT_SRC, LEROBOT_SRC):
    if path.exists():
        sys.path.insert(0, str(path))

from lerobot.datasets.video_utils import decode_video_frames
from openpi_client import websocket_client_policy


TASK_WHITE_BOX_BARCODE = (
    "Pick up the white box, place it in the center of the green rectangular tray, "
    "and orient the box so the barcode side faces upward and is clearly visible"
)
DIM_NAMES_7D = [
    "joint_1.pos",
    "joint_2.pos",
    "joint_3.pos",
    "joint_4.pos",
    "joint_5.pos",
    "joint_6.pos",
    "gripper.pos",
]


@dataclass(frozen=True)
class EpisodeSample:
    dataset_root: Path
    episode_index: int
    frame_id: int
    task: str
    row_index: int
    episode: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline OpenPI eval for Piper LeRobot v3 datasets.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--lerobot-dataset-path",
        default=str(EVO_ROOT / "piper_white_box_barcode_up_green_tray_dual_rs_v1_cleaned"),
    )
    parser.add_argument("--episode-id", type=int, default=None)
    parser.add_argument("--frame-id", type=int, default=0)
    parser.add_argument("--random-frame", action="store_true")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--chunk-num", type=int, default=1)
    parser.add_argument("--chunk-stride", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument(
        "--take-actions",
        type=int,
        default=None,
        help="Only keep the first N actions from each policy chunk, useful for matching --query-every.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--output-dir", default=str(EVO_ROOT / "outputs" / "openpi_offline_eval"))
    return parser.parse_args()


def load_tasks(dataset_root: Path) -> dict[int, str]:
    tasks_df = pd.read_parquet(dataset_root / "meta" / "tasks.parquet")
    tasks: dict[int, str] = {}
    for idx, row in tasks_df.iterrows():
        task_index = int(row["task_index"])
        task = row["task"] if "task" in tasks_df.columns else idx
        tasks[task_index] = str(task)
    return tasks


def load_episode_table(dataset_root: Path) -> pd.DataFrame:
    paths = sorted((dataset_root / "meta" / "episodes").glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata parquet found under {dataset_root / 'meta' / 'episodes'}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True).sort_values("episode_index")


def load_data_table(dataset_root: Path) -> pd.DataFrame:
    paths = sorted((dataset_root / "data").glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No data parquet found under {dataset_root / 'data'}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True).sort_values("index").reset_index(drop=True)


def resolve_sample(
    dataset_root: Path,
    episodes: pd.DataFrame,
    data: pd.DataFrame,
    tasks: dict[int, str],
    args: argparse.Namespace,
) -> EpisodeSample:
    rng = random.Random(args.seed)
    if args.episode_id is None:
        episode = episodes.iloc[rng.randrange(len(episodes))].to_dict()
    else:
        matches = episodes[episodes["episode_index"] == args.episode_id]
        if len(matches) == 0:
            raise ValueError(f"episode-id {args.episode_id} not found")
        episode = matches.iloc[0].to_dict()

    ep_idx = int(episode["episode_index"])
    ep_rows = data[data["episode_index"] == ep_idx].sort_values("frame_index")
    if len(ep_rows) == 0:
        raise ValueError(f"No rows found for episode {ep_idx}")

    if args.random_frame:
        frame_id = int(rng.randrange(max(1, len(ep_rows) - 1)))
    else:
        frame_id = int(args.frame_id)
    if not (0 <= frame_id < len(ep_rows)):
        raise ValueError(f"frame-id {frame_id} out of range for episode {ep_idx}; valid [0, {len(ep_rows) - 1}]")

    row = ep_rows[ep_rows["frame_index"] == frame_id]
    if len(row) == 0:
        raise ValueError(f"frame-id {frame_id} missing in episode {ep_idx}")
    row_index = int(row.index[0])

    if args.instruction is not None:
        task = args.instruction
    elif int(data.iloc[row_index]["task_index"]) in tasks:
        task = tasks[int(data.iloc[row_index]["task_index"])]
    else:
        task = TASK_WHITE_BOX_BARCODE

    return EpisodeSample(
        dataset_root=dataset_root,
        episode_index=ep_idx,
        frame_id=frame_id,
        task=task,
        row_index=row_index,
        episode=episode,
    )


def video_path_for(dataset_root: Path, info: dict, episode: dict, video_key: str) -> Path:
    return dataset_root / info["video_path"].format(
        video_key=video_key,
        chunk_index=int(episode[f"videos/{video_key}/chunk_index"]),
        file_index=int(episode[f"videos/{video_key}/file_index"]),
    )


def load_video_frame(
    dataset_root: Path,
    info: dict,
    episode: dict,
    row: pd.Series,
    video_key: str,
    backend: str,
) -> np.ndarray:
    video_path = video_path_for(dataset_root, info, episode, video_key)
    timestamp = float(episode[f"videos/{video_key}/from_timestamp"]) + float(row["timestamp"])
    fps = float(info["fps"])
    frame = decode_video_frames(video_path, [timestamp], tolerance_s=(1.0 / fps) + 1e-3, backend=backend).squeeze(0)
    chw = np.clip(frame.numpy() * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(chw)


def chw_to_hwc(chw: np.ndarray) -> np.ndarray:
    return np.moveaxis(np.asarray(chw), 0, -1)


def action_sequence(data: pd.DataFrame, row: pd.Series, episode: dict, horizon: int) -> np.ndarray:
    abs_idx = int(row["index"])
    ep_start = int(episode["dataset_from_index"])
    ep_end = int(episode["dataset_to_index"])
    by_abs = data.set_index("index", drop=False)
    indices = [max(ep_start, min(ep_end - 1, abs_idx + t)) for t in range(horizon)]
    return np.stack([np.asarray(by_abs.loc[idx]["action"], dtype=np.float32) for idx in indices], axis=0)


def infer_policy(
    client: websocket_client_policy.WebsocketClientPolicy,
    *,
    front_chw: np.ndarray,
    wrist_chw: np.ndarray,
    state: np.ndarray,
    prompt: str,
) -> tuple[np.ndarray, float]:
    payload = {
        "images": {
            "cam_high": front_chw,
            "cam_wrist": wrist_chw,
        },
        "state": state.astype(np.float32),
        "prompt": prompt,
    }
    start = time.time()
    result = client.infer(payload)
    return np.asarray(result["actions"], dtype=np.float32), time.time() - start


def plot_result(
    *,
    output_path: Path,
    sample: EpisodeSample,
    front_chw: np.ndarray,
    wrist_chw: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    state0: np.ndarray,
    chunk_boundaries: Sequence[int],
    summary: dict,
) -> None:
    fig = plt.figure(figsize=(18, 22))
    gs = fig.add_gridspec(8, 6, height_ratios=[1.4, 1, 1, 1, 1, 1, 1, 1])
    fig.suptitle(
        (
            f"episode={sample.episode_index} frame={sample.frame_id} "
            f"MAE={summary['mae']:.3f} RMSE={summary['rmse']:.3f}\n{sample.task}"
        ),
        fontsize=13,
    )

    for idx, (title, image) in enumerate((("front", front_chw), ("wrist", wrist_chw))):
        ax = fig.add_subplot(gs[0, idx * 3 : (idx + 1) * 3])
        ax.imshow(chw_to_hwc(image))
        ax.set_title(title)
        ax.axis("off")

    t = np.arange(len(gt))
    for dim_idx, name in enumerate(DIM_NAMES_7D):
        ax = fig.add_subplot(gs[dim_idx + 1, :])
        ax.plot(t, gt[:, dim_idx], "b-", label="GT action", linewidth=1.8, alpha=0.85)
        ax.plot(t, pred[:, dim_idx], "r-", label="Pred action", linewidth=1.6, alpha=0.85)
        ax.axhline(float(state0[dim_idx]), color="gray", linestyle=":", linewidth=1.0, label="Current state")
        for boundary in chunk_boundaries[1:-1]:
            ax.axvline(boundary, color="k", linestyle="--", linewidth=0.8, alpha=0.35)
        ax.set_title(name)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = parse_args()
    if args.chunk_num <= 0:
        raise ValueError("--chunk-num must be >= 1")

    dataset_root = Path(args.lerobot_dataset_path).expanduser().resolve()
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    tasks = load_tasks(dataset_root)
    episodes = load_episode_table(dataset_root)
    data = load_data_table(dataset_root)
    sample = resolve_sample(dataset_root, episodes, data, tasks, args)

    logging.info("Connecting OpenPI policy server at %s:%d", args.host, args.port)
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    logging.info("Server metadata: %s", client.get_server_metadata())
    logging.info("Evaluating episode=%d frame=%d task=%s", sample.episode_index, sample.frame_id, sample.task)

    current_frame = int(sample.frame_id)
    chunk_stride = args.chunk_stride
    chunk_boundaries = [0]
    rtts: list[float] = []
    records: list[dict] = []
    gt_chunks: list[np.ndarray] = []
    pred_chunks: list[np.ndarray] = []
    first_front: np.ndarray | None = None
    first_wrist: np.ndarray | None = None
    first_state: np.ndarray | None = None

    episode_rows = data[data["episode_index"] == sample.episode_index].sort_values("frame_index")
    for chunk_idx in range(args.chunk_num):
        if current_frame >= len(episode_rows):
            logging.warning("Reached end of episode at frame %d; stopping", current_frame)
            break

        row_match = episode_rows[episode_rows["frame_index"] == current_frame]
        if len(row_match) == 0:
            logging.warning("Missing frame %d in episode %d; stopping", current_frame, sample.episode_index)
            break
        row = row_match.iloc[0]
        state = np.asarray(row["observation.state"], dtype=np.float32)
        front = load_video_frame(dataset_root, info, sample.episode, row, "observation.images.front", args.video_backend)
        wrist = load_video_frame(dataset_root, info, sample.episode, row, "observation.images.wrist", args.video_backend)
        pred, rtt = infer_policy(client, front_chw=front, wrist_chw=wrist, state=state, prompt=sample.task)
        if pred.ndim != 2 or pred.shape[1] < 7:
            raise ValueError(f"Expected actions shape (T, >=7), got {pred.shape}")
        pred = pred[:, :7]
        horizon = int(args.chunk_size or pred.shape[0])
        gt = action_sequence(data, row, sample.episode, horizon)
        n = min(len(pred), len(gt), horizon)
        if args.take_actions is not None:
            if args.take_actions <= 0:
                raise ValueError("--take-actions must be positive")
            n = min(n, int(args.take_actions))
        pred = pred[:n]
        gt = gt[:n]

        if first_front is None:
            first_front = front
            first_wrist = wrist
            first_state = state

        rtts.append(rtt)
        gt_chunks.append(gt)
        pred_chunks.append(pred)
        chunk_boundaries.append(chunk_boundaries[-1] + n)
        records.append(
            {
                "chunk_index": chunk_idx,
                "frame_id": current_frame,
                "rtt_sec": rtt,
                "pred_action_shape": list(pred.shape),
                "state": state.tolist(),
                "pred_first": pred[0].tolist(),
                "gt_first": gt[0].tolist(),
                "pred_first_minus_state": (pred[0] - state).tolist(),
                "gt_first_minus_state": (gt[0] - state).tolist(),
            }
        )
        logging.info(
            "chunk=%03d frame=%d rtt=%.3fs pred_first_minus_state=%s gt_first_minus_state=%s",
            chunk_idx,
            current_frame,
            rtt,
            np.round(pred[0] - state, 3).tolist(),
            np.round(gt[0] - state, 3).tolist(),
        )

        if chunk_stride is None:
            chunk_stride = n
        current_frame += int(chunk_stride)

    if not gt_chunks:
        raise RuntimeError("No chunks evaluated")

    gt_all = np.concatenate(gt_chunks, axis=0)
    pred_all = np.concatenate(pred_chunks, axis=0)
    error = pred_all - gt_all
    summary = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae_by_dim": np.mean(np.abs(error), axis=0).tolist(),
        "rmse_by_dim": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "mean_rtt_sec": float(np.mean(rtts)),
        "mean_abs_pred_minus_gt_first": float(np.mean(np.abs(pred_all[0] - gt_all[0]))),
        "mean_abs_pred_step_motion": float(np.mean(np.abs(np.diff(pred_all, axis=0)))) if len(pred_all) > 1 else 0.0,
        "mean_abs_gt_step_motion": float(np.mean(np.abs(np.diff(gt_all, axis=0)))) if len(gt_all) > 1 else 0.0,
    }

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{sample.episode_index:06d}_frame_{sample.frame_id:06d}_chunks_{len(records):03d}"
    plot_path = output_dir / f"{stem}.png"
    json_path = output_dir / f"{stem}.json"
    plot_result(
        output_path=plot_path,
        sample=sample,
        front_chw=first_front if first_front is not None else np.zeros((3, 480, 640), dtype=np.uint8),
        wrist_chw=first_wrist if first_wrist is not None else np.zeros((3, 480, 640), dtype=np.uint8),
        gt=gt_all,
        pred=pred_all,
        state0=first_state if first_state is not None else np.zeros(7, dtype=np.float32),
        chunk_boundaries=chunk_boundaries,
        summary=summary,
    )

    payload = {
        "host": args.host,
        "port": args.port,
        "dataset_root": str(dataset_root),
        "episode_index": sample.episode_index,
        "start_frame_id": sample.frame_id,
        "task": sample.task,
        "chunk_num_requested": args.chunk_num,
        "chunk_num_completed": len(records),
        "chunk_stride": int(chunk_stride if chunk_stride is not None else 0),
        "take_actions": args.take_actions,
        "summary": summary,
        "chunk_records": records,
        "gt_actions": gt_all.tolist(),
        "pred_actions": pred_all.tolist(),
        "chunk_boundaries": chunk_boundaries,
        "plot_path": str(plot_path),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("summary=%s", json.dumps(summary, ensure_ascii=False))
    logging.info("Saved plot to %s", plot_path)
    logging.info("Saved raw outputs to %s", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
