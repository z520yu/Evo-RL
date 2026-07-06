#!/usr/bin/env python

"""Audit LeRobot Piper episodes without modifying the dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_dataset


ACTION_NAMES = [f"joint_{idx}" for idx in range(1, 7)] + ["gripper"]


def _stack(series: pd.Series) -> np.ndarray:
    return np.stack(series.to_numpy()).astype(np.float64, copy=False)


def _longest_true_run(values: np.ndarray) -> int:
    if not values.any():
        return 0
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    return int(np.max(changes[1::2] - changes[::2]))


def _quantile(values: pd.Series, q: float) -> float:
    return float(values.quantile(q))


def audit_dataset(root: Path) -> tuple[pd.DataFrame, dict]:
    tasks_path = root / "meta" / "tasks.parquet"
    tasks_df = pd.read_parquet(tasks_path)
    task_names = {int(row.task_index): str(index) for index, row in tasks_df.iterrows()}

    table = pa_dataset.dataset(root / "data", format="parquet").to_table(
        columns=[
            "episode_index",
            "frame_index",
            "timestamp",
            "task_index",
            "action",
            "observation.state",
        ]
    )
    frames = table.to_pandas()

    records: list[dict] = []
    for episode_index, episode in frames.groupby("episode_index", sort=True):
        episode = episode.sort_values("frame_index")
        action = _stack(episode["action"])
        state = _stack(episode["observation.state"])
        action_delta = np.abs(np.diff(action, axis=0))
        state_delta = np.abs(np.diff(state, axis=0))
        arm_step = action_delta[:, :6].max(axis=1)
        gripper_step = action_delta[:, 6]
        state_arm_step = state_delta[:, :6].max(axis=1)
        state_gripper_step = state_delta[:, 6]
        timestamps = episode["timestamp"].to_numpy(dtype=np.float64)
        frame_indices = episode["frame_index"].to_numpy(dtype=np.int64)
        stationary = np.max(action_delta, axis=1) < 0.05

        record = {
            "episode_index": int(episode_index),
            "task_index": int(episode["task_index"].iloc[0]),
            "task": task_names.get(int(episode["task_index"].iloc[0]), "unknown"),
            "frames": int(len(episode)),
            "duration_s": float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0,
            "frame_gap_count": int(np.count_nonzero(np.diff(frame_indices) != 1)),
            "timestamp_bad_count": int(np.count_nonzero(np.diff(timestamps) <= 0)),
            "timestamp_max_error_s": float(np.max(np.abs(np.diff(timestamps) - 1 / 30)))
            if len(timestamps) > 1
            else 0.0,
            "nonfinite_count": int(
                np.size(action)
                + np.size(state)
                - np.count_nonzero(np.isfinite(action))
                - np.count_nonzero(np.isfinite(state))
            ),
            "arm_max_delta": float(arm_step.max(initial=0)),
            "arm_p99_delta": float(np.quantile(arm_step, 0.99)) if len(arm_step) else 0.0,
            "arm_jump_gt_3_count": int(np.count_nonzero(arm_step > 3.0)),
            "arm_jump_gt_5_count": int(np.count_nonzero(arm_step > 5.0)),
            "gripper_max_delta": float(gripper_step.max(initial=0)),
            "gripper_p99_delta": float(np.quantile(gripper_step, 0.99)) if len(gripper_step) else 0.0,
            "gripper_jump_gt_10_count": int(np.count_nonzero(gripper_step > 10.0)),
            "state_arm_max_delta": float(state_arm_step.max(initial=0)),
            "state_gripper_max_delta": float(state_gripper_step.max(initial=0)),
            "arm_path_length": float(action_delta[:, :6].sum()) if len(action_delta) else 0.0,
            "gripper_path_length": float(gripper_step.sum()) if len(gripper_step) else 0.0,
            "stationary_fraction": float(stationary.mean()) if len(stationary) else 1.0,
            "longest_stationary_frames": _longest_true_run(stationary),
            "start_arm_norm": float(np.linalg.norm(action[0, :6])),
            "end_arm_norm": float(np.linalg.norm(action[-1, :6])),
        }
        for idx, name in enumerate(ACTION_NAMES):
            record[f"{name}_min"] = float(action[:, idx].min())
            record[f"{name}_max"] = float(action[:, idx].max())
        records.append(record)

    audit = pd.DataFrame(records)

    duration_low = _quantile(audit["duration_s"], 0.01)
    duration_high = _quantile(audit["duration_s"], 0.99)
    arm_max_high = max(5.0, _quantile(audit["arm_max_delta"], 0.99))
    gripper_max_high = max(10.0, _quantile(audit["gripper_max_delta"], 0.99))
    stationary_high = _quantile(audit["stationary_fraction"], 0.99)
    path_high = _quantile(audit["arm_path_length"], 0.99)

    reasons: list[list[str]] = []
    for row in audit.itertuples(index=False):
        row_reasons: list[str] = []
        if row.nonfinite_count:
            row_reasons.append("nonfinite")
        if row.frame_gap_count:
            row_reasons.append("frame_gap")
        if row.timestamp_bad_count or row.timestamp_max_error_s > 0.002:
            row_reasons.append("timestamp")
        if row.duration_s < duration_low:
            row_reasons.append("very_short")
        if row.duration_s > duration_high:
            row_reasons.append("very_long")
        if row.arm_max_delta > arm_max_high:
            row_reasons.append("arm_jump")
        if row.gripper_max_delta > gripper_max_high:
            row_reasons.append("gripper_jump")
        if row.stationary_fraction > stationary_high:
            row_reasons.append("mostly_stationary")
        if row.arm_path_length > path_high:
            row_reasons.append("long_path")
        reasons.append(row_reasons)

    audit["flag_count"] = [len(value) for value in reasons]
    audit["flag_reasons"] = [",".join(value) for value in reasons]

    summary = {
        "dataset_root": str(root),
        "episodes": int(len(audit)),
        "frames": int(len(frames)),
        "tasks": {str(key): int(value) for key, value in audit["task_index"].value_counts().sort_index().items()},
        "thresholds": {
            "duration_low_s": duration_low,
            "duration_high_s": duration_high,
            "arm_max_delta": arm_max_high,
            "gripper_max_delta": gripper_max_high,
            "stationary_fraction": stationary_high,
            "arm_path_length": path_high,
        },
        "hard_integrity_failures": int(
            (
                (audit["nonfinite_count"] > 0)
                | (audit["frame_gap_count"] > 0)
                | (audit["timestamp_bad_count"] > 0)
            ).sum()
        ),
        "flagged_episodes": int((audit["flag_count"] > 0).sum()),
        "flagged_episode_indices": audit.loc[audit["flag_count"] > 0, "episode_index"].astype(int).tolist(),
        "metrics": {
            column: {
                "min": float(audit[column].min()),
                "median": float(audit[column].median()),
                "p95": _quantile(audit[column], 0.95),
                "p99": _quantile(audit[column], 0.99),
                "max": float(audit[column].max()),
            }
            for column in [
                "duration_s",
                "arm_max_delta",
                "gripper_max_delta",
                "arm_path_length",
                "stationary_fraction",
                "longest_stationary_frames",
            ]
        },
    }
    return audit, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    audit, summary = audit_dataset(args.dataset_root.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_dir / "episode_audit.csv", index=False)
    flagged = audit[audit["flag_count"] > 0].sort_values(
        ["flag_count", "arm_max_delta", "gripper_max_delta"], ascending=False
    )
    flagged.to_csv(args.output_dir / "flagged_episodes.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if not flagged.empty:
        print("\nFlagged episodes:")
        print(
            flagged[
                [
                    "episode_index",
                    "task_index",
                    "frames",
                    "duration_s",
                    "arm_max_delta",
                    "gripper_max_delta",
                    "stationary_fraction",
                    "flag_reasons",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
