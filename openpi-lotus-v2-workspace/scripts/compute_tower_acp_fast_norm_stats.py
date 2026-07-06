#!/usr/bin/env python3
"""Compute OpenPI norm stats for the tower ACP dataset without decoding videos."""

from __future__ import annotations

import argparse
import pathlib
import re

import numpy as np
import pyarrow.parquet as pq

from openpi import transforms
from openpi.policies import yam_policy
from openpi.shared import normalize


DATA_ROOT = pathlib.Path(
    "/home/zhangyu/datasets/posttraining_rfm_rss2026/datasets/"
    "Challenge-phase1-dataset/tower-of-hanoi-game/"
    "expert-hil-acp-pistar06-bs12-r30-data"
)
OUT_DIR = pathlib.Path(
    "/home/zhangyu/datasets/posttraining_rfm_rss2026/repos/openpi-baseline/assets/"
    "pi05_tower-of-hanoi-game_expert_hil_acp_tagged_from_expert_lora/"
    "tower-of-hanoi-game/expert-hil-acp-pistar06-bs12-r30-data"
)


def _episode_id(path: pathlib.Path) -> int:
    match = re.search(r"episode_(\d+)\.parquet$", path.name)
    if not match:
        raise ValueError(f"Cannot parse episode id from {path}")
    return int(match.group(1))


def _list_array_to_numpy(column) -> np.ndarray:
    arr = column.combine_chunks()
    values = np.asarray(arr.values.to_numpy(zero_copy_only=False), dtype=np.float32)
    return values.reshape(len(arr), -1)


def _load_episode(path: pathlib.Path) -> tuple[int, np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["observation.state", "action", "episode_index"])
    states = _list_array_to_numpy(table["observation.state"])
    actions = _list_array_to_numpy(table["action"])
    ep = int(np.asarray(table["episode_index"].combine_chunks())[0])
    return ep, states, actions


def _prepare_arrays(data_root: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    files = sorted((data_root / "data").rglob("*.parquet"), key=_episode_id)
    if not files:
        raise FileNotFoundError(f"No parquet files found under {data_root / 'data'}")

    episodes: list[tuple[int, np.ndarray, np.ndarray]] = []
    for i, path in enumerate(files, start=1):
        episodes.append(_load_episode(path))
        if i % 100 == 0:
            print(f"loaded {i}/{len(files)} parquet files", flush=True)

    episodes.sort(key=lambda x: x[0])
    states = np.concatenate([x[1] for x in episodes], axis=0)
    actions = np.concatenate([x[2] for x in episodes], axis=0)

    starts: list[int] = []
    ends: list[int] = []
    cursor = 0
    for _, ep_states, _ in episodes:
        starts.append(cursor)
        cursor += len(ep_states)
        ends.append(cursor)

    return states, actions, np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)


def _state_transform(states: np.ndarray, action_dim: int) -> np.ndarray:
    mask = yam_policy._yam_joint_flip_mask()
    decoded = states[..., : len(mask)].copy() * mask
    decoded[..., [6, 13]] = yam_policy._gripper_to_angular(decoded[..., [6, 13]])
    return transforms.pad_to_dim(decoded, action_dim)


def _action_transform(actions: np.ndarray, state: np.ndarray, action_dim: int) -> np.ndarray:
    mask = yam_policy._yam_joint_flip_mask()
    encoded = actions[..., : len(mask)].copy() * mask
    encoded[..., [6, 13]] = yam_policy._gripper_from_angular_inv(encoded[..., [6, 13]])
    encoded = transforms.pad_to_dim(encoded, action_dim)

    delta_mask = np.asarray(transforms.make_bool_mask(6, -1, 6, -1))
    dims = delta_mask.shape[-1]
    encoded[..., :dims] -= np.expand_dims(np.where(delta_mask, state[..., :dims], 0), axis=-2)
    return encoded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=pathlib.Path, default=DATA_ROOT)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--max-frames", type=int, default=65536)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    states_raw, actions_raw, ep_starts, ep_ends = _prepare_arrays(args.data_root)
    total = len(states_raw)
    n = min(args.max_frames, total)
    rng = np.random.default_rng(args.seed)
    sample_indices = rng.choice(total, size=n, replace=False)
    sample_indices.sort()
    print(f"total_frames={total} sampled_frames={n}", flush=True)

    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    deltas = np.arange(args.horizon, dtype=np.int64)

    for start in range(0, n, args.batch_size):
        idx = sample_indices[start : start + args.batch_size]
        ep_ids = np.searchsorted(ep_ends, idx, side="right")
        end = ep_ends[ep_ids]
        query = np.minimum(idx[:, None] + deltas[None, :], end[:, None] - 1)

        state_batch = _state_transform(states_raw[idx], args.action_dim)
        action_batch = _action_transform(actions_raw[query], state_batch, args.action_dim)
        state_stats.update(state_batch)
        action_stats.update(action_batch)

        if (start // args.batch_size + 1) % 10 == 0:
            print(f"processed {min(start + args.batch_size, n)}/{n}", flush=True)

    norm_stats = {
        "state": state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
    }
    normalize.save(args.out_dir, norm_stats)
    print(f"wrote {args.out_dir / 'norm_stats.json'}", flush=True)


if __name__ == "__main__":
    main()
