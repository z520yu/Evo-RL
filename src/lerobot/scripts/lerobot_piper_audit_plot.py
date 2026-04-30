#!/usr/bin/env python

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _get_nested(obj: dict[str, Any], dotted_key: str) -> Any:
    if dotted_key in obj:
        cur: Any = obj[dotted_key]
    else:
        parts = dotted_key.split(".")
        cur = obj
        idx = 0
        while idx < len(parts):
            if not isinstance(cur, dict):
                return np.nan
            matched = False
            for end_idx in range(len(parts), idx, -1):
                key = ".".join(parts[idx:end_idx])
                if key in cur:
                    cur = cur[key]
                    idx = end_idx
                    matched = True
                    break
            if not matched:
                return np.nan
    if isinstance(cur, bool):
        return float(cur)
    if isinstance(cur, (int, float)):
        return float(cur)
    return np.nan


def _series(rows: list[dict[str, Any]], dotted_key: str, scale: float = 1.0) -> np.ndarray:
    return np.asarray([_get_nested(row, dotted_key) * scale for row in rows], dtype=float)


def _load_rows(run_dir: Path) -> list[dict[str, Any]]:
    steps_path = run_dir / "audit_steps.jsonl"
    rows: list[dict[str, Any]] = []
    with steps_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {steps_path}")
    return rows


def _finish(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_gripper(rows: list[dict[str, Any]], t: np.ndarray, out_dir: Path) -> None:
    grip_obs = _series(rows, "raw_observation.gripper.pos")
    grip_cmd = _series(rows, "sent_action.gripper.pos")
    grip_sdk = _series(rows, "piper_sdk.GetArmGripperMsgs.gripper_state.grippers_angle", 1e-3)
    grip_effort = _series(rows, "piper_sdk.GetArmGripperMsgs.gripper_state.grippers_effort", 1e-3)
    grip_err = _series(rows, "derived.target_minus_obs.gripper.pos")
    grip_vel = _series(rows, "derived.obs_velocity_per_s.gripper.pos")

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(t, grip_obs, label="raw_observation gripper.pos")
    axes[0].plot(t, grip_cmd, label="sent_action gripper.pos", alpha=0.8)
    axes[0].plot(t, grip_sdk, label="SDK grippers_angle / 1000", alpha=0.7, linestyle="--")
    axes[0].set_ylabel("stroke (mm)")
    axes[0].legend(loc="upper right", ncols=3, fontsize=8)

    axes[1].plot(t, grip_effort, color="tab:red")
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    axes[1].set_ylabel("effort proxy (N*m)")

    axes[2].plot(t, grip_err, color="tab:purple")
    axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    axes[2].set_ylabel("target - obs (mm)")

    axes[3].plot(t, grip_vel, color="tab:green")
    axes[3].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    axes[3].set_ylabel("velocity (mm/s)")
    axes[3].set_xlabel("time (s)")

    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Gripper stroke, effort, tracking residual, and realized velocity", y=0.995)
    _finish(fig, out_dir / "01_gripper_contact_proxy.png")


def _plot_motor_currents(rows: list[dict[str, Any]], t: np.ndarray, out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    axes = axes.ravel()
    for idx in range(1, 7):
        ax = axes[idx - 1]
        current_a = _series(rows, f"piper_sdk.GetArmHighSpdInfoMsgs.motor_{idx}.current", 1e-3)
        speed = _series(rows, f"piper_sdk.GetArmHighSpdInfoMsgs.motor_{idx}.motor_speed", 1e-3)
        ax.plot(t, current_a, label="current (A)")
        ax2 = ax.twinx()
        ax2.plot(t, speed, color="tab:orange", alpha=0.35, label="speed (rad/s)")
        ax.set_title(f"motor_{idx}")
        ax.set_ylabel("current (A)")
        ax2.set_ylabel("speed (rad/s)")
        ax.grid(True, alpha=0.25)
    axes[-2].set_xlabel("time (s)")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Motor current with motor speed overlay", y=0.995)
    _finish(fig, out_dir / "02_motor_current_speed.png")


def _plot_motor_effort(rows: list[dict[str, Any]], t: np.ndarray, out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    axes = axes.ravel()
    for idx in range(1, 7):
        ax = axes[idx - 1]
        effort = _series(rows, f"piper_sdk.GetArmHighSpdInfoMsgs.motor_{idx}.effort", 1e-3)
        current_a = _series(rows, f"piper_sdk.GetArmHighSpdInfoMsgs.motor_{idx}.current", 1e-3)
        ax.plot(t, effort, label="effort proxy")
        ax.plot(t, current_a, label="current (A)", alpha=0.45, linestyle="--")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_title(f"motor_{idx}")
        ax.set_ylabel("proxy units")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    axes[-2].set_xlabel("time (s)")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("SDK effort proxy compared with current", y=0.995)
    _finish(fig, out_dir / "03_motor_effort_proxy.png")


def _plot_tracking(rows: list[dict[str, Any]], t: np.ndarray, out_dir: Path) -> None:
    joint_keys = [f"joint_{idx}.pos" for idx in range(1, 7)]
    err = np.vstack([_series(rows, f"derived.target_minus_obs.{key}") for key in joint_keys])
    vel = np.vstack([_series(rows, f"derived.obs_velocity_per_s.{key}") for key in joint_keys])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for idx, key in enumerate(joint_keys):
        axes[0].plot(t, err[idx], label=key)
        axes[1].plot(t, vel[idx], label=key)
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    axes[0].set_ylabel("target - obs (deg)")
    axes[1].set_ylabel("velocity (deg/s)")
    axes[1].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", ncols=6, fontsize=8)
    fig.suptitle("Joint tracking residual and realized velocity", y=0.995)
    _finish(fig, out_dir / "04_joint_tracking_velocity.png")


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-9:
        scale = np.nanstd(x)
    if not np.isfinite(scale) or scale < 1e-9:
        return np.zeros_like(x, dtype=float)
    return (x - med) / scale


def _plot_overview(rows: list[dict[str, Any]], t: np.ndarray, out_dir: Path) -> None:
    currents = np.vstack(
        [_series(rows, f"piper_sdk.GetArmHighSpdInfoMsgs.motor_{idx}.current", 1e-3) for idx in range(1, 7)]
    )
    current_load = np.sqrt(np.nanmean(np.square(np.vstack([_robust_z(c) for c in currents])), axis=0))

    joint_err = np.vstack([_series(rows, f"derived.target_minus_obs.joint_{idx}.pos") for idx in range(1, 7)])
    tracking_load = np.sqrt(np.nanmean(np.square(np.vstack([_robust_z(e) for e in joint_err])), axis=0))

    gripper_effort = np.abs(_robust_z(_series(rows, "piper_sdk.GetArmGripperMsgs.gripper_state.grippers_effort", 1e-3)))
    gripper_err = np.abs(_robust_z(_series(rows, "derived.target_minus_obs.gripper.pos")))
    gripper_vel = np.abs(_robust_z(_series(rows, "derived.obs_velocity_per_s.gripper.pos")))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, current_load, label="motor current load score")
    ax.plot(t, tracking_load, label="joint tracking residual score")
    ax.plot(t, gripper_effort, label="abs gripper effort score")
    ax.plot(t, gripper_err, label="abs gripper tracking score", alpha=0.85)
    ax.plot(t, gripper_vel, label="abs gripper velocity score", alpha=0.65)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("robust z score")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncols=2, fontsize=8)
    fig.suptitle("Normalized physical-evidence overview", y=0.995)
    _finish(fig, out_dir / "05_physical_evidence_overview.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Piper teleop audit low-level signals.")
    parser.add_argument("run_dir", type=Path, help="Path to an audit run directory containing audit_steps.jsonl.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory for PNG outputs. Defaults to run_dir/plots.")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser()
    out_dir = (args.out_dir or (run_dir / "plots")).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(run_dir)
    t = _series(rows, "timestamp_s")
    t = t - np.nanmin(t)

    _plot_gripper(rows, t, out_dir)
    _plot_motor_currents(rows, t, out_dir)
    _plot_motor_effort(rows, t, out_dir)
    _plot_tracking(rows, t, out_dir)
    _plot_overview(rows, t, out_dir)

    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
