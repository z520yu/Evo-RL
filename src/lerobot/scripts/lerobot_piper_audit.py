#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Record Piper teleoperation with Evo-RL/LeRobot plus sidecar execution audit logs."""

import json
import logging
import math
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera.configuration_reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import RobotAction, RobotObservation, make_default_processors
from lerobot.robots import RobotConfig, make_robot_from_config, piper_follower  # noqa: F401
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config, piper_leader  # noqa: F401
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

PIPER_GETTERS = (
    "GetArmStatus",
    "GetArmJointMsgs",
    "GetArmGripperMsgs",
    "GetArmEndPoseMsgs",
    "GetArmLowSpdInfoMsgs",
    "GetArmHighSpdInfoMsgs",
    "GetArmEnableStatus",
    "GetMotorStates",
    "GetDriverStates",
    "GetCrashProtectionLevelFeedback",
    "GetArmJointCtrl",
    "GetArmGripperCtrl",
    "GetArmCtrlCode151",
    "GetArmModeCtrl",
    "GetCurrentEndVelAndAccParam",
    "GetCurrentMotorAngleLimitMaxVel",
    "GetCurrentMotorMaxAccLimit",
    "GetAllMotorAngleLimitMaxSpd",
    "GetAllMotorMaxAccLimit",
    "GetFK",
)


@dataclass
class PiperAuditDatasetConfig:
    repo_id: str = "local/piper_audit"
    single_task: str = "piper teleop audit"
    root: str | Path | None = "VLA_RL_papers/vla_rl_ideas/teleop_audit/datasets"
    fps: int = 20
    episode_time_s: float = 120.0
    reset_time_s: float = 0.0
    num_episodes: int = 1
    video: bool = True
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    vcodec: str = "h264"


@dataclass
class PiperAuditConfig:
    robot: RobotConfig
    teleop: TeleoperatorConfig
    dataset: PiperAuditDatasetConfig
    audit_dir: str | Path = "VLA_RL_papers/vla_rl_ideas/teleop_audit/runs"
    run_name: str | None = None
    sdk_snapshot: bool = True
    flush_every: int = 1
    display_data: bool = False
    display_compressed_images: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    communication_retry_timeout_s: float = 2.0
    communication_retry_interval_s: float = 0.1


def _run_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _summarize_array(value: np.ndarray) -> dict[str, Any]:
    summary: dict[str, Any] = {"kind": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if value.size == 0:
        return summary
    if value.size <= 32:
        summary["values"] = value.tolist()
        return summary
    if np.issubdtype(value.dtype, np.number):
        numeric = value.astype(np.float64, copy=False)
        summary.update(
            {
                "min": float(np.nanmin(numeric)),
                "max": float(np.nanmax(numeric)),
                "mean": float(np.nanmean(numeric)),
                "std": float(np.nanstd(numeric)),
            }
        )
    return summary


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        if isinstance(value, float) and not math.isfinite(value):
            return repr(value)
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _summarize_array(value)
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return _to_jsonable(value.detach().cpu().numpy())
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        public = {k: v for k, v in vars(value).items() if not k.startswith("_")}
        return {"type": type(value).__name__, **_to_jsonable(public)}
    return repr(value)


def _flatten_numeric(prefix: str, value: Any, out: dict[str, float]) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        out[prefix] = float(value)
        return
    if isinstance(value, int | float | np.generic):
        number = float(value)
        if math.isfinite(number):
            out[prefix] = number
        return
    if isinstance(value, np.ndarray):
        if value.size == 1:
            out[prefix] = float(value.reshape(-1)[0])
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten_numeric(f"{prefix}.{key}" if prefix else str(key), item, out)
        return
    if isinstance(value, list | tuple) and len(value) <= 16:
        for idx, item in enumerate(value):
            _flatten_numeric(f"{prefix}.{idx}" if prefix else str(idx), item, out)


def _read_piper_sdk_snapshot(arm: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for getter_name in PIPER_GETTERS:
        getter = getattr(arm, getter_name, None)
        if getter is None:
            continue
        try:
            snapshot[getter_name] = getter()
        except Exception as exc:
            snapshot[getter_name] = {"error": repr(exc)}
    return snapshot


def _compute_derived(
    obs: RobotObservation,
    prev_obs: RobotObservation | None,
    sent_action: RobotAction,
    dt_s: float | None,
) -> dict[str, Any]:
    numeric_obs: dict[str, float] = {}
    numeric_prev: dict[str, float] = {}
    numeric_sent: dict[str, float] = {}
    _flatten_numeric("", obs, numeric_obs)
    _flatten_numeric("", prev_obs, numeric_prev)
    _flatten_numeric("", sent_action, numeric_sent)

    derived: dict[str, Any] = {}
    target_minus_obs = {
        key: target - numeric_obs[key] for key, target in numeric_sent.items() if key in numeric_obs
    }
    if target_minus_obs:
        derived["target_minus_obs"] = target_minus_obs

    if prev_obs is not None and dt_s is not None and dt_s > 1e-6:
        obs_velocity = {
            key: (value - numeric_prev[key]) / dt_s
            for key, value in numeric_obs.items()
            if key in numeric_prev
        }
        if obs_velocity:
            derived["obs_velocity_per_s"] = obs_velocity

    return derived


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(payload), fh, ensure_ascii=False, indent=2, sort_keys=True)


def _write_summary(run_dir: Path) -> None:
    steps_path = run_dir / "audit_steps.jsonl"
    if not steps_path.exists():
        return
    steps = [json.loads(line) for line in steps_path.read_text(encoding="utf-8").splitlines() if line]
    series: dict[str, list[float]] = defaultdict(list)
    for step in steps:
        flat: dict[str, float] = {}
        _flatten_numeric("", step, flat)
        for key, value in flat.items():
            series[key].append(value)

    stats: dict[str, dict[str, float]] = {}
    for key, values in series.items():
        if not values:
            continue
        if len(values) == 1:
            stats[key] = {
                "count": 1.0,
                "mean": values[0],
                "std": 0.0,
                "min": values[0],
                "max": values[0],
                "range": 0.0,
            }
        else:
            stats[key] = {
                "count": float(len(values)),
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values),
                "min": min(values),
                "max": max(values),
                "range": max(values) - min(values),
            }

    interesting_prefixes = (
        "raw_observation.",
        "processed_action.",
        "robot_action_to_send.",
        "sent_action.",
        "derived.",
        "piper_sdk.",
        "dt_s",
    )
    interesting = {
        key: value
        for key, value in stats.items()
        if key == "dt_s" or any(key.startswith(prefix) for prefix in interesting_prefixes)
    }
    _write_json(run_dir / "analysis_stats.json", interesting)

    sorted_by_range = sorted(
        interesting.items(),
        key=lambda item: (item[1].get("range", 0.0), item[1].get("std", 0.0)),
        reverse=True,
    )
    lines = [
        "# Piper Audit Summary",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Frames: `{len(steps)}`",
    ]
    if steps:
        total_time = steps[-1].get("timestamp_s", 0.0) - steps[0].get("timestamp_s", 0.0)
        fps = len(steps) / total_time if total_time > 1e-6 else 0.0
        lines.append(f"- Approx FPS: `{fps:.2f}`")
    lines.extend(
        [
            "",
            "## Highest-Variation Signals",
            "",
            "| key | count | mean | std | min | max | range |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, value in sorted_by_range[:100]:
        lines.append(
            "| `{}` | {:.0f} | {:.5g} | {:.5g} | {:.5g} | {:.5g} | {:.5g} |".format(
                key,
                value.get("count", 0.0),
                value.get("mean", 0.0),
                value.get("std", 0.0),
                value.get("min", 0.0),
                value.get("max", 0.0),
                value.get("range", 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Inspect First",
            "",
            "- `derived.target_minus_obs.*`: command target minus observed joint/gripper state.",
            "- `derived.obs_velocity_per_s.*`: realized response between frames.",
            "- `sent_action.*` vs `raw_observation.*`: whether commands produce measurable motion.",
            "- `piper_sdk.GetArmLowSpdInfoMsgs` / `GetArmHighSpdInfoMsgs`: useful only if stable and varying.",
            "- The LeRobot dataset videos: use them to label drawer/object/button progress.",
        ]
    )
    (run_dir / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _record_one_episode(
    *,
    cfg: PiperAuditConfig,
    robot: Any,
    teleop: Any,
    dataset: LeRobotDataset,
    audit_fh: Any,
    episode_index: int,
    teleop_action_processor: Any,
    robot_action_processor: Any,
    robot_observation_processor: Any,
) -> int:
    frame_count = 0
    prev_obs: RobotObservation | None = None
    prev_t: float | None = None
    start_episode_t = time.perf_counter()
    period_s = 1.0 / cfg.dataset.fps

    def run_with_connection_retry(action_name: str, fn: Any) -> Any:
        timeout_s = max(cfg.communication_retry_timeout_s, 0.0)
        interval_s = max(cfg.communication_retry_interval_s, 0.0)
        deadline_t = time.perf_counter() + timeout_s
        attempts = 0

        while True:
            attempts += 1
            try:
                result = fn()
                if attempts > 1:
                    logging.warning("%s recovered after %d retries.", action_name, attempts - 1)
                return result
            except ConnectionError as error:
                if timeout_s <= 0.0:
                    raise
                remaining_s = deadline_t - time.perf_counter()
                if remaining_s <= 0.0:
                    raise
                logging.warning(
                    "%s failed with transient communication error; retrying for %.2fs (%s)",
                    action_name,
                    remaining_s,
                    error,
                )
                precise_sleep(min(interval_s, remaining_s))

    while True:
        loop_t = time.perf_counter()
        timestamp_s = loop_t - start_episode_t
        if timestamp_s >= cfg.dataset.episode_time_s:
            break

        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        raw_action = run_with_connection_retry("teleop.get_action", teleop.get_action)
        processed_action = teleop_action_processor((raw_action, obs))
        robot_action_to_send = robot_action_processor((processed_action, obs))
        sent_action = run_with_connection_retry(
            "robot.send_action",
            lambda robot_action_to_send=robot_action_to_send: robot.send_action(robot_action_to_send),
        )

        observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
        action_frame = build_dataset_frame(dataset.features, processed_action, prefix=ACTION)
        dataset.add_frame({**observation_frame, **action_frame, "task": cfg.dataset.single_task})

        dt_s = None if prev_t is None else loop_t - prev_t
        arm = getattr(robot, "arm", None)
        piper_sdk = _read_piper_sdk_snapshot(arm) if cfg.sdk_snapshot and arm is not None else {}
        audit_step = {
            "episode_index": episode_index,
            "frame_index": frame_count,
            "timestamp_s": timestamp_s,
            "wall_time_s": time.time(),
            "dt_s": dt_s,
            "raw_observation": obs,
            "processed_observation": obs_processed,
            "raw_teleop_action": raw_action,
            "processed_action": processed_action,
            "robot_action_to_send": robot_action_to_send,
            "sent_action": sent_action,
            "derived": _compute_derived(obs, prev_obs, sent_action, dt_s),
            "piper_sdk": piper_sdk,
        }
        audit_fh.write(json.dumps(_to_jsonable(audit_step), ensure_ascii=False, sort_keys=True) + "\n")
        if frame_count % max(1, cfg.flush_every) == 0:
            audit_fh.flush()

        if cfg.display_data:
            log_rerun_data(
                observation=obs_processed,
                action=processed_action,
                compress_images=cfg.display_compressed_images,
            )

        prev_obs = obs
        prev_t = loop_t
        frame_count += 1

        dt_loop_s = time.perf_counter() - loop_t
        precise_sleep(max(period_s - dt_loop_s, 0.0))

    return frame_count


@parser.wrap()
def piper_audit(cfg: PiperAuditConfig) -> LeRobotDataset:
    init_logging()
    logging.info("Piper audit config:\n%s", json.dumps(_to_jsonable(asdict(cfg)), indent=2, ensure_ascii=False))
    if cfg.display_data:
        init_rerun(session_name="piper_audit", ip=cfg.display_ip, port=cfg.display_port)

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    run_name = cfg.run_name or _run_name()
    run_dir = Path(cfg.audit_dir).expanduser() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )
    dataset_repo_id = cfg.dataset.repo_id
    if dataset_repo_id == "local/piper_audit":
        dataset_repo_id = f"local/piper_audit_{run_name}"
    dataset_root = Path(cfg.dataset.root).expanduser() / dataset_repo_id

    audit_path = run_dir / "audit_steps.jsonl"

    logging.info("Audit run dir: %s", run_dir)
    logging.info("Dataset root: %s", dataset_root)
    dataset = None
    try:
        robot.connect()
        teleop.connect()
        dataset = LeRobotDataset.create(
            dataset_repo_id,
            cfg.dataset.fps,
            root=dataset_root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
            * len(getattr(robot, "cameras", {})),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            vcodec=cfg.dataset.vcodec,
        )
        _write_json(
            run_dir / "meta.json",
            {
                "config": asdict(cfg),
                "dataset_root": str(dataset.root),
                "dataset_repo_id": dataset.repo_id,
                "dataset_features": dataset.features,
                "robot_type": robot.name,
                "robot_id": robot.id,
                "teleop_type": teleop.name,
                "teleop_id": teleop.id,
                "piper_getters": PIPER_GETTERS,
            },
        )
        with audit_path.open("w", encoding="utf-8") as audit_fh, VideoEncodingManager(dataset):
            for episode_idx in range(cfg.dataset.num_episodes):
                logging.info("Recording audit episode %s", dataset.num_episodes)
                frame_count = _record_one_episode(
                    cfg=cfg,
                    robot=robot,
                    teleop=teleop,
                    dataset=dataset,
                    audit_fh=audit_fh,
                    episode_index=dataset.num_episodes,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                )
                if frame_count > 0:
                    dataset.save_episode()
                if episode_idx < cfg.dataset.num_episodes - 1 and cfg.dataset.reset_time_s > 0:
                    logging.info("Reset time: %.2fs", cfg.dataset.reset_time_s)
                    precise_sleep(cfg.dataset.reset_time_s)
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()
        _write_summary(run_dir)
        logging.info("Audit summary: %s", run_dir / "analysis_summary.md")

    return dataset


def main() -> None:
    register_third_party_plugins()
    piper_audit()


if __name__ == "__main__":
    main()
