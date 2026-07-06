#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections.abc import Callable
import logging
from pathlib import Path
import signal
import sys
import time

import numpy as np

EVO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_CLIENT_SRC = EVO_ROOT / "openpi-lotus-v2-workspace" / "packages" / "openpi-client" / "src"
if OPENPI_CLIENT_SRC.exists():
    sys.path.insert(0, str(OPENPI_CLIENT_SRC))

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots.piper_follower.config_piper_follower import PiperFollowerConfig
from lerobot.robots.piper_follower.piper_follower import PiperFollower
from lerobot.utils.piper_sdk import PIPER_ACTION_KEYS
from openpi_client import websocket_client_policy

TASK_WHITE_BOX_BARCODE = (
    "Pick up the white box, place it in the center of the green rectangular tray, "
    "and orient the box so the barcode side faces upward and is clearly visible"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Piper eval by querying a local OpenPI websocket policy server.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--robot-port", default="can1")
    parser.add_argument("--robot-id", default="my_piper_follower")
    parser.add_argument("--front-serial", default="352122272924")
    parser.add_argument("--wrist-serial", default="409122274629")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-warmup-s", type=int, default=2)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--query-every", type=int, default=10)
    parser.add_argument(
        "--action-start-index",
        type=int,
        default=0,
        help="Start executing each policy chunk from this index. Default 0 keeps normal behavior.",
    )
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--episode-time-s", type=float, default=40.0)
    parser.add_argument("--reset-time-s", type=float, default=12.0)
    parser.add_argument("--reset-gripper-pos", type=float, default=60.0)
    parser.add_argument("--max-reset-joint-delta-per-step", type=float, default=2.0)
    parser.add_argument("--max-reset-gripper-delta-per-step", type=float, default=5.0)
    parser.add_argument(
        "--max-joint-delta-per-step",
        type=float,
        default=0.0,
        help="Optional policy-action joint delta limit per control step. <=0 disables limiting.",
    )
    parser.add_argument(
        "--max-gripper-delta-per-step",
        type=float,
        default=0.0,
        help="Optional policy-action gripper delta limit per control step. <=0 disables limiting.",
    )
    parser.add_argument("--prompt", default=TASK_WHITE_BOX_BARCODE)
    parser.add_argument("--no-wait", action="store_true", help="Do not pause for ENTER before each episode.")
    parser.add_argument("--no-reset", action="store_true", help="Do not send zero-joint/open-gripper reset actions.")
    parser.add_argument("--reset-only", action="store_true", help="Send reset actions, then exit without running inference.")
    parser.add_argument("--dry-run", action="store_true", help="Connect to server and cameras, but do not send actions.")
    parser.add_argument("--log-policy-io", action="store_true", help="Log image/state/action diagnostics for policy calls.")
    parser.add_argument("--debug-image-dir", type=Path, default=None, help="Save the first policy input images as PPM files.")
    return parser.parse_args()


def make_robot(args: argparse.Namespace) -> PiperFollower:
    cameras = {
        "front": RealSenseCameraConfig(
            serial_number_or_name=args.front_serial,
            width=args.width,
            height=args.height,
            fps=args.camera_fps,
            warmup_s=args.camera_warmup_s,
        ),
        "wrist": RealSenseCameraConfig(
            serial_number_or_name=args.wrist_serial,
            width=args.width,
            height=args.height,
            fps=args.camera_fps,
            warmup_s=args.camera_warmup_s,
        ),
    }
    return PiperFollower(
        PiperFollowerConfig(
            port=args.robot_port,
            id=args.robot_id,
            require_calibration=False,
            cameras=cameras,
        )
    )


def image_to_chw(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {image.shape}")
    if image.shape[0] in (1, 3, 4):
        return image
    if image.shape[-1] in (1, 3, 4):
        return np.moveaxis(image, -1, 0)
    raise ValueError(f"Expected image in CHW or HWC format, got shape {image.shape}")


def image_to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected image in CHW or HWC format, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if image.max(initial=0) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    return np.ascontiguousarray(image)


def write_ppm(path: Path, image: np.ndarray) -> None:
    image = image_to_hwc_uint8(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = image.shape[:2]
    with path.open("wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        f.write(image.tobytes())


def image_summary(name: str, image: np.ndarray) -> str:
    image = np.asarray(image)
    return (
        f"{name}: shape={tuple(image.shape)} dtype={image.dtype} "
        f"min={float(image.min(initial=0)):.1f} max={float(image.max(initial=0)):.1f} "
        f"mean={float(image.mean()):.1f}"
    )


def rounded(values: np.ndarray, decimals: int = 3) -> list[float]:
    return np.round(np.asarray(values, dtype=np.float32), decimals).tolist()


def action_preview(action_chunk: np.ndarray, current_state: np.ndarray, args: argparse.Namespace) -> dict[str, list[float]]:
    end_index = min(args.action_start_index + args.query_every - 1, len(action_chunk) - 1)
    indices = {
        "idx0": 0,
        "idx_start": args.action_start_index,
        "idx_end": end_index,
        "idx_last": len(action_chunk) - 1,
    }
    return {
        name: rounded(action_chunk[max(0, min(index, len(action_chunk) - 1)), :7] - current_state)
        for name, index in indices.items()
    }


def observation_to_openpi(obs: dict, prompt: str) -> tuple[dict, np.ndarray]:
    state = np.asarray([obs[key] for key in PIPER_ACTION_KEYS], dtype=np.float32)
    payload = {
        "images": {
            "cam_high": image_to_chw(obs["front"]),
            "cam_wrist": image_to_chw(obs["wrist"]),
        },
        "state": state,
        "prompt": prompt,
    }
    return payload, state


def infer_policy_chunk(
    client: websocket_client_policy.WebsocketClientPolicy,
    payload: dict,
) -> tuple[np.ndarray, float]:
    t0 = time.monotonic()
    result = client.infer(payload)
    rtt = time.monotonic() - t0
    action_chunk = np.asarray(result["actions"], dtype=np.float32)
    if action_chunk.ndim != 2 or action_chunk.shape[1] < 7:
        raise ValueError(f"Expected action chunk with shape (T, >=7), got {action_chunk.shape}")
    return action_chunk, rtt


def action_array_to_dict(action: np.ndarray, current_state: np.ndarray, args: argparse.Namespace) -> dict[str, float]:
    action = np.asarray(action[:7], dtype=np.float32)
    if not np.all(np.isfinite(action)):
        raise ValueError(f"OpenPI returned non-finite action: {action}")

    commanded = action.copy()
    if args.max_joint_delta_per_step > 0:
        joint_delta = np.clip(
            action[:6] - current_state[:6],
            -args.max_joint_delta_per_step,
            args.max_joint_delta_per_step,
        )
        commanded[:6] = current_state[:6] + joint_delta
    if args.max_gripper_delta_per_step > 0:
        gripper_delta = np.clip(
            action[6] - current_state[6],
            -args.max_gripper_delta_per_step,
            args.max_gripper_delta_per_step,
        )
        commanded[6] = current_state[6] + gripper_delta
    return {key: float(value) for key, value in zip(PIPER_ACTION_KEYS, commanded, strict=True)}


def send_reset(robot: PiperFollower, args: argparse.Namespace) -> None:
    if args.no_reset or args.reset_time_s <= 0:
        return
    obs = robot.get_observation()
    current = np.asarray([obs[key] for key in PIPER_ACTION_KEYS], dtype=np.float32)
    target = np.zeros_like(current)
    target[6] = args.reset_gripper_pos
    period_s = 1.0 / args.control_hz
    steps = max(1, int(args.reset_time_s * args.control_hz))

    for step in range(1, steps + 1):
        alpha = step / steps
        commanded = current + (target - current) * alpha
        previous = current if step == 1 else previous_commanded
        delta = commanded - previous
        delta[:6] = np.clip(
            delta[:6],
            -args.max_reset_joint_delta_per_step,
            args.max_reset_joint_delta_per_step,
        )
        delta[6] = np.clip(
            delta[6],
            -args.max_reset_gripper_delta_per_step,
            args.max_reset_gripper_delta_per_step,
        )
        commanded = previous + delta
        previous_commanded = commanded
        reset_action = {key: float(value) for key, value in zip(PIPER_ACTION_KEYS, commanded, strict=True)}
        if not args.dry_run:
            robot.send_action(reset_action)
        time.sleep(period_s)


def run_episode(
    robot: PiperFollower,
    client: websocket_client_policy.WebsocketClientPolicy,
    args: argparse.Namespace,
    should_stop: Callable[[], bool],
) -> None:
    period_s = 1.0 / args.control_hz
    action_chunk = None
    chunk_index = 0
    step = 0
    saved_debug_images = False
    start_t = time.monotonic()
    next_log_t = start_t

    while time.monotonic() - start_t < args.episode_time_s and not should_stop():
        loop_t = time.monotonic()
        obs = robot.get_observation()
        payload, current_state = observation_to_openpi(obs, args.prompt)

        if args.debug_image_dir is not None and not saved_debug_images:
            for name, image in payload["images"].items():
                write_ppm(args.debug_image_dir / f"{name}.ppm", image)
            logging.info("Saved policy input images to %s", args.debug_image_dir)
            saved_debug_images = True

        chunk_limit = None if action_chunk is None else min(args.action_start_index + args.query_every, len(action_chunk))
        if action_chunk is None or (chunk_limit is not None and chunk_index >= chunk_limit):
            if args.log_policy_io:
                logging.info(
                    "policy_input step=%d state=%s %s %s",
                    step,
                    rounded(current_state),
                    image_summary("cam_high", payload["images"]["cam_high"]),
                    image_summary("cam_wrist", payload["images"]["cam_wrist"]),
                )
            action_chunk, rtt = infer_policy_chunk(client, payload)
            if args.action_start_index < 0 or args.action_start_index >= len(action_chunk):
                raise ValueError(
                    f"--action-start-index must be in [0, {len(action_chunk) - 1}], got {args.action_start_index}"
                )
            chunk_index = args.action_start_index
            if args.log_policy_io:
                first_action = action_chunk[0, :7]
                logging.info(
                    "policy_output step=%d rtt=%.3fs shape=%s first_action=%s first_minus_state=%s preview=%s min=%.3f max=%.3f",
                    step,
                    rtt,
                    tuple(action_chunk.shape),
                    rounded(first_action),
                    rounded(first_action - current_state),
                    action_preview(action_chunk, current_state, args),
                    float(action_chunk[:, :7].min(initial=0)),
                    float(action_chunk[:, :7].max(initial=0)),
                )

        raw_action = action_chunk[chunk_index]
        action = action_array_to_dict(raw_action, current_state, args)
        commanded_state = np.asarray([action[key] for key in PIPER_ACTION_KEYS], dtype=np.float32)
        chunk_index += 1
        step += 1

        if not args.dry_run:
            robot.send_action(action)

        now = time.monotonic()
        if now >= next_log_t:
            if args.log_policy_io:
                logging.info(
                    "step=%d state=%s raw_action=%s raw_minus_state=%s command_delta=%s",
                    step,
                    rounded(current_state),
                    rounded(raw_action[:7]),
                    rounded(raw_action[:7] - current_state),
                    rounded(commanded_state - current_state),
                )
            else:
                logging.info("step=%d action=%s", step, {k: round(v, 3) for k, v in action.items()})
            next_log_t = now + 1.0

        sleep_s = period_s - (time.monotonic() - loop_t)
        if sleep_s > 0:
            time.sleep(sleep_s)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = parse_args()
    stop_requested = False

    def _handle_signal(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    robot = make_robot(args)

    try:
        logging.info("Connecting Piper follower on %s", args.robot_port)
        robot.connect(calibrate=False)
        if args.reset_only:
            logging.info("Reset-only mode: sending reset for %.1fs", args.reset_time_s)
            send_reset(robot, args)
            return

        logging.info("Connecting OpenPI policy server at %s:%d", args.host, args.port)
        client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
        logging.info("Server metadata: %s", client.get_server_metadata())

        for episode in range(args.num_episodes):
            if stop_requested:
                break
            if not args.no_wait:
                input(f"Episode {episode}: place the white box and green tray, then press ENTER to start...")
            send_reset(robot, args)
            logging.info("Episode %d started", episode)
            run_episode(robot, client, args, lambda: stop_requested)
            logging.info("Episode %d finished", episode)
            if stop_requested:
                logging.info("Stop requested; exiting eval loop")
                break
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
