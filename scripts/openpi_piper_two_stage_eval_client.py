#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
from collections.abc import Callable
import logging
from pathlib import Path
import signal

from openpi_piper_eval_client import (
    TASK_WHITE_BOX_BARCODE,
    make_robot,
    run_episode,
    send_reset,
    websocket_client_policy,
)


TASK_WHITE_BOX_RIGHT_SUCCESS = "Pick up the white box from the green tray and place it in the success area on the right"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a two-stage Piper eval with one robot/camera connection and two OpenPI policy servers. "
            "Press ENTER once for task A, then ENTER again for task B."
        )
    )
    parser.add_argument("--host", default="localhost", help="Default host for both policy servers.")
    parser.add_argument("--host-a", default=None, help="Task A policy server host. Defaults to --host.")
    parser.add_argument("--host-b", default=None, help="Task B policy server host. Defaults to --host.")
    parser.add_argument("--port-a", type=int, default=8000)
    parser.add_argument("--port-b", type=int, default=8001)
    parser.add_argument("--robot-port", default="can1")
    parser.add_argument("--robot-id", default="my_piper_follower")
    parser.add_argument("--front-serial", default="352122272924")
    parser.add_argument("--wrist-serial", default="409122274629")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-warmup-s", type=int, default=2)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--query-every", type=int, default=50)
    parser.add_argument("--query-every-a", type=int, default=None)
    parser.add_argument("--query-every-b", type=int, default=None)
    parser.add_argument(
        "--action-start-index",
        type=int,
        default=0,
        help="Start executing each policy chunk from this index. Default 0 keeps normal behavior.",
    )
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--stage-a-time-s", type=float, default=100.0)
    parser.add_argument("--stage-b-time-s", type=float, default=100.0)
    parser.add_argument("--reset-time-s", type=float, default=4.0)
    parser.add_argument("--reset-gripper-pos", type=float, default=60.0)
    parser.add_argument("--max-reset-joint-delta-per-step", type=float, default=5.0)
    parser.add_argument("--max-reset-gripper-delta-per-step", type=float, default=15.0)
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
    parser.add_argument("--prompt-a", default=TASK_WHITE_BOX_BARCODE)
    parser.add_argument("--prompt-b", default=TASK_WHITE_BOX_RIGHT_SUCCESS)
    parser.add_argument("--no-wait", action="store_true", help="Run A then B without waiting for ENTER prompts.")
    parser.add_argument("--no-reset", action="store_true", help="Do not reset before stage A.")
    parser.add_argument(
        "--reset-between-stages",
        action="store_true",
        help="Reset before stage B. Usually keep this off because stage A output is stage B input.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Connect to servers and cameras, but do not send actions.")
    parser.add_argument("--log-policy-io", action="store_true", help="Log image/state/action diagnostics for policy calls.")
    parser.add_argument("--debug-image-dir", type=Path, default=None, help="Save first policy input images per stage.")
    return parser.parse_args()


def _stage_args(
    base_args: argparse.Namespace,
    *,
    prompt: str,
    episode_time_s: float,
    query_every: int,
    debug_stage_name: str,
) -> argparse.Namespace:
    args = copy.copy(base_args)
    args.prompt = prompt
    args.episode_time_s = episode_time_s
    args.query_every = query_every
    if base_args.debug_image_dir is not None:
        args.debug_image_dir = base_args.debug_image_dir / debug_stage_name
    return args


def _wait_for_enter(enabled: bool, message: str) -> None:
    if enabled:
        input(message)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = parse_args()
    stop_requested = False

    def _handle_signal(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    host_a = args.host_a or args.host
    host_b = args.host_b or args.host
    query_every_a = args.query_every_a or args.query_every
    query_every_b = args.query_every_b or args.query_every
    wait_enabled = not args.no_wait

    robot = make_robot(args)

    try:
        logging.info("Connecting Piper follower on %s", args.robot_port)
        robot.connect(calibrate=False)

        logging.info("Connecting task A OpenPI policy server at %s:%d", host_a, args.port_a)
        client_a = websocket_client_policy.WebsocketClientPolicy(host=host_a, port=args.port_a)
        logging.info("Task A server metadata: %s", client_a.get_server_metadata())

        logging.info("Connecting task B OpenPI policy server at %s:%d", host_b, args.port_b)
        client_b = websocket_client_policy.WebsocketClientPolicy(host=host_b, port=args.port_b)
        logging.info("Task B server metadata: %s", client_b.get_server_metadata())

        stage_a_args = _stage_args(
            args,
            prompt=args.prompt_a,
            episode_time_s=args.stage_a_time_s,
            query_every=query_every_a,
            debug_stage_name="stage_a",
        )
        stage_b_args = _stage_args(
            args,
            prompt=args.prompt_b,
            episode_time_s=args.stage_b_time_s,
            query_every=query_every_b,
            debug_stage_name="stage_b",
        )

        for episode in range(args.num_episodes):
            if stop_requested:
                break

            _wait_for_enter(
                wait_enabled,
                f"Episode {episode} stage A: place the initial scene, then press ENTER to run barcode-up task...",
            )
            send_reset(robot, args)
            logging.info("Episode %d stage A started", episode)
            run_episode(robot, client_a, stage_a_args, lambda: stop_requested)
            logging.info("Episode %d stage A finished", episode)
            if stop_requested:
                break

            _wait_for_enter(
                wait_enabled,
                f"Episode {episode} stage B: scan/check barcode, then press ENTER to move box to right success area...",
            )
            if args.reset_between_stages:
                send_reset(robot, args)
            logging.info("Episode %d stage B started", episode)
            run_episode(robot, client_b, stage_b_args, lambda: stop_requested)
            logging.info("Episode %d stage B finished", episode)
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
