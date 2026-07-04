# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Local leader-arm server.

Runs on the machine that the physical SO-101 *leader* arm is plugged into.
It opens the arm through LeRobot and exposes its joint readings over a ZeroMQ
REP socket so a remote Isaac Sim process can teleoperate the sim over the network.

Only the six joint values (a small dict of floats) cross the network; the chatty
Feetech serial traffic stays local, so WAN latency does not stall the servo bus.

Requires only `pyzmq`, `msgpack`, and `lerobot` locally -- no Isaac Sim / GPU.

Example:
    python leader_server.py --port /dev/ttyACM0 --robot_id leader_arm_1 \
        --host 0.0.0.0 --zmq_port 5556
"""
import argparse
import os

import msgpack
import zmq

from lerobot.robots import make_robot_from_config
from lerobot.teleoperators.so101_leader import SO101LeaderConfig


def main():
    parser = argparse.ArgumentParser(description="SO-101 leader-arm ZMQ server.")
    parser.add_argument(
        "--port",
        type=str,
        default=os.getenv("TELEOP_PORT", "/dev/ttyACM0"),
        help="Serial port of the physical leader arm.",
    )
    parser.add_argument(
        "--robot_id",
        type=str,
        default=os.getenv("TELEOP_ID", "leader_arm_1"),
        help="LeRobot id of the leader arm (selects the calibration file).",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Interface to bind (0.0.0.0 = all)."
    )
    parser.add_argument("--zmq_port", type=int, default=5556, help="TCP port to listen on.")
    parser.add_argument(
        "--api_token",
        type=str,
        default=os.getenv("LEADER_API_TOKEN", None),
        help="Optional shared secret; clients must present the same token.",
    )
    args = parser.parse_args()

    # Open the leader arm exactly like LeRobotSO101Interface(kind="leader") does.
    cfg = SO101LeaderConfig(port=args.port, id=args.robot_id)
    robot = make_robot_from_config(cfg)
    robot.connect()
    print(f"[INFO]: Connected to leader arm at {args.port} (id={args.robot_id})")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{args.host}:{args.zmq_port}")
    print(f"[INFO]: Leader server listening on tcp://{args.host}:{args.zmq_port}")

    try:
        while True:
            request = msgpack.unpackb(socket.recv())

            # Optional shared-secret check.
            if args.api_token and request.get("api_token") != args.api_token:
                socket.send(msgpack.packb({"error": "unauthorized"}))
                continue

            endpoint = request.get("endpoint", "get_action")

            if endpoint == "ping":
                socket.send(msgpack.packb({"status": "ok"}))
            elif endpoint == "get_action":
                # get_action() returns e.g. {"shoulder_pan.pos": 12.3, ..., "gripper.pos": 45.0}.
                # Cast to plain floats so msgpack can serialize numpy scalars.
                action = {k: float(v) for k, v in robot.get_action().items()}
                socket.send(msgpack.packb(action))
            else:
                socket.send(msgpack.packb({"error": f"unknown endpoint: {endpoint}"}))
    except KeyboardInterrupt:
        print("\n[INFO]: Shutting down leader server.")
    finally:
        robot.disconnect()
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
