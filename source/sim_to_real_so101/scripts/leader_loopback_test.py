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
"""Loopback sanity-check for the remote leader-arm bridge.

Spins up a *mock* leader server on 127.0.0.1 (no physical arm, no `lerobot`) and
drives it with ``RemoteLeaderClient`` over both transports. Use this to verify the
ZMQ/msgpack wiring end-to-end before wheeling out the real arm and cloud sim.

The mock speaks the exact wire protocol of ``leader_server.py`` but returns
synthetic joint values, so only `pyzmq` and `msgpack` are required to run it.

    python leader_loopback_test.py
"""
import sys
import threading
import time
from pathlib import Path

import msgpack
import zmq

# Allow running directly from a checkout (adds .../source to sys.path).
try:
    from sim_to_real_so101.utils.remote_leader import RemoteLeaderClient
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from sim_to_real_so101.utils.remote_leader import RemoteLeaderClient

EXPECTED_JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def make_fake_action(pan: float = 0.0) -> dict:
    """A plausible SO-101 leader reading; ``pan`` lets a test track freshness."""
    return {
        "shoulder_pan.pos": float(pan),
        "shoulder_lift.pos": 2.0,
        "elbow_flex.pos": 3.0,
        "wrist_flex.pos": 4.0,
        "wrist_roll.pos": 5.0,
        "gripper.pos": 50.0,
    }


def bind_ephemeral(socket) -> int:
    """Bind to an OS-chosen port on localhost and return it."""
    socket.bind("tcp://127.0.0.1:0")
    endpoint = socket.getsockopt_string(zmq.LAST_ENDPOINT)
    return int(endpoint.rsplit(":", 1)[1])


def serve_reqrep_mock(socket, stop):
    while not stop.is_set():
        if socket.poll(50) == 0:
            continue
        request = msgpack.unpackb(socket.recv())
        if request.get("endpoint") == "ping":
            socket.send(msgpack.packb({"status": "ok"}))
        else:
            socket.send(msgpack.packb(make_fake_action()))


def serve_pubsub_mock(socket, stop):
    pan = 0.0
    while not stop.is_set():
        socket.send(msgpack.packb(make_fake_action(pan)))
        pan += 1.0
        time.sleep(0.01)  # ~100 Hz


def check(condition: bool, msg: str):
    print(f"  [{'PASS' if condition else 'FAIL'}] {msg}")
    return condition


def test_reqrep() -> bool:
    print("reqrep transport:")
    ctx = zmq.Context.instance()
    server = ctx.socket(zmq.REP)
    port = bind_ephemeral(server)
    stop = threading.Event()
    t = threading.Thread(target=serve_reqrep_mock, args=(server, stop), daemon=True)
    t.start()

    ok = True
    try:
        client = RemoteLeaderClient(host="127.0.0.1", port=port, transport="reqrep")
        ok &= check(client.ping(), "ping succeeds")
        action = client.get_action()
        ok &= check(sorted(action) == sorted(EXPECTED_JOINTS), "get_action has all 6 joints")
        ok &= check(
            all(isinstance(v, float) for v in action.values()), "values are floats"
        )
    finally:
        stop.set()
        t.join(timeout=1)
        server.close()
    return ok


def test_pubsub() -> bool:
    print("pubsub transport:")
    ctx = zmq.Context.instance()
    server = ctx.socket(zmq.PUB)
    port = bind_ephemeral(server)
    stop = threading.Event()
    t = threading.Thread(target=serve_pubsub_mock, args=(server, stop), daemon=True)
    t.start()

    ok = True
    try:
        client = RemoteLeaderClient(host="127.0.0.1", port=port, transport="pubsub")
        ok &= check(client.ping(), "ping receives a published sample")
        action = client.get_action()
        ok &= check(sorted(action) == sorted(EXPECTED_JOINTS), "get_action has all 6 joints")

        # The publisher increments shoulder_pan; after a pause we should see a newer value.
        first = client.get_action()["shoulder_pan.pos"]
        time.sleep(0.1)
        second = client.get_action()["shoulder_pan.pos"]
        ok &= check(second > first, f"receives fresh samples ({first} -> {second})")
    finally:
        stop.set()
        t.join(timeout=1)
        server.close()
    return ok


def main():
    print("Leader bridge loopback test (mock server on 127.0.0.1)\n")
    results = [test_reqrep(), test_pubsub()]
    print()
    if all(results):
        print("All checks passed.")
        return 0
    print("Some checks FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
