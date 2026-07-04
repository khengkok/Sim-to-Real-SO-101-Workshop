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
"""Client for the remote leader-arm server (see scripts/leader_server.py).

Runs inside the Isaac Sim process on the cloud. Its ``get_action()`` returns the
same joint dict shape as LeRobot's ``robot.get_action()``, so it can be dropped
straight into the teleop loop in place of a locally-connected leader arm.

Two transports mirror the server (``transport``):

- ``reqrep``: REQ socket, one round-trip per ``get_action()`` -- same pattern as
  gr00t_client/server_client.py::PolicyClient.
- ``pubsub``: SUB socket with ``CONFLATE`` (keeps only the newest sample).
  ``get_action()`` returns the freshest published reading without blocking on a
  round-trip, giving lower latency over a high-latency link.
"""
import msgpack
import zmq


class RemoteLeaderClient:
    def __init__(
        self,
        host: str,
        port: int = 5556,
        transport: str = "reqrep",
        timeout_ms: int = 2000,
        api_token: str | None = None,
    ):
        if transport not in ("reqrep", "pubsub"):
            raise ValueError(f"unknown transport: {transport}")
        self.host = host
        self.port = port
        self.transport = transport
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self.context = zmq.Context()
        self._last: dict | None = None  # most recent reading (pubsub)
        self._init_socket()

    def _init_socket(self):
        if self.transport == "reqrep":
            self.socket = self.context.socket(zmq.REQ)
        else:
            self.socket = self.context.socket(zmq.SUB)
            # CONFLATE keeps only the latest message, so we never fall behind.
            # It must be set before connect and requires an empty subscription filter.
            self.socket.setsockopt(zmq.CONFLATE, 1)
            self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    # ------------------------------------------------------------------ reqrep
    def _call(self, endpoint: str) -> dict:
        request = {"endpoint": endpoint}
        if self.api_token:
            request["api_token"] = self.api_token
        try:
            self.socket.send(msgpack.packb(request))
            response = msgpack.unpackb(self.socket.recv())
        except zmq.error.ZMQError as e:
            # A REQ socket in an errored state can't be reused -- recreate it.
            self.socket.close()
            self._init_socket()
            raise RuntimeError(f"Leader server call '{endpoint}' failed: {e}") from e
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Leader server error: {response['error']}")
        return response

    # ------------------------------------------------------------------ pubsub
    def _recv_latest(self) -> dict:
        """Return the newest published reading.

        With CONFLATE the socket buffers at most one message. If a new one has
        arrived, take it; otherwise reuse the last reading. On the very first call
        we block (up to timeout) until a reading is available.
        """
        if self._last is None:
            self._last = msgpack.unpackb(self.socket.recv())  # blocks up to RCVTIMEO
            return self._last
        try:
            self._last = msgpack.unpackb(self.socket.recv(zmq.NOBLOCK))
        except zmq.error.Again:
            pass  # nothing new yet -> reuse previous reading
        return self._last

    # ------------------------------------------------------------------ public
    def ping(self) -> bool:
        try:
            if self.transport == "reqrep":
                self._call("ping")
            else:
                # PUB/SUB is one-way: "reachable" == a message arrives within timeout.
                self._last = msgpack.unpackb(self.socket.recv())
            return True
        except (RuntimeError, zmq.error.ZMQError):
            self.socket.close()
            self._init_socket()
            return False

    def get_action(self) -> dict:
        """Return {'shoulder_pan.pos': float, ..., 'gripper.pos': float}."""
        if self.transport == "reqrep":
            return self._call("get_action")
        return self._recv_latest()

    def __del__(self):
        try:
            self.socket.close()
            self.context.term()
        except Exception:
            pass
