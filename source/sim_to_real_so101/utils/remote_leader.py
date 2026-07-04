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

Uses the same ZeroMQ REQ pattern as gr00t_client/server_client.py::PolicyClient.
"""
import msgpack
import zmq


class RemoteLeaderClient:
    def __init__(
        self,
        host: str,
        port: int = 5556,
        timeout_ms: int = 2000,
        api_token: str | None = None,
    ):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self.context = zmq.Context()
        self._init_socket()

    def _init_socket(self):
        self.socket = self.context.socket(zmq.REQ)
        # Fail fast instead of blocking forever if the local server dies.
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

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

    def ping(self) -> bool:
        try:
            self._call("ping")
            return True
        except RuntimeError:
            return False

    def get_action(self) -> dict:
        """Return {'shoulder_pan.pos': float, ..., 'gripper.pos': float}."""
        return self._call("get_action")

    def __del__(self):
        try:
            self.socket.close()
            self.context.term()
        except Exception:
            pass
