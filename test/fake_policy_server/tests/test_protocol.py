"""fake server 协议一致性测试：真实 socket 端到端验证，不使用 mock。"""

from __future__ import annotations

import socket
import threading
import time
import urllib.request

import numpy as np
import pytest
import msgpack_numpy
from websockets.exceptions import ConnectionClosedError
from websockets.sync.client import connect

from fake_policy_server.actions import SineActionSource
from fake_policy_server.policy import FakePolicy
from fake_policy_server.server import FakePolicyServer

STATE = np.zeros(14, dtype=np.float32)
IMAGE = np.zeros((480, 640, 3), dtype=np.uint8)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerThread:
    """在后台线程运行 FakePolicyServer，等待端口就绪。"""

    def __init__(self, **server_kwargs) -> None:
        self.port = _free_port()
        self.server = FakePolicyServer(port=self.port, host="127.0.0.1", **server_kwargs)
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "ServerThread":
        self._thread.start()
        self._wait_ready()
        return self

    def __exit__(self, *exc) -> None:
        # 服务端为 daemon 线程，随测试进程退出；无需显式关闭。
        pass

    def _wait_ready(self, timeout: float = 10.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with connect(f"ws://127.0.0.1:{self.port}", open_timeout=1.0):
                    return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("fake server did not become ready in time")

    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"


def _make_server(**kwargs) -> ServerThread:
    return ServerThread(policy=FakePolicy(), **kwargs)


def _observation(**extra) -> dict:
    obs = {"state": STATE, "images": {"top_head": IMAGE, "hand_left": IMAGE, "hand_right": IMAGE}, "prompt": "test"}
    obs.update(extra)
    return obs


def _roundtrip(url: str, obs: dict) -> dict:
    with connect(url, compression=None, max_size=None, open_timeout=10.0) as websocket:
        metadata_frame = websocket.recv(timeout=10.0)
        metadata = msgpack_numpy.unpackb(metadata_frame)
        websocket.send(msgpack_numpy.packb(obs, use_bin_type=True))
        response_frame = websocket.recv(timeout=10.0)
    if isinstance(response_frame, str):
        raise RuntimeError(f"server returned an error frame:\n{response_frame}")
    return metadata, msgpack_numpy.unpackb(response_frame)


def test_smoke_flow_matches_protocol():
    """metadata 帧先行；响应含 actions/server_timing，维度正确。"""
    with _make_server() as server:
        metadata, result = _roundtrip(server.url(), _observation())

    assert isinstance(metadata, dict)
    actions = np.asarray(result["actions"])
    assert actions.shape == (50, 14)
    assert actions.dtype == np.float32
    assert "infer_ms" in result["server_timing"]


def test_second_request_includes_prev_total_ms():
    """第二次请求起响应应携带 prev_total_ms（与真实服务端一致）。"""
    with _make_server() as server, connect(server.url(), compression=None, max_size=None) as websocket:
        websocket.recv(timeout=10.0)  # metadata
        for _ in range(2):
            websocket.send(msgpack_numpy.packb(_observation(), use_bin_type=True))
            result = msgpack_numpy.unpackb(websocket.recv(timeout=10.0))
        assert "prev_total_ms" in result["server_timing"]


def test_healthz():
    with _make_server() as server:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/healthz", timeout=10.0) as response:
            assert response.status == 200
            assert response.read() == b"OK\n"


def test_error_sends_text_traceback_then_closes_1011():
    with _make_server() as server:
        with connect(server.url(), compression=None, max_size=None) as websocket:
            websocket.recv(timeout=10.0)  # metadata
            websocket.send(msgpack_numpy.packb(_observation(trigger_error=True), use_bin_type=True))
            frame = websocket.recv(timeout=10.0)
            assert isinstance(frame, str)
            assert "Traceback" in frame
            with pytest.raises(ConnectionClosedError) as exc_info:
                websocket.recv(timeout=10.0)
            assert exc_info.value.rcvd.code == 1011


def test_sticky_task_value_reused():
    """带 prompt_id 的请求会更新 sticky 值，后续请求复用。"""
    with _make_server() as server, connect(server.url(), compression=None, max_size=None) as websocket:
        websocket.recv(timeout=10.0)  # metadata
        websocket.send(msgpack_numpy.packb(_observation(prompt_id=8), use_bin_type=True))
        websocket.recv(timeout=10.0)
        websocket.send(msgpack_numpy.packb(_observation(), use_bin_type=True))
        websocket.recv(timeout=10.0)
        assert server.server.last_task_value == 8


def test_rtc_extra_keys_tolerated():
    """RTC 模式的附加字段（prev_action_chunk/inference_delay/execute_horizon）不应破坏协议。"""
    with _make_server() as server:
        obs = _observation(
            prev_action_chunk=np.zeros((50, 14), dtype=np.float32),
            inference_delay=4,
            execute_horizon=18,
        )
        metadata, result = _roundtrip(server.url(), obs)
        assert np.asarray(result["actions"]).shape == (50, 14)


def test_sine_source_joints_distinguishable():
    """逐关节正弦动作：各关节轨迹不同，夹爪与关节维度可区分。"""
    source = SineActionSource()
    actions = source(0, action_dim=14, chunk_size=50)
    assert actions.shape == (50, 14)
    assert np.all(np.isfinite(actions))
    # 关节列两两不同（频率/相位错开）。
    joint_cols = [j for j in range(14) if j % 7 != 6]
    for i in range(len(joint_cols)):
        for k in range(i + 1, len(joint_cols)):
            assert not np.allclose(actions[:, joint_cols[i]], actions[:, joint_cols[k]])
    # 左右夹爪可区分。
    assert not np.allclose(actions[:, 6], actions[:, 13])
    # 夹爪列在 mm 量级（30±20），关节列在弧度量级（±0.3）。
    assert np.all(np.abs(actions[:, 6] - 30.0) <= 20.0 + 1e-6)
    assert np.all(np.abs(actions[:, 13] - 30.0) <= 20.0 + 1e-6)
    assert np.all(np.abs(actions[:, 0]) <= 0.3 + 1e-6)
