"""Kai0RtcAdapter 单测：后台推理线程与动作流集成。

传输层注入 FakeAsyncTransport（asyncio，不连网）；与 StreamActionBuffer 纯逻辑
测试（test_stream_buffer.py）互补：本文件验证 RTC 的线程握手、故障语义与
stop/reset 生命周期。
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from kai0_inference.inference import contract
from kai0_inference.inference.policies.kai0.rtc_adapter import Kai0RtcAdapter
from kai0_inference.inference.transport.base import AsyncPersistentTransport


class FakeAsyncTransport(AsyncPersistentTransport):
    """常驻连接假实现：记录请求，返回可配置动作块，可注入故障。"""

    def __init__(
        self,
        *,
        chunk_size: int = 8,
        gripper_mm: float = 50.0,
        fail: BaseException | None = None,
    ) -> None:
        self.requests: list[dict] = []
        self.chunk_size = chunk_size
        self.gripper_mm = gripper_mm
        self.fail = fail
        self.connect_count = 0
        self.close_count = 0

    async def connect(self) -> None:
        self.connect_count += 1

    async def request_actions(self, payload: dict) -> dict:
        self.requests.append(payload)
        if self.fail is not None:
            raise self.fail
        # 关节值 = 行号×0.1（可区分各行），夹爪 = gripper_mm（服务端毫米单位）。
        rows = np.zeros((self.chunk_size, contract.ACTION_DIM), dtype=np.float64)
        for t in range(self.chunk_size):
            rows[t, :6] = rows[t, 7:13] = t * 0.1
            rows[t, 6] = rows[t, 13] = self.gripper_mm
        return {"actions": rows, "server_timing": {"infer_ms": 5.0}}

    async def close(self) -> None:
        self.close_count += 1


def make_observation(state: np.ndarray | None = None, image_shape=(64, 64, 3)) -> dict:
    obs = {
        "observation.state": (
            np.zeros(contract.ACTION_DIM, dtype=np.float64)
            if state is None
            else np.asarray(state, dtype=np.float64)
        )
    }
    for alias in contract.CAMERA_ALIASES:
        obs[f"observation.images.{alias}"] = np.full(image_shape, 128, dtype=np.uint8)
    return obs


def make_rtc_adapter(transport: FakeAsyncTransport, **overrides) -> Kai0RtcAdapter:
    """小参数、快速周期的 RTC 适配器（测试用，避免真实部署的长超时）。"""
    defaults = dict(
        transport_factory=lambda: transport,
        rtc_chunk_size=8,
        rtc_execute_horizon=4,
        rtc_inference_hz=20.0,
        rtc_control_rate=20.0,
        rtc_startup_timeout_s=2.0,
        rtc_stale_timeout_s=30.0,
        rtc_latency_k=2,
        rtc_min_smooth_steps=2,
        rtc_delay_window=4,
        rtc_initial_delay_steps=2,
    )
    defaults.update(overrides)
    return Kai0RtcAdapter(**defaults)


def wait_until(predicate, timeout: float = 3.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def collect_action(adapter: Kai0RtcAdapter, observation: dict, timeout: float = 3.0):
    """轮询 generate_action 直到拿到动作（startup 等待期间的异常重试）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return adapter.generate_action(observation)
        except (RuntimeError, TimeoutError):
            time.sleep(0.02)
    return None


# —— 构造校验（均在启动推理线程前失败，不泄漏线程） ——


def test_rtc_requires_server_url_or_transport_factory():
    with pytest.raises(ValueError, match="server_url"):
        Kai0RtcAdapter(transport_factory=None)
    with pytest.raises(ValueError, match="callable"):
        Kai0RtcAdapter(transport_factory="not callable", server_url="ws://x")


def test_rtc_rejects_invalid_rtc_params():
    def make(**kw) -> Kai0RtcAdapter:
        return Kai0RtcAdapter(transport_factory=lambda: FakeAsyncTransport(), **kw)

    with pytest.raises(ValueError, match="control"):
        make(rtc_control_rate=0)
    with pytest.raises(ValueError, match="inference"):
        make(rtc_inference_hz=-1)
    with pytest.raises(ValueError, match="chunk size"):
        make(rtc_chunk_size=0)
    with pytest.raises(ValueError, match="execute horizon"):
        make(rtc_execute_horizon=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        make(rtc_chunk_size=4, rtc_execute_horizon=8)
    with pytest.raises(ValueError, match="latency_k"):
        make(rtc_latency_k=-1)
    with pytest.raises(ValueError, match="smoothing steps"):
        make(rtc_min_smooth_steps=0)
    with pytest.raises(ValueError, match="timeouts"):
        make(rtc_stale_timeout_s=0)
    with pytest.raises(ValueError, match="delay_window"):
        make(rtc_delay_window=0)
    with pytest.raises(ValueError, match="initial_delay"):
        make(rtc_initial_delay_steps=99)


# —— 正常推理流 ——


def test_rtc_generates_finite_actions_from_fake_server():
    transport = FakeAsyncTransport(chunk_size=8, gripper_mm=50.0)
    adapter = make_rtc_adapter(transport)
    try:
        action = collect_action(adapter, make_observation())
        assert action is not None
        assert action.shape == (contract.ACTION_DIM,)
        assert action.dtype == np.float64
        assert np.all(np.isfinite(action))
        # 夹爪 50mm → 0.05 m（服务端毫米 → Forge 米）。
        assert action[6] == pytest.approx(0.05)
        assert action[13] == pytest.approx(0.05)
    finally:
        adapter.stop()


def test_rtc_payload_has_guidance_fields():
    transport = FakeAsyncTransport()
    adapter = make_rtc_adapter(transport)
    try:
        assert collect_action(adapter, make_observation()) is not None
        # 继续发布观测驱动第二个推理周期：此时缓冲已含首个动作块，
        # 后续请求的 prev_action_chunk 应来自缓冲快照而非 state_hold。
        for _ in range(3):
            adapter.generate_action(make_observation())
        assert wait_until(lambda: len(transport.requests) >= 2)
        payload = transport.requests[0]
        assert "state" in payload and "images" in payload and "prompt" in payload
        # RTC 引导三字段：前序动作块（[chunk, 14]）、推理延迟、引导前缀长度。
        assert payload["prev_action_chunk"].shape == (8, contract.ACTION_DIM)
        assert payload["execute_horizon"] == 4
        assert isinstance(payload["inference_delay"], int)
        assert 0 <= payload["inference_delay"] <= 8
        # 首请求为 state_hold（夹爪 0）；首个动作块入队后，后续请求的快照应
        # 为缓冲轨迹，夹爪 0.05 m → 服务端 50 mm（Forge 米 → 服务端毫米）。
        prev = np.asarray(transport.requests[-1]["prev_action_chunk"])
        assert prev[0, 6] == pytest.approx(50.0)
    finally:
        adapter.stop()


# —— 生命周期与故障语义 ——


def test_rtc_stop_then_generate_action_raises():
    adapter = make_rtc_adapter(FakeAsyncTransport())
    try:
        adapter.stop()
        with pytest.raises(RuntimeError, match="stopped"):
            adapter.generate_action(make_observation())
    finally:
        adapter.stop()


def test_rtc_stop_is_idempotent():
    adapter = make_rtc_adapter(FakeAsyncTransport())
    adapter.stop()
    adapter.stop()  # 不应抛错


def test_rtc_three_consecutive_failures_raise_fault():
    transport = FakeAsyncTransport(fail=RuntimeError("server down"))
    # 短 startup 超时：每次调用不在首块等待上耗满 2s，让故障尽快累积到 3 次。
    adapter = make_rtc_adapter(transport, rtc_startup_timeout_s=0.5)
    try:
        obs = make_observation()
        deadline = time.monotonic() + 8.0
        error: RuntimeError | None = None
        while time.monotonic() < deadline:
            try:
                adapter.generate_action(obs)
            except (RuntimeError, TimeoutError) as exc:
                error = exc
                if "unavailable" in str(exc):
                    break
            time.sleep(0.02)
        assert error is not None
        assert "unavailable" in str(error)
    finally:
        adapter.stop()


def test_rtc_reset_requests_new_observation():
    adapter = make_rtc_adapter(FakeAsyncTransport())
    try:
        # 直接置 False 模拟"推理线程尚未请求"的状态，reset 应重新请求。
        with adapter._observation_condition:
            adapter._async_observation_requested = False
        adapter.reset()
        assert adapter.is_observation_needed() is True
    finally:
        adapter.stop()


def test_rtc_instruction_setter():
    adapter = make_rtc_adapter(FakeAsyncTransport(), instruction="initial")
    try:
        assert adapter.instruction == "initial"
        adapter.instruction = "fold the box"
        assert adapter.instruction == "fold the box"
    finally:
        adapter.stop()
