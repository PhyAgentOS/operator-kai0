"""Kai0SyncAdapter 单测：chunk 摊销 / 单位契约 / 观测组包校验 / 失败语义。

传输层注入 FakeTransport（不连网），策略层与传输层分离的测试路径；
覆盖输入校验、调用顺序、action shape/dtype、action queue、reset、服务端不可达
明确失败。
"""

from __future__ import annotations

import numpy as np
import pytest

from kai0_inference.inference import contract
from kai0_inference.inference.policies.kai0.adapter import Kai0SyncAdapter


class FakeTransport:
    """同步传输层假实现：记录请求，返回可配置动作块，可注入故障。"""

    def __init__(
        self,
        *,
        chunk_size: int = 5,
        gripper_mm: float = 50.0,
        fail: BaseException | None = None,
        include_actions: bool = True,
    ) -> None:
        self.requests: list[dict] = []
        self.chunk_size = chunk_size
        self.gripper_mm = gripper_mm
        self.fail = fail
        self.include_actions = include_actions

    def request_actions(self, payload: dict) -> dict:
        self.requests.append(payload)
        if self.fail is not None:
            raise self.fail
        return self._make_response()

    def _make_response(self) -> dict:
        # 关节值 = 行号×0.1（可区分各行），夹爪 = gripper_mm（服务端毫米单位）。
        rows = np.zeros((self.chunk_size, contract.ACTION_DIM), dtype=np.float64)
        for t in range(self.chunk_size):
            rows[t, :6] = rows[t, 7:13] = t * 0.1
            rows[t, 6] = rows[t, 13] = self.gripper_mm
        response: dict = {"server_timing": {"infer_ms": 12.0}}
        if self.include_actions:
            response["actions"] = rows
        return response


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


def make_adapter(transport: FakeTransport | None = None, **kwargs) -> Kai0SyncAdapter:
    return Kai0SyncAdapter(transport or FakeTransport(), **kwargs)


# —— 构造校验 ——


def test_adapter_requires_positive_execute_action_steps():
    with pytest.raises(ValueError, match="execute_action_steps"):
        Kai0SyncAdapter(FakeTransport(), execute_action_steps=0)
    with pytest.raises(ValueError, match="execute_action_steps"):
        Kai0SyncAdapter(FakeTransport(), execute_action_steps=-3)
    with pytest.raises(ValueError, match="execute_action_steps"):
        Kai0SyncAdapter(FakeTransport(), execute_action_steps=True)


# —— chunk 摊销与队列 ——


def test_adapter_first_generate_action_requests_and_returns_first_row():
    transport = FakeTransport(chunk_size=5)
    adapter = make_adapter(transport)
    action = adapter.generate_action(make_observation())
    assert len(transport.requests) == 1
    assert action.shape == (contract.ACTION_DIM,)
    assert action.dtype == np.float64
    # 首行：关节 0.0，夹爪 50mm → 0.05 m（服务端毫米 → Forge 米）
    assert action[0] == pytest.approx(0.0)
    assert action[6] == pytest.approx(0.05)
    assert action[13] == pytest.approx(0.05)


def test_adapter_amortizes_chunk_without_extra_requests():
    transport = FakeTransport(chunk_size=3)
    adapter = make_adapter(transport)
    obs = make_observation()
    for _ in range(3):
        adapter.generate_action(obs)
    assert len(transport.requests) == 1
    assert adapter.is_observation_needed() is True


def test_adapter_execute_action_steps_truncates_chunk():
    transport = FakeTransport(chunk_size=5)
    adapter = make_adapter(transport, execute_action_steps=2)
    obs = make_observation()
    for _ in range(2):
        adapter.generate_action(obs)
    assert len(transport.requests) == 1
    adapter.generate_action(obs)  # 队列已耗尽 → 重新请求
    assert len(transport.requests) == 2


def test_adapter_reset_clears_queue_and_triggers_new_request():
    transport = FakeTransport(chunk_size=3)
    adapter = make_adapter(transport)
    obs = make_observation()
    adapter.generate_action(obs)
    adapter.generate_action(obs)  # 队列还剩 1
    assert len(transport.requests) == 1
    adapter.reset()
    assert adapter.is_observation_needed() is True
    adapter.generate_action(obs)
    assert len(transport.requests) == 2


# —— 观测组包（Forge → 服务端契约） ——


def test_adapter_payload_state_grippers_converted_to_mm():
    transport = FakeTransport()
    adapter = make_adapter(transport)
    state = np.zeros(contract.ACTION_DIM)
    state[6] = 0.02  # 夹爪 0.02 m
    state[13] = 0.03
    adapter.generate_action(make_observation(state=state))
    sent = np.asarray(transport.requests[0]["state"])
    assert sent.dtype == np.float32
    assert sent[6] == pytest.approx(20.0)  # m → mm
    assert sent[13] == pytest.approx(30.0)


def test_adapter_payload_contains_prompt_and_images():
    transport = FakeTransport()
    adapter = make_adapter(transport, instruction="fold the box")
    adapter.generate_action(make_observation())
    payload = transport.requests[0]
    assert payload["prompt"] == "fold the box"
    for alias in contract.CAMERA_ALIASES:
        img = payload["images"][alias]
        assert img.dtype == np.uint8
        assert img.shape == (64, 64, 3)


def test_adapter_build_payload_rejects_missing_state():
    adapter = make_adapter()
    with pytest.raises(ValueError, match="observation.state"):
        adapter.build_payload({})


def test_adapter_build_payload_rejects_wrong_state_dim():
    adapter = make_adapter()
    with pytest.raises(ValueError, match=r"\[14\]"):
        adapter.build_payload({"observation.state": np.zeros(13)})


def test_adapter_build_payload_rejects_non_finite_state():
    adapter = make_adapter()
    obs = make_observation(state=np.array([np.nan] + [0.0] * 13))
    with pytest.raises(ValueError, match="finite"):
        adapter.build_payload(obs)


def test_adapter_build_payload_rejects_missing_image():
    adapter = make_adapter()
    obs = make_observation()
    del obs["observation.images.top_head"]
    with pytest.raises(ValueError, match="top_head"):
        adapter.build_payload(obs)


def test_adapter_build_payload_rejects_bad_image_channels():
    adapter = make_adapter()
    obs = make_observation()
    obs["observation.images.top_head"] = np.zeros((64, 64, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="3 channels"):
        adapter.build_payload(obs)


# —— 响应拆包校验 ——


def test_adapter_parse_actions_rejects_missing_actions():
    adapter = make_adapter(FakeTransport(include_actions=False))
    with pytest.raises(ValueError, match="actions"):
        adapter.generate_action(make_observation())


def test_adapter_parse_actions_rejects_wrong_width():
    adapter = make_adapter()
    with pytest.raises(ValueError, match="ACTION_DIM"):
        adapter._parse_actions({"actions": np.zeros((3, 13))})


def test_adapter_parse_actions_rejects_non_finite():
    adapter = make_adapter()
    with pytest.raises(ValueError, match="finite"):
        adapter._parse_actions({"actions": np.array([[np.nan] * contract.ACTION_DIM])})


# —— 失败语义与图像键 ——


def test_adapter_transport_failure_propagates():
    adapter = make_adapter(FakeTransport(fail=ConnectionError("no server")))
    with pytest.raises(ConnectionError, match="no server"):
        adapter.generate_action(make_observation())


def test_adapter_required_image_keys_match_camera_aliases():
    adapter = make_adapter()
    assert adapter.required_image_keys == frozenset(
        f"observation.images.{alias}" for alias in contract.CAMERA_ALIASES
    )
