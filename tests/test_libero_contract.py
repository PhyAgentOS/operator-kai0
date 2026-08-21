"""LIBERO 契约单测：配置校验 / 契约校验 / payload 组拆包 / adapter 行为 / 负例矩阵。

覆盖规划第 4/5 步的 kai0 侧防护层验收（增强 D 的替代口径）：
- 维度严格 ==7（8D/14D 负例 fail fast）
- 关节名与顺序（错名/错序在配置层 fail fast）
- velocity/effort 恒零由「joints 全部 mode=position」构造保证（见 _build_joint_command）
- NaN/Inf fail fast
双臂契约（DUAL_ARM）的值与行为不变由既有 tests/test_config.py、test_kai0_adapter.py
与 test_contract.py 回归覆盖。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kai0_inference.inference import contract
from kai0_inference.inference.config import PolicyNodeConfig, load_config
from kai0_inference.inference.policies.kai0 import payloads
from kai0_inference.inference.policies.kai0.adapter import Kai0SyncAdapter
from kai0_inference.inference.policies.registry import create_policy_adapter

REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_YAML = REPO_ROOT / "config" / "inference" / "kai0_libero.yaml"

LIBERO_STATE_KEYS = ("observation/image", "observation/wrist_image")


class FakeTransport:
    """同步传输层假实现：记录请求，返回可配置 7 维动作块。"""

    def __init__(self, *, chunk_size: int = 5, width: int = 7) -> None:
        self.requests: list[dict] = []
        self.chunk_size = chunk_size
        self.width = width

    def request_actions(self, payload: dict) -> dict:
        self.requests.append(payload)
        rows = np.zeros((self.chunk_size, self.width), dtype=np.float64)
        for t in range(self.chunk_size):
            rows[t, :6] = t * 0.1
            rows[t, 6] = -1.0 + 0.02 * t  # 夹爪在 [-1,1] 原生域内
        return {"actions": rows, "server_timing": {"infer_ms": 10.0}}


def _valid_dict() -> dict:
    return {
        "joints": [{"name": n} for n in contract.LIBERO.joint_names],
        "policy": {"type": "kai0_libero"},
        "image_inputs": {"image/main": "image", "image/second": "wrist_image"},
        "state_joints": list(contract.LIBERO.joint_names) + ["gripper"],
    }


def make_observation(state_dim: int = 8, image_shape=(64, 64, 3)) -> dict:
    obs = {"observation.state": np.zeros(state_dim, dtype=np.float64)}
    for alias in contract.LIBERO.camera_aliases:
        obs[f"observation.images.{alias}"] = np.zeros(image_shape, dtype=np.uint8)
    return obs


# —— 配置层 ——


def test_load_bundled_libero_yaml():
    config = load_config(LIBERO_YAML)
    assert config.policy["type"] == "kai0_libero"
    assert config.contract is contract.LIBERO
    assert len(config.joints) == 7
    assert config.joint_order == list(contract.LIBERO.joint_names)
    assert config.state_joint_order == list(contract.LIBERO.joint_names) + ["gripper"]
    assert config.alias_for_cameras == list(contract.LIBERO.camera_aliases)
    assert config.auto_start is False


def test_libero_contract_selected_by_policy_type():
    config = PolicyNodeConfig.from_dict(_valid_dict())
    assert config.contract is contract.LIBERO


def test_libero_contract_explicit_key_override():
    data = _valid_dict()
    data["contract"] = "libero"
    assert PolicyNodeConfig.from_dict(data).contract is contract.LIBERO


def test_libero_config_rejects_unknown_contract_key():
    data = _valid_dict()
    data["contract"] = "nope"
    with pytest.raises(ValueError, match="unknown policy contract"):
        PolicyNodeConfig.from_dict(data)


def test_libero_config_rejects_wrong_joint_names():
    data = _valid_dict()
    # 错名：双臂关节名
    data["joints"] = [{"name": n} for n in contract.GLOBAL_JOINT_ORDER]
    with pytest.raises(ValueError, match="joint_names 顺序一致"):
        PolicyNodeConfig.from_dict(data)


def test_libero_config_rejects_wrong_joint_order():
    data = _valid_dict()
    data["joints"] = [{"name": n} for n in reversed(contract.LIBERO.joint_names)]
    with pytest.raises(ValueError, match="joint_names 顺序一致"):
        PolicyNodeConfig.from_dict(data)


def test_libero_config_rejects_14d_joints():
    data = _valid_dict()
    # 14 维双臂动作布局（维度对、名字不对）：拒绝
    data["joints"] = [{"name": n} for n in contract.GLOBAL_JOINT_ORDER]
    with pytest.raises(ValueError):
        PolicyNodeConfig.from_dict(data)


def test_libero_config_rejects_wrong_camera_aliases():
    data = _valid_dict()
    data["image_inputs"] = {"image/main": "image", "image/second": "top_head"}
    with pytest.raises(ValueError, match="camera_aliases"):
        PolicyNodeConfig.from_dict(data)


def test_libero_config_rejects_missing_camera():
    data = _valid_dict()
    data["image_inputs"] = {"image/main": "image"}
    with pytest.raises(ValueError, match="camera_aliases"):
        PolicyNodeConfig.from_dict(data)


# —— 契约层（动作维度 / 有限性 / 单位） ——


def test_libero_validate_action_accepts_7d():
    arr = contract.LIBERO.validate_action(np.zeros(7))
    assert arr.shape == (7,)
    assert arr.dtype == np.float64


def test_libero_validate_action_rejects_8d_and_14d():
    for dim in (6, 8, 14):
        with pytest.raises(ValueError, match="ACTION_DIM=7"):
            contract.LIBERO.validate_action(np.zeros(dim))


def test_libero_validate_action_rejects_nan_inf():
    for bad in (np.nan, np.inf, -np.inf):
        arr = np.zeros(7)
        arr[3] = bad
        with pytest.raises(ValueError, match="finite"):
            contract.LIBERO.validate_action(arr)


def test_libero_validate_action_rejects_2d():
    with pytest.raises(ValueError, match="1-D"):
        contract.LIBERO.validate_action(np.zeros((1, 7)))


def test_libero_gripper_range_check_disabled():
    # 夹爪 [-1,1] 原生域：任何值都不触发米制范围告警
    arr = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5])
    assert contract.LIBERO.gripper_range_violations(arr) == []


def test_libero_no_gripper_unit_conversion():
    state = np.arange(8, dtype=np.float64) / 10.0
    out = contract.LIBERO.to_server_state(state)
    assert out.dtype == np.float32
    assert np.allclose(out, state)  # 原样透传，不做米↔毫米换算


# —— payload 组包 / 拆包 ——


def test_libero_build_payload_flat_observation_keys():
    payload = payloads.build_payload(make_observation(), "pick up bowl", contract.LIBERO)
    assert set(payload) == {"observation/state", "observation/image",
                            "observation/wrist_image", "prompt"}
    assert payload["prompt"] == "pick up bowl"
    assert payload["observation/state"].shape == (8,)
    assert payload["observation/state"].dtype == np.float32


def test_libero_build_payload_rejects_wrong_state_dim():
    with pytest.raises(ValueError, match=r"\[8\]"):
        payloads.build_payload(make_observation(state_dim=14), "x", contract.LIBERO)


def test_libero_build_payload_rejects_missing_image():
    obs = make_observation()
    del obs["observation.images.wrist_image"]
    with pytest.raises(ValueError, match="wrist_image"):
        payloads.build_payload(obs, "x", contract.LIBERO)


def test_libero_parse_actions_accepts_7w_chunk():
    rows = np.zeros((10, 7))
    out = payloads.parse_actions({"actions": rows}, contract.LIBERO)
    assert out.shape == (10, 7)


def test_libero_parse_actions_rejects_14w_chunk():
    rows = np.zeros((10, 14))
    with pytest.raises(ValueError, match="ACTION_DIM=7"):
        payloads.parse_actions({"actions": rows}, contract.LIBERO)


def test_libero_parse_actions_rejects_nan():
    rows = np.zeros((10, 7))
    rows[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        payloads.parse_actions({"actions": rows}, contract.LIBERO)


# —— adapter 行为 ——


def test_libero_adapter_required_image_keys():
    adapter = Kai0SyncAdapter(FakeTransport(), policy_contract=contract.LIBERO)
    assert adapter.required_image_keys == {
        "observation.images.image", "observation.images.wrist_image"
    }


def test_libero_adapter_generate_action_returns_7d():
    transport = FakeTransport(chunk_size=3)
    adapter = Kai0SyncAdapter(transport, policy_contract=contract.LIBERO)
    action = adapter.generate_action(make_observation())
    assert action.shape == (7,)
    assert len(transport.requests) == 1
    # chunk 摊销：后续两步走本地队列，不再发请求
    adapter.generate_action({})
    adapter.generate_action({})
    assert len(transport.requests) == 1
    # 队列耗尽后重新请求；夹爪原样透传（不做毫米→米）
    action = adapter.generate_action(make_observation())
    assert len(transport.requests) == 2
    assert action[6] == pytest.approx(-1.0)  # 服务端第一行夹爪值原样返回


def test_libero_adapter_payload_state_unchanged_and_prompt_updated():
    transport = FakeTransport()
    adapter = Kai0SyncAdapter(
        transport, instruction="initial", policy_contract=contract.LIBERO
    )
    adapter.instruction = "updated instruction"  # set_instruction 语义
    obs = make_observation()
    obs["observation.state"] = np.arange(8, dtype=np.float64)
    adapter.generate_action(obs)
    sent = transport.requests[-1]
    assert sent["prompt"] == "updated instruction"
    assert np.allclose(sent["observation/state"], np.arange(8, dtype=np.float32))


def test_registry_creates_libero_adapter():
    policy_config = {
        "type": "kai0_libero",
        "server_url": "ws://127.0.0.1:8000",
        "instruction": "",
        "auto_start": False,
        "contract": "libero",
    }
    adapter = create_policy_adapter(policy_config)
    assert isinstance(adapter, Kai0SyncAdapter)
    assert adapter.required_image_keys == contract.LIBERO.required_image_keys()


def test_dual_arm_contract_unchanged_smoke():
    # 双臂契约回归锚点：值/行为不变
    assert contract.DUAL_ARM.action_dim == 14
    assert contract.GLOBAL_JOINT_ORDER == contract.DUAL_ARM.joint_names
    assert contract.validate_action(np.zeros(14)).shape == (14,)
    out = contract.grippers_to_server_units(np.ones(14))
    assert out[6] == 1000.0 and out[13] == 1000.0 and out[0] == 1.0


# —— JointCommand 构造（main._build_joint_command 的 LIBERO 路径） ——


def test_build_joint_command_libero_7d():
    from kai0_inference.inference.main import _build_joint_command

    config = PolicyNodeConfig.from_dict(_valid_dict())
    cmd = _build_joint_command(np.zeros(7), config)
    assert list(cmd.name) == list(contract.LIBERO.joint_names)
    assert list(cmd.position) == [0.0] * 7
    # velocity/effort 恒零（joints 全部 mode=position 构造保证——增强 D 替代口径）
    assert list(cmd.velocity) == [0.0] * 7
    assert list(cmd.effort) == [0.0] * 7


def test_build_joint_command_libero_rejects_wrong_dim():
    from kai0_inference.inference.main import _build_joint_command

    config = PolicyNodeConfig.from_dict(_valid_dict())
    for dim in (8, 14):
        with pytest.raises(ValueError, match="ACTION_DIM=7"):
            _build_joint_command(np.zeros(dim), config)
