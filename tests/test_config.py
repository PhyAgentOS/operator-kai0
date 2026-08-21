"""配置加载与契约校验单测。"""

from __future__ import annotations

from pathlib import Path

import pytest

from kai0_inference.inference import contract
from kai0_inference.inference.config import PolicyNodeConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
KAI0_YAML = REPO_ROOT / "config" / "inference" / "kai0.yaml"
DUMMY_YAML = REPO_ROOT / "config" / "inference" / "dummy.yaml"


def test_load_bundled_kai0_yaml():
    config = load_config(KAI0_YAML)
    assert len(config.joints) == 14
    assert config.joint_order == list(contract.GLOBAL_JOINT_ORDER)
    assert config.state_joint_order == list(contract.GLOBAL_JOINT_ORDER)
    assert config.policy["type"] == "kai0"
    assert config.auto_start is False


def test_gripper_joint_modes():
    config = load_config(KAI0_YAML)
    for i in contract.GRIPPER_INDICES:
        joint = config.joints[i]
        assert joint.mode == "prismatic"
        assert joint.unit == "meters"
    for i, joint in enumerate(config.joints):
        if i not in contract.GRIPPER_INDICES:
            assert joint.mode == "position"
            assert joint.unit == "radians"


def test_image_inputs_alias_contract():
    config = load_config(KAI0_YAML)
    assert set(config.alias_for_cameras) == set(contract.CAMERA_ALIASES)
    assert config.image_input_id_to_alias == contract.DEFAULT_IMAGE_INPUT_ID_TO_ALIAS


def test_runtime_policy_config():
    config = load_config(KAI0_YAML)
    runtime = config.runtime_policy_config()
    assert runtime["action_joint_count"] == 14
    assert runtime["state_joint_count"] == 14
    assert runtime["camera_names"] == config.alias_for_cameras
    assert runtime["action_joint_names"] == list(contract.GLOBAL_JOINT_ORDER)
    assert runtime["auto_start"] is False
    assert runtime["server_url"] == "ws://127.0.0.1:8000"


def test_load_dummy_yaml():
    config = load_config(DUMMY_YAML)
    assert config.policy["type"] == "dummy"
    assert config.joint_order == list(contract.GLOBAL_JOINT_ORDER)


def _valid_dict() -> dict:
    return {
        "joints": [{"name": n} for n in contract.GLOBAL_JOINT_ORDER],
        "policy": {"type": "kai0"},
        "image_inputs": dict(contract.DEFAULT_IMAGE_INPUT_ID_TO_ALIAS),
    }


def test_from_dict_rejects_contract_violations():
    # 关节顺序错误
    data = _valid_dict()
    data["joints"][0], data["joints"][7] = data["joints"][7], data["joints"][0]
    with pytest.raises(ValueError, match="joint_names 顺序一致"):
        PolicyNodeConfig.from_dict(data)

    # 缺 policy.type
    data = _valid_dict()
    data["policy"].pop("type")
    with pytest.raises(ValueError, match="type"):
        PolicyNodeConfig.from_dict(data)

    # alias 集合与契约不一致
    data = _valid_dict()
    data["image_inputs"] = {"image/top": "top_head", "image/left": "hand_left"}
    with pytest.raises(ValueError, match="camera_aliases"):
        PolicyNodeConfig.from_dict(data)

    # alias 重复
    data = _valid_dict()
    data["image_inputs"] = {
        "image/top": "top_head",
        "image/left": "top_head",
        "image/right": "hand_right",
    }
    with pytest.raises(ValueError, match="唯一"):
        PolicyNodeConfig.from_dict(data)


def test_from_dict_minimal_valid():
    config = PolicyNodeConfig.from_dict(_valid_dict())
    assert config.joint_order == list(contract.GLOBAL_JOINT_ORDER)
    assert config.alias_for_cameras == list(contract.CAMERA_ALIASES)


def test_image_inputs_for_filtering():
    config = PolicyNodeConfig.from_dict(_valid_dict())
    assert config.image_inputs_for(None) == contract.DEFAULT_IMAGE_INPUT_ID_TO_ALIAS
    filtered = config.image_inputs_for({"observation.images.top_head"})
    assert filtered == {"image/top": "top_head"}
