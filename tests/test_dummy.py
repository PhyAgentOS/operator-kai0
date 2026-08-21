"""DummyAdapter 行为单测：固定动作维度、数值安全、协议接口。"""

from __future__ import annotations

import numpy as np
import pytest

from kai0_inference.inference import contract
from kai0_inference.inference.policies.dummy import DummyAdapter


def test_generate_action_shape_and_dtype():
    adapter = DummyAdapter()
    action = adapter.generate_action({})
    assert isinstance(action, np.ndarray)
    assert action.shape == (contract.ACTION_DIM,)
    assert np.issubdtype(action.dtype, np.floating)


def test_fixed_action_values_within_contract():
    adapter = DummyAdapter()
    action = adapter.generate_action({})
    # 关节维度固定 0
    for i, name in enumerate(contract.GLOBAL_JOINT_ORDER):
        if i not in contract.GRIPPER_INDICES:
            assert action[i] == contract.DUMMY_JOINT_POSITION, name
    # 夹爪维度在量程内
    for i in contract.GRIPPER_INDICES:
        assert action[i] == contract.DUMMY_GRIPPER_OPEN_M
    low, high = contract.GRIPPER_RANGE_M
    for i in contract.GRIPPER_INDICES:
        assert low <= action[i] <= high
    # 整体通过契约校验
    validated = contract.validate_action(action)
    assert validated.shape == (contract.ACTION_DIM,)
    assert contract.gripper_range_violations(action) == []


def test_returned_action_is_a_copy():
    adapter = DummyAdapter()
    first = adapter.generate_action({})
    first[0] = 999.0
    second = adapter.generate_action({})
    assert second[0] == contract.DUMMY_JOINT_POSITION


def test_protocol_interface():
    adapter = DummyAdapter()
    assert adapter.is_observation_needed() is True
    assert adapter.required_image_keys is None
    adapter.reset()  # 无内部状态，不应抛错


def test_constructor_rejects_out_of_range_gripper():
    with pytest.raises(ValueError, match="夹爪量程"):
        DummyAdapter(gripper_open_m=0.2)
    with pytest.raises(ValueError, match="夹爪量程"):
        DummyAdapter(gripper_open_m=-0.01)
