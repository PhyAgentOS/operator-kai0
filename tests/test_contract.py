"""契约常量单测：14 维布局、夹爪索引与量程、相机 alias、动作校验。"""

from __future__ import annotations

import numpy as np
import pytest

from kai0_inference.inference import contract


def test_action_dim_is_14():
    assert contract.ACTION_DIM == 14
    assert len(contract.GLOBAL_JOINT_ORDER) == 14


def test_joint_order_layout():
    """左 6 关节+左夹爪、右 6 关节+右夹爪，与 kai0 服务端 / robot_router 一致。"""
    expected = (
        [f"robot_left/{n}" for n in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")]
        + ["robot_left/gripper"]
        + [f"robot_right/{n}" for n in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")]
        + ["robot_right/gripper"]
    )
    assert list(contract.GLOBAL_JOINT_ORDER) == expected


def test_gripper_indices_mod7_eq6():
    """夹爪索引为 6、13，满足 idx % 7 == 6（服务端布局）。"""
    assert contract.GRIPPER_INDICES == {6, 13}
    for i in contract.GRIPPER_INDICES:
        assert i % 7 == 6
    for i, name in enumerate(contract.GLOBAL_JOINT_ORDER):
        assert (name.endswith("/gripper")) == (i in contract.GRIPPER_INDICES)


def test_gripper_range_meters():
    assert contract.GRIPPER_RANGE_M == (0.0, 0.105)


def test_dummy_values_within_contract():
    low, high = contract.GRIPPER_RANGE_M
    assert low <= contract.DUMMY_GRIPPER_OPEN_M <= high


def test_camera_aliases_and_default_inputs():
    assert contract.CAMERA_ALIASES == ("top_head", "hand_left", "hand_right")
    assert contract.IMAGE_HEIGHT == 224
    assert contract.IMAGE_WIDTH == 224
    assert set(contract.DEFAULT_IMAGE_INPUT_ID_TO_ALIAS.values()) == set(
        contract.CAMERA_ALIASES
    )


def test_validate_action_accepts_valid():
    action = np.zeros(contract.ACTION_DIM)
    out = contract.validate_action(action)
    assert out.shape == (contract.ACTION_DIM,)
    assert np.isfinite(out).all()


def test_validate_action_rejects_bad_input():
    with pytest.raises(ValueError):
        contract.validate_action(np.zeros(7))  # 维度不符
    with pytest.raises(ValueError):
        contract.validate_action(np.zeros((2, 7)))  # 非 1-D
    bad = np.zeros(contract.ACTION_DIM)
    bad[0] = np.nan
    with pytest.raises(ValueError):
        contract.validate_action(bad)  # 非有限


def test_gripper_range_violations():
    ok = np.zeros(contract.ACTION_DIM)
    ok[6] = 0.03
    ok[13] = 0.105
    assert contract.gripper_range_violations(ok) == []

    out = np.zeros(contract.ACTION_DIM)
    out[6] = -0.01
    out[13] = 0.2
    violations = contract.gripper_range_violations(out)
    assert sorted(v[0] for v in violations) == [6, 13]
    assert all(v[1].endswith("/gripper") for v in violations)


def test_gripper_unit_conversion_1d():
    """1-D：夹爪 米↔毫米 换算，关节维度原样透传。"""
    value = np.full(contract.ACTION_DIM, 0.5, dtype=np.float64)
    to_server = contract.grippers_to_server_units(value)
    for i in contract.GRIPPER_INDICES:
        assert to_server[i] == pytest.approx(500.0)
    for i in range(contract.ACTION_DIM):
        if i not in contract.GRIPPER_INDICES:
            assert to_server[i] == pytest.approx(0.5)

    roundtrip = contract.grippers_from_server_units(to_server)
    assert roundtrip[6] == pytest.approx(0.5)


def test_gripper_unit_conversion_2d():
    """2-D 动作块：夹爪列换算，关节列原样透传。"""
    chunk = np.full((3, contract.ACTION_DIM), 1.0, dtype=np.float64)
    back = contract.grippers_from_server_units(contract.grippers_to_server_units(chunk))
    np.testing.assert_allclose(back, chunk)
    # 输入不被就地修改
    assert np.all(chunk == 1.0)


def test_gripper_unit_conversion_rejects_bad_ndim():
    with pytest.raises(ValueError, match="1-D or 2-D"):
        contract.grippers_to_server_units(np.zeros((2, 2, 2)))
