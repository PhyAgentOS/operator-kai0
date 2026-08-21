"""registry 单测：kai0 → Kai0SyncAdapter；kai0_rtc → Kai0RtcAdapter；dummy → DummyAdapter。"""

from __future__ import annotations

import pytest

from kai0_inference.inference.policies.dummy import DummyAdapter
from kai0_inference.inference.policies.kai0.adapter import Kai0SyncAdapter
from kai0_inference.inference.policies.kai0.rtc_adapter import Kai0RtcAdapter
from kai0_inference.inference.policies.registry import (
    create_policy_adapter,
    normalize_policy_type,
    supported_policy_types,
)


def test_supported_types():
    assert supported_policy_types() == ["dummy", "kai0", "kai0_libero", "kai0_rtc"]


def test_create_kai0_requires_server_url():
    with pytest.raises(ValueError, match="server_url"):
        create_policy_adapter({"type": "kai0"})


def test_create_kai0_returns_sync_adapter():
    adapter = create_policy_adapter({"type": "kai0", "server_url": "ws://127.0.0.1:8000"})
    assert isinstance(adapter, Kai0SyncAdapter)


def test_create_kai0_rtc_requires_server_url():
    with pytest.raises(ValueError, match="server_url"):
        create_policy_adapter({"type": "kai0_rtc"})


def test_create_kai0_rtc_returns_rtc_adapter():
    adapter = create_policy_adapter({"type": "kai0_rtc", "server_url": "ws://127.0.0.1:8000"})
    try:
        assert isinstance(adapter, Kai0RtcAdapter)
    finally:
        adapter.stop()


def test_create_kai0_rtc_forwards_rtc_params():
    adapter = create_policy_adapter(
        {
            "type": "kai0_rtc",
            "server_url": "ws://127.0.0.1:8000",
            "rtc_chunk_size": 10,
            "rtc_execute_horizon": 5,
            "rtc_control_rate": 50,
        }
    )
    try:
        assert adapter._rtc_chunk_size == 10
        assert adapter._rtc_execute_horizon == 5
        assert adapter._rtc_control_rate == 50.0
    finally:
        adapter.stop()


def test_create_dummy_returns_dummy_adapter():
    adapter = create_policy_adapter({"type": "dummy"})
    assert isinstance(adapter, DummyAdapter)


def test_normalize_aliases():
    assert normalize_policy_type("kai0") == "kai0"
    assert normalize_policy_type("KAI0") == "kai0"
    assert normalize_policy_type("kai0_rtc") == "kai0_rtc"
    assert normalize_policy_type("KAI0_RTC") == "kai0_rtc"
    assert normalize_policy_type("dummy") == "dummy"
    with pytest.raises(ValueError):
        normalize_policy_type("pi05")


def test_dummy_overrides_applied():
    adapter = create_policy_adapter(
        {"type": "dummy", "dummy": {"joint_position": 0.1, "gripper_open_m": 0.02}}
    )
    action = adapter.generate_action({})
    assert action[0] == 0.1
    assert action[6] == 0.02
    assert action[13] == 0.02


def test_dummy_overrides_out_of_range_rejected():
    with pytest.raises(ValueError, match="夹爪量程"):
        create_policy_adapter({"type": "dummy", "dummy": {"gripper_open_m": 0.2}})


def test_dummy_unknown_key_rejected():
    with pytest.raises(ValueError, match="Unknown policy.dummy keys"):
        create_policy_adapter({"type": "dummy", "dummy": {"bogus": 1}})


def test_dummy_not_a_mapping_rejected():
    with pytest.raises(ValueError, match="policy.dummy"):
        create_policy_adapter({"type": "dummy", "dummy": [1, 2]})
