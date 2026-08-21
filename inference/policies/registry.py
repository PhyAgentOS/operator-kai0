"""kai0 策略适配器注册表。

已注册类型：
- kai0：同步模式 WS 客户端（Kai0SyncAdapter，短连接 + chunk 摊销；双臂契约）；
- kai0_libero：同步模式 WS 客户端 + LIBERO 契约（Kai0SyncAdapter(policy_contract=LIBERO)，
  7 维动作 / 8 维 state / 两路相机 / LiberoInputs 平铺键 payload）；
- kai0_rtc：RTC 异步推理 WS 客户端（Kai0RtcAdapter，常驻连接 + 后台推理线程；
  服务端需以 _rtc 配置启动，权重与非 RTC 共用同一 checkpoint；双臂契约）；
- dummy：不经 WS 返回固定动作（DummyAdapter，无服务端链路调试用；双臂契约）。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from kai0_inference.inference import contract
from kai0_inference.inference.policies.base import Kai0PolicyAdapter
from kai0_inference.inference.policies.dummy import DummyAdapter
from kai0_inference.inference.policies.kai0.adapter import Kai0SyncAdapter
from kai0_inference.inference.policies.kai0.rtc_adapter import Kai0RtcAdapter
from kai0_inference.inference.transport.ws_client import WsActionTransport

PolicyFactory = Callable[[dict[str, Any]], Kai0PolicyAdapter]

logger = logging.getLogger(__name__)

_POLICY_ALIASES: dict[str, str] = {
    "kai0": "kai0",
    "KAI0": "kai0",
    "kai0_libero": "kai0_libero",
    "KAI0_LIBERO": "kai0_libero",
    "kai0_rtc": "kai0_rtc",
    "KAI0_RTC": "kai0_rtc",
    "dummy": "dummy",
}

# Implemented adapters keyed by normalized policy type.
_IMPLEMENTED: dict[str, PolicyFactory] = {}


def normalize_policy_type(raw: str) -> str:
    key = raw.strip()
    if key not in _POLICY_ALIASES:
        raise ValueError(
            f"Unsupported policy.type: {raw!r}. "
            f"Known aliases: {sorted(_POLICY_ALIASES)}"
        )
    return _POLICY_ALIASES[key]


def register_policy_type(policy_type: str, factory: PolicyFactory) -> None:
    normalized = normalize_policy_type(policy_type)
    _IMPLEMENTED[normalized] = factory


def supported_policy_types() -> list[str]:
    return sorted(_IMPLEMENTED)


def _dummy_overrides(policy_config: dict[str, Any]) -> dict[str, Any]:
    """读取可选 policy.dummy 覆盖项（joint_position / gripper_open_m）。"""
    dummy = policy_config.get("dummy") or {}
    if not isinstance(dummy, dict):
        raise ValueError("policy.dummy must be a mapping")
    unknown = set(dummy) - {"joint_position", "gripper_open_m"}
    if unknown:
        raise ValueError(f"Unknown policy.dummy keys: {sorted(unknown)}")
    return {
        key: float(dummy[key])
        for key in ("joint_position", "gripper_open_m")
        if key in dummy
    }


def _create_kai0(policy_config: dict[str, Any]) -> Kai0PolicyAdapter:
    server_url = policy_config.get("server_url")
    if not server_url:
        raise ValueError("policy.server_url is required for kai0 (e.g. ws://127.0.0.1:8000)")
    transport = WsActionTransport(
        server_url,
        timeout_s=float(policy_config.get("timeout_s", 60.0)),
    )
    return Kai0SyncAdapter(
        transport,
        instruction=str(policy_config.get("instruction", "") or ""),
        execute_action_steps=policy_config.get("execute_action_steps"),
    )


def _create_kai0_libero(policy_config: dict[str, Any]) -> Kai0PolicyAdapter:
    """kai0_libero：同步 WS 客户端 + LIBERO 契约（7 维动作 / 8 维 state / 两路相机）。"""
    server_url = policy_config.get("server_url")
    if not server_url:
        raise ValueError("policy.server_url is required for kai0_libero (e.g. ws://127.0.0.1:8000)")
    transport = WsActionTransport(
        server_url,
        timeout_s=float(policy_config.get("timeout_s", 60.0)),
    )
    return Kai0SyncAdapter(
        transport,
        instruction=str(policy_config.get("instruction", "") or ""),
        execute_action_steps=policy_config.get("execute_action_steps"),
        policy_contract=contract.LIBERO,
    )


# Kai0RtcAdapter 支持的 rtc_* 配置键（默认值在 rtc_adapter.py 中定义，
# 对齐真机客户端 YAML；此处只透传配置中显式给出的键）。
_RTC_KEYS = (
    "rtc_control_rate",
    "rtc_chunk_size",
    "rtc_execute_horizon",
    "rtc_latency_k",
    "rtc_min_smooth_steps",
    "rtc_inference_hz",
    "rtc_stale_timeout_s",
    "rtc_startup_timeout_s",
    "rtc_delay_window",
    "rtc_initial_delay_steps",
)


def _create_kai0_rtc(policy_config: dict[str, Any]) -> Kai0PolicyAdapter:
    server_url = policy_config.get("server_url")
    if not server_url:
        raise ValueError("policy.server_url is required for kai0_rtc (e.g. ws://127.0.0.1:8000)")
    return Kai0RtcAdapter(
        server_url=server_url,
        instruction=str(policy_config.get("instruction", "") or ""),
        timeout_s=float(policy_config.get("timeout_s", 60.0)),
        **{key: policy_config[key] for key in _RTC_KEYS if key in policy_config},
    )


def _create_dummy(policy_config: dict[str, Any]) -> Kai0PolicyAdapter:
    return DummyAdapter(**_dummy_overrides(policy_config))


register_policy_type("kai0", _create_kai0)
register_policy_type("kai0_libero", _create_kai0_libero)
register_policy_type("kai0_rtc", _create_kai0_rtc)
register_policy_type("dummy", _create_dummy)


def create_policy_adapter(policy_config: dict[str, Any]) -> Kai0PolicyAdapter:
    """从运行时 policy 配置实例化已注册的适配器。"""
    policy_type = normalize_policy_type(str(policy_config["type"]))
    factory = _IMPLEMENTED[policy_type]
    return factory(policy_config)
