"""观测组包 / 响应拆包共享逻辑（同步与 RTC adapter 共用）。

组包：Forge 观测 → 服务端请求 payload；拆包：服务端响应 actions → [H, D]
float64（服务端单位）。契约 profile 见 inference/contract.py（PolicyContract）：

- dual_arm：state 夹爪 米→毫米、图像 repack 为 uint8 HWC、payload =
  {"state", "images": {alias: ...}, "prompt"}；
- libero：state/夹爪原样透传、payload = {"observation/image", "observation/wrist_image",
  "observation/state", "prompt"}（openpi LiberoInputs 直读键形，与官方 openpi-client 一致）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kai0_inference.inference import contract


def build_payload(
    observation: dict[str, Any],
    instruction: str,
    policy_contract: contract.PolicyContract = contract.DUAL_ARM,
) -> dict[str, Any]:
    """把 Forge 观测组包为服务端请求 payload（state/单位换算、图像 repack、prompt）。"""
    state_value = observation.get("observation.state")
    if state_value is None:
        raise ValueError("kai0 policy requires 'observation.state' in observation")
    state = policy_contract.to_server_state(state_value)

    images: dict[str, np.ndarray] = {}
    for alias in policy_contract.camera_aliases:
        key = f"observation.images.{alias}"
        if key not in observation:
            raise ValueError(f"observation missing required image key {key!r}")
        images[alias] = _as_uint8_image(observation[key], alias)

    if policy_contract.server_payload_style == "flat_observation_keys":
        # LIBERO：LiberoInputs 直读 observation/image 与 observation/wrist_image
        return {
            "observation/state": state,
            **{f"observation/{alias}": img for alias, img in images.items()},
            "prompt": str(instruction),
        }
    # dual_arm：{"state", "images": {alias: ...}, "prompt"}
    return {
        "state": state,
        "images": images,
        "prompt": str(instruction),
    }


def parse_actions(
    result: dict[str, Any],
    policy_contract: contract.PolicyContract = contract.DUAL_ARM,
) -> np.ndarray:
    """解析服务端响应的 actions 字段为 [H, action_dim] float64 数组（未做单位换算）。"""
    actions = result.get("actions")
    if actions is None:
        raise ValueError(f"server response missing 'actions': {sorted(result)}")
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"unexpected server actions shape: {np.asarray(actions).shape}")
    if arr.shape[1] != policy_contract.action_dim:
        raise ValueError(
            f"server actions width={arr.shape[1]} does not match "
            f"ACTION_DIM={policy_contract.action_dim}（contract {policy_contract.name}）"
        )
    if not np.isfinite(arr).all():
        raise ValueError("server actions must contain only finite values")
    return arr


def _as_uint8_image(value: Any, alias: str) -> np.ndarray:
    """归一化图像为 uint8 HWC（服务端契约），仅支持 uint8 或 [0,1] float 输入。"""
    arr = np.asarray(value)
    if arr.ndim != 3:
        raise ValueError(f"image {alias!r} must be HWC or CHW 3-D, got shape={arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = np.rint(arr).astype(np.uint8)
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    arr = arr.astype(np.uint8)
    if arr.shape[-1] != 3:
        raise ValueError(f"image {alias!r} must have 3 channels, got shape={arr.shape}")
    return arr
