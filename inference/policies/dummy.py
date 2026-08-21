"""DummyAdapter：不经 WS 的占位适配器，返回固定安全动作。

用途：在无服务端链路的场景下验证节点链路（infer-once / dataflow 启动），
也用于隔离排查服务端相关问题时确认节点侧正常。
固定动作满足契约：14 维、关节 0、夹爪 0.03 m（在 0..0.105 m 量程内）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kai0_inference.inference import contract
from kai0_inference.inference.policies.base import Kai0PolicyAdapter


class DummyAdapter(Kai0PolicyAdapter):
    """返回固定 14 维动作的占位适配器；有服务端链路时用 Kai0SyncAdapter 取代。"""

    def __init__(
        self,
        *,
        joint_position: float = contract.DUMMY_JOINT_POSITION,
        gripper_open_m: float = contract.DUMMY_GRIPPER_OPEN_M,
    ) -> None:
        # 夹爪值必须落在契约量程内，构造即校验，越界直接报错
        low, high = contract.GRIPPER_RANGE_M
        if not (low <= gripper_open_m <= high):
            raise ValueError(
                f"dummy gripper_open_m={gripper_open_m} 超出夹爪量程 "
                f"{contract.GRIPPER_RANGE_M}（米）"
            )
        self._action = np.full(contract.ACTION_DIM, float(joint_position), dtype=np.float64)
        for i in contract.GRIPPER_INDICES:
            self._action[i] = float(gripper_open_m)

    def reset(self) -> None:
        """无内部状态，无需重置。"""

    def is_observation_needed(self) -> bool:
        # 同步语义：每次 tick 都需要新观测
        return True

    def generate_action(
        self,
        observation: dict[str, Any],
        alias_for_cameras: list[str] | None = None,
    ) -> np.ndarray:
        # 返回副本，避免调用方改动污染内部状态
        return self._action.copy()
