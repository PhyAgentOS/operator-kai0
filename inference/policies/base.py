"""kai0 策略适配器基类。

实现 forge_policy 的 PolicyAdapter 协议（is_observation_needed / generate_action）。
WS 同步 Adapter（kai0/adapter.py）组合传输层（可注入 fake server）与本基类，
传输层与策略层分离；DummyAdapter（dummy.py）提供无服务端链路的调试实现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Kai0PolicyAdapter(ABC):
    """kai0 策略适配器抽象基类。

    生命周期钩子（start / pause / stop / reset_scene / set_instruction）可选实现，
    由 forge_policy.run_dora_policy_node 在 call_lifecycle_hooks=True 时调用。
    """

    @abstractmethod
    def reset(self) -> None:
        """重置内部状态（同步 Adapter 为动作块队列等）。"""

    @abstractmethod
    def is_observation_needed(self) -> bool:
        """是否需要新观测。同步模式下恒为 True。"""

    @property
    def required_image_keys(self) -> frozenset[str] | None:
        """需要的图像观测键；None 表示使用配置的全部图像输入。"""
        return None

    @abstractmethod
    def generate_action(
        self,
        observation: dict[str, Any],
        alias_for_cameras: list[str] | None = None,
    ) -> np.ndarray | None:
        """返回一个 14 维动作向量（契约见 inference/contract.py）；None 表示暂不可用。"""
