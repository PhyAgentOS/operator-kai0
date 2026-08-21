"""可配置的动作生成源：fake server 用假动作替代真实模型输出。"""

from __future__ import annotations

import abc
import math

import numpy as np


class ActionSource(abc.ABC):
    """按步数生成一个动作块（chunk_size × action_dim）的假模型输出。"""

    @abc.abstractmethod
    def __call__(self, step: int, *, action_dim: int, chunk_size: int) -> np.ndarray:
        ...


class ZeroActionSource(ActionSource):
    """全零动作，用于最基础的链路验证。"""

    def __call__(self, step: int, *, action_dim: int, chunk_size: int) -> np.ndarray:
        return np.zeros((chunk_size, action_dim), dtype=np.float32)


class SineActionSource(ActionSource):
    """每个关节输出不同频率的正弦波，用于目视验证动作顺序映射。

    约定与 kai0 14 维布局一致：索引 ``%7 == 6`` 为夹爪维度，其余为关节维度。
    """

    JOINT_AMPLITUDE_RAD = 0.3
    GRIPPER_BASE_MM = 30.0
    GRIPPER_AMPLITUDE_MM = 20.0
    GRIPPER_PERIOD_STEPS = 200

    def __call__(self, step: int, *, action_dim: int, chunk_size: int) -> np.ndarray:
        actions = np.zeros((chunk_size, action_dim), dtype=np.float32)
        for t in range(chunk_size):
            s = step + t
            for j in range(action_dim):
                if j % 7 == 6:
                    # 夹爪维度：慢速往复，左右夹爪相位错开，与关节维度明显区分。
                    phase = math.pi if j >= 7 else 0.0
                    actions[t, j] = self.GRIPPER_BASE_MM + self.GRIPPER_AMPLITUDE_MM * math.sin(
                        2 * math.pi * s / self.GRIPPER_PERIOD_STEPS + phase
                    )
                else:
                    # 关节维度：按绝对索引取频率与相位，保证 14 维两两不同，顺序错位一眼可见。
                    period = 20.0 + 5.0 * j
                    phase = 2 * math.pi * j / action_dim
                    actions[t, j] = self.JOINT_AMPLITUDE_RAD * math.sin(2 * math.pi * s / period + phase)
        return actions.astype(np.float32)
