"""RTC 动作流缓冲：新动作块裁剪已流逝前缀后与队列剩余轨迹线性混合。

语义与真机部署口径一致，适配 kai0_runner 契约：

- 缓冲内存储 **Forge 单位** 动作（关节弧度、夹爪米）；发送 prev_action_chunk 时
  由 RTC adapter 组包换算回服务端毫米（契约见 inference/contract.py）。
- integrate_new_chunk：按「已流逝控制步数」裁剪新块前缀（上限 max_trim_steps），
  剩余部分与旧队列重叠段线性混合——关节维度交叉淡化（连续轨迹），夹爪维度
  直接取新块值（抓取指令不被平均延迟）。
- pop_next_action / snapshot_future：消费线程（dora 主线程）与推理线程经
  Condition 同步；snapshot 截取队列未来轨迹并 pad 到目标 horizon，作为下一请求
  的 RTC 引导前缀 prev_action_chunk。
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from kai0_inference.inference import contract


class StreamActionBuffer:
    """线程安全的动作流缓冲（RTC 延迟裁剪 + 块重叠混合）。"""

    def __init__(self, state_dim: int = contract.ACTION_DIM) -> None:
        self.state_dim = int(state_dim)
        self._actions: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._last_action: np.ndarray | None = None
        self._steps_since_integrate = 0
        self._generation = 0
        self._last_popped_generation = 0
        self._closed = False

    # —— 动作布局（%7==6 为夹爪维度，契约 contract.GRIPPER_INDICES） ——

    @staticmethod
    def _joint_indices(action_dim: int) -> np.ndarray:
        return np.asarray(
            [index for index in range(action_dim) if index % 7 != 6], dtype=np.int64
        )

    @staticmethod
    def _gripper_indices(action_dim: int) -> np.ndarray:
        return np.asarray(
            [index for index in range(action_dim) if index % 7 == 6], dtype=np.int64
        )

    def integrate_new_chunk(
        self,
        actions_chunk: np.ndarray,
        *,
        max_trim_steps: int,
        min_blend_steps: int,
        elapsed_control_steps: int | None = None,
    ) -> int:
        """裁剪已流逝前缀并把重叠段混合进当前轨迹；返回实际裁剪步数。

        关节维度线性交叉淡化；夹爪维度始终取新块（对齐的）值。
        """
        actions = np.asarray(actions_chunk, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[0] == 0:
            raise ValueError(f"Expected non-empty [horizon, action_dim] chunk, got {actions.shape}")
        if actions.shape[1] != self.state_dim:
            raise ValueError(
                f"Action dim mismatch: buffer expects {self.state_dim}, got {actions.shape[1]}"
            )

        with self._condition:
            if self._closed:
                return 0

            elapsed_steps = (
                self._steps_since_integrate
                if elapsed_control_steps is None
                else max(0, int(elapsed_control_steps))
            )
            trim_steps = min(
                elapsed_steps,
                max(0, int(max_trim_steps)),
                max(0, actions.shape[0] - 1),
            )
            new_actions = [row.copy() for row in actions[trim_steps:]]
            old_actions = [row.copy() for row in self._actions]
            blend_steps = max(1, int(min_blend_steps))

            if not old_actions and self._last_action is not None:
                # 队列耗尽后的新块：以上一步动作为参考起点混合，保证轨迹连续。
                old_actions = [self._last_action.copy() for _ in range(blend_steps)]
            elif old_actions and len(old_actions) < blend_steps:
                old_actions.extend(
                    old_actions[-1].copy() for _ in range(blend_steps - len(old_actions))
                )

            if old_actions:
                overlap_len = min(len(old_actions), len(new_actions), blend_steps)
                joint_indices = self._joint_indices(self.state_dim)
                gripper_indices = self._gripper_indices(self.state_dim)
                if overlap_len == 1:
                    old_weights = np.asarray([1.0], dtype=np.float64)
                else:
                    old_weights = np.linspace(1.0, 0.0, overlap_len, dtype=np.float64)

                blended: list[np.ndarray] = []
                for index in range(overlap_len):
                    old_action = np.asarray(old_actions[index], dtype=np.float64)
                    new_action = np.asarray(new_actions[index], dtype=np.float64)
                    merged = new_action.copy()
                    old_weight = float(old_weights[index])
                    merged[joint_indices] = (
                        old_weight * old_action[joint_indices]
                        + (1.0 - old_weight) * new_action[joint_indices]
                    )
                    merged[gripper_indices] = new_action[gripper_indices]
                    blended.append(merged)
                combined = blended + new_actions[overlap_len:]
            else:
                combined = new_actions

            self._actions = deque(combined)
            self._steps_since_integrate = 0
            self._generation += 1
            self._condition.notify_all()
            return trim_steps

    def pop_next_action(
        self,
        *,
        wait_timeout_s: float = 0.0,
    ) -> tuple[np.ndarray | None, bool]:
        """弹出一个动作；第二个返回值为「本次是否新并入块的首个动作」。"""
        deadline = time.monotonic() + max(0.0, float(wait_timeout_s))
        with self._condition:
            while not self._actions and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)

            if not self._actions:
                return None, False

            action = np.asarray(self._actions.popleft(), dtype=np.float64)
            self._last_action = action.copy()
            self._steps_since_integrate += 1
            is_new_generation = self._last_popped_generation != self._generation
            self._last_popped_generation = self._generation
            return action, is_new_generation

    def remaining(self) -> int:
        with self._lock:
            return len(self._actions)

    def snapshot_future(self, horizon: int) -> np.ndarray | None:
        """复制队列中的未来执行轨迹并 pad 到 ``horizon``（不足以末行补齐）。"""
        requested_horizon = int(horizon)
        if requested_horizon <= 0:
            raise ValueError("snapshot horizon must be positive.")

        with self._lock:
            queued = [
                np.asarray(action, dtype=np.float64).copy()
                for action in list(self._actions)[:requested_horizon]
            ]
            if queued:
                pad_action = queued[-1]
            elif self._last_action is not None:
                pad_action = self._last_action.copy()
            else:
                return None

            queued.extend(
                pad_action.copy() for _ in range(requested_horizon - len(queued))
            )
            return np.stack(queued, axis=0)

    def clear(self) -> None:
        with self._condition:
            self._actions.clear()
            self._last_action = None
            self._steps_since_integrate = 0
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
