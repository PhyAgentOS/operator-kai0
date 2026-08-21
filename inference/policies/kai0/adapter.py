"""Kai0SyncAdapter：同步模式 WS 客户端适配器。

- 传输层与策略层分离：本类只依赖 ActionTransport 抽象（实现见
  inference/transport/，真实 socket 连服务端）。
- chunk 摊销：服务端一次返回 [H, D] 动作块，本地队列逐步消耗；
  队列非空时 is_observation_needed() 为 False，耗尽才重新请求。
- 契约 profile（见 inference/contract.py 的 PolicyContract）：默认 DUAL_ARM
  （14 维双臂，观测 state 夹爪 米→毫米 组包、服务端动作块夹爪 毫米→米 拆包）；
  kai0_libero 注入 LIBERO（7 维动作、8 维 state、两路相机、夹爪原样透传、
  LiberoInputs 平铺键 payload）。
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

import numpy as np

from kai0_inference.inference import contract
from kai0_inference.inference.policies.base import Kai0PolicyAdapter
from kai0_inference.inference.policies.kai0 import payloads
from kai0_inference.inference.transport.base import ActionTransport

logger = logging.getLogger(__name__)


class _ActionStats:
    """执行动作的逐维分布统计（口径判据 S-3 诊断：min/max/mean、夹爪正负、饱和/恒零）。"""

    def __init__(self) -> None:
        self.count = 0
        self.mins: np.ndarray | None = None
        self.maxs: np.ndarray | None = None
        self.sums: np.ndarray | None = None
        self.saturated = 0  # |x| > 2 的维度计数（域外饱和信号）

    def record(self, action: np.ndarray) -> None:
        arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if self.count == 0:
            self.mins = arr.copy()
            self.maxs = arr.copy()
            self.sums = arr.copy()
        else:
            np.minimum(self.mins, arr, out=self.mins)
            np.maximum(self.maxs, arr, out=self.maxs)
            self.sums += arr
        self.saturated += int((np.abs(arr) > 2.0).sum())
        self.count += 1

    def summary(self) -> str:
        if self.count == 0:
            return "no actions"
        mins, maxs, means = self.mins, self.maxs, self.sums / self.count
        gripper_sign = f"grip_min={mins[-1]:+.3f} grip_max={maxs[-1]:+.3f}"
        return (
            f"count={self.count} "
            f"min={np.round(mins, 3).tolist()} "
            f"max={np.round(maxs, 3).tolist()} "
            f"mean={np.round(means, 3).tolist()} "
            f"{gripper_sign} saturated_dim_count={self.saturated}"
        )

    def clear(self) -> None:
        self.count = 0
        self.mins = self.maxs = self.sums = None
        self.saturated = 0


class Kai0SyncAdapter(Kai0PolicyAdapter):
    """同步模式 kai0 策略适配器：队列空则请求服务端动作块并摊销执行。

    契约 profile 可注入（默认 DUAL_ARM）：kai0_libero 用 contract.LIBERO
    （7 维动作、8 维 state、两路相机、夹爪不换算、LiberoInputs 平铺键 payload）。
    """

    def __init__(
        self,
        transport: ActionTransport,
        *,
        instruction: str = "",
        execute_action_steps: int | None = None,
        policy_contract: contract.PolicyContract = contract.DUAL_ARM,
    ) -> None:
        self._transport = transport
        self._instruction = str(instruction)
        self._contract = policy_contract
        if execute_action_steps is None:
            self._execute_action_steps = None
        else:
            if isinstance(execute_action_steps, bool) or not int(execute_action_steps) > 0:
                raise ValueError("execute_action_steps must be a positive integer or null")
            self._execute_action_steps = int(execute_action_steps)
        self._action_queue: deque[np.ndarray] = deque()
        self._request_count = 0
        self._action_stats = _ActionStats()
        self._episode_first_obs_logged = False

    # —— 指令读写（forge_policy 的 set_instruction 经 setattr(policy, "instruction", ...) 注入）——

    @property
    def instruction(self) -> str:
        """当前指令；逐 episode 由 policy_command set_instruction 更新。"""
        return self._instruction

    @instruction.setter
    def instruction(self, value: str) -> None:
        self._instruction = str(value)

    # —— Kai0PolicyAdapter 协议 ——

    @property
    def required_image_keys(self) -> frozenset[str]:
        """本契约需要的相机观测键（runner 以 observation.images.<alias> 提供）。"""
        return self._contract.required_image_keys()

    def reset(self) -> None:
        """清空本地动作队列；下一个 tick 将重新请求动作块。

        同时输出上一 episode 的动作分布诊断（口径判据 S-3）并重置统计——
        benchmark 每 episode 发 reset_scene，此处即 episode 边界。
        """
        self._action_queue.clear()
        self._log_episode_stats("episode_end")
        self._episode_first_obs_logged = False

    def stop(self) -> None:
        """批次末（policy_command stop）：输出最后 episode 的动作分布诊断。"""
        self._log_episode_stats("batch_end")

    def _log_episode_stats(self, marker: str) -> None:
        stats = self._action_stats
        if stats.count == 0:
            return
        summary = stats.summary()
        logger.info(
            "action_stats[%s] %s", marker, summary
        )
        stats.clear()

    def is_observation_needed(self) -> bool:
        # 同步语义：队列耗尽才需要新观测触发下一次请求
        return not self._action_queue

    def generate_action(
        self,
        observation: dict[str, Any],
        alias_for_cameras: list[str] | None = None,
    ) -> np.ndarray:
        if not self._episode_first_obs_logged and isinstance(observation, dict):
            state = observation.get("observation.state")
            if state is not None:
                # episode 边界观测（增强 A/B 风险探针）：首帧 state 前 3 维 + 指令
                logger.info(
                    "episode_first_obs instruction=%r state_head=%s",
                    self._instruction,
                    np.round(np.asarray(state, dtype=np.float64).reshape(-1)[:3], 4),
                )
                self._episode_first_obs_logged = True

        if self._action_queue:
            return self._action_queue.popleft()

        payload = self.build_payload(observation)
        request_start = time.perf_counter()
        result = self._transport.request_actions(payload)
        round_trip_ms = (time.perf_counter() - request_start) * 1000
        self._request_count += 1

        chunk = self._contract.from_server_actions(payloads.parse_actions(result, self._contract))
        if self._execute_action_steps is not None:
            chunk = chunk[: self._execute_action_steps]

        for row in chunk:
            self._action_queue.append(np.asarray(row, dtype=np.float64))

        server_timing = result.get("server_timing", {})
        infer_ms = server_timing.get("infer_ms")
        logger.info(
            "request=%d round_trip_ms=%.1f server_infer_ms=%s queued=%d",
            self._request_count,
            round_trip_ms,
            f"{float(infer_ms):.1f}" if isinstance(infer_ms, (int, float)) else "n/a",
            len(self._action_queue),
        )
        action = self._action_queue.popleft()
        self._action_stats.record(action)
        return action

    # —— 观测组包 / 响应拆包（公开以便单测直接校验） ——

    def build_payload(self, observation: dict[str, Any]) -> dict[str, Any]:
        """把 Forge 观测组包为服务端请求 payload（共享逻辑见 payloads.build_payload）。"""
        return payloads.build_payload(observation, self._instruction, self._contract)

    def _parse_actions(self, result: dict[str, Any]) -> np.ndarray:
        """解析服务端响应的 actions 字段为 [H, D] float64 数组（未做单位换算）。"""
        return payloads.parse_actions(result, self._contract)
