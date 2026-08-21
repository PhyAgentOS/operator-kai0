"""模拟 openpi Policy.infer 的返回结构：{state, actions, policy_timing}。"""

from __future__ import annotations

import time

import numpy as np

from fake_policy_server.actions import ActionSource, ZeroActionSource


class FakePolicy:
    """替代真实 Policy：不做任何推理，按 ActionSource 生成动作块。"""

    def __init__(
        self,
        source: ActionSource | None = None,
        *,
        action_dim: int = 14,
        chunk_size: int = 50,
        infer_delay_ms: float = 0.0,
    ) -> None:
        self._source = source or ZeroActionSource()
        self._action_dim = action_dim
        self._chunk_size = chunk_size
        self._infer_delay_ms = infer_delay_ms
        self._step = 0

    def infer(self, obs: dict) -> dict:
        start = time.monotonic()
        if self._infer_delay_ms > 0:
            time.sleep(self._infer_delay_ms / 1000)
        actions = self._source(self._step, action_dim=self._action_dim, chunk_size=self._chunk_size)
        self._step += 1
        return {
            # 真实 Policy.infer 会返回模型内部 state；假服务端返回空数组占位。
            "state": np.zeros(0, dtype=np.float32),
            "actions": actions,
            "policy_timing": {"infer_ms": (time.monotonic() - start) * 1000},
        }
