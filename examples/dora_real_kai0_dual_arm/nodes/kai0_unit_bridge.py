#!/usr/bin/env python3
"""真机单位桥：kai0_policy 与 task_robot 之间的夹爪「Forge 米 ↔ Piper 毫米」双向换算。

kai0 契约（inference/contract.py）：服务端夹爪毫米 ↔ Forge 米，策略节点输出/消费
Forge 米（仿真 mujoco 同口径）；而 Piper 驱动固定消费/上报毫米
（PIPER_ACTUATOR_ORDER，两指总开口 0..105 mm），task_robot 只透传不换算
（action_handler / state_handler 无 scale 逻辑）。本节点把两段接起来：

- action（kai0_policy → task_robot）：夹爪米 → 毫米（关节透传）
- proprio_state（task_robot → kai0_policy）：夹爪毫米 → 米（关节透传）

仅换算 14 维布局中的夹爪列（索引 6、13，见 contract.GRIPPER_INDICES）。
"""

from __future__ import annotations

import sys

import numpy as np
from dora import Node
from forge_msgs import JointCommand, JointState

from kai0_inference.inference import contract


def _scale_positions(values: list[float], factor: float) -> list[float]:
    """按 14 维布局换算夹爪列（contract.GRIPPER_INDICES），关节列不变。"""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0 or arr.size == 0:
        return list(values)
    for i in contract.GRIPPER_INDICES:
        if i < arr.size:
            arr[i] *= factor
    return arr.tolist()


def main() -> int:
    node = Node()
    for event in node:
        if event.get("kind") != "dora" or event["type"] != "INPUT":
            continue
        value = event.get("value")
        if value is None:
            continue
        input_id = event["id"]

        if input_id == "action":
            # kai0_policy 输出（Forge 米）→ task_robot 输入（Piper 毫米）
            cmd = JointCommand.from_arrow(value)
            scaled = JointCommand(
                name=cmd.name,
                mode=cmd.mode,
                position=_scale_positions(cmd.position, contract.GRIPPER_MM_PER_M),
                velocity=cmd.velocity,
                effort=cmd.effort,
                kp=cmd.kp,
                kd=cmd.kd,
            )
            node.send_output("action", scaled.to_arrow())
        elif input_id == "proprio_state":
            # task_robot 聚合的 Piper 状态（毫米）→ kai0_policy 输入（Forge 米）
            state = JointState.from_arrow(value)
            scaled = JointState(
                name=state.name,
                position=_scale_positions(
                    state.position, 1.0 / contract.GRIPPER_MM_PER_M
                ),
                velocity=state.velocity,
                effort=state.effort,
            )
            node.send_output("proprio_state", scaled.to_arrow())

    return 0


if __name__ == "__main__":
    sys.exit(main())
