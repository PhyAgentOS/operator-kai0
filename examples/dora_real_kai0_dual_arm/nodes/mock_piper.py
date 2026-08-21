#!/usr/bin/env python3
"""Mock Piper 从站驱动 + 标准节点循环（无硬件软件在环）。

与真机 piper 节点（forge_robots/packages/piper/main.py）**共用同一节点循环**
（robot_core.run_dora_robot_node），只把 PiperDriver 换成 FakePiperDriver：
- 订阅 {robot_id}/action（task_robot 输出，RobotAction 毫米/弧度）→ 一阶跟随积分；
- tick 上报 {robot_id}/state（RobotState，与真机同单位：夹爪毫米、关节弧度）；
- 初始状态全零（与仿真 mujoco 起始口径一致）。

用途：
- 无硬件验收 dataflow（mock 模式：scripts/start_real.sh --mock）；
- 提前暴露 task_robot / forge-msgs schema 兼容性问题（action 校验与真机同路径）。

args 透传 --debug（打开 action/state 采样日志）。
"""

from __future__ import annotations

import sys

import numpy as np
from forge_robots_msgs import ActuatorValue, RobotAction, RobotState
from robot_core import BaseRobotDriver, run_dora_robot_node

# 与 Piper 固定执行器契约一致（PIPER_ACTUATOR_ORDER）
ACTUATOR_ORDER = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
_ARM_NAMES = set(ACTUATOR_ORDER[:-1])

FOLLOW_GAIN = 0.2   # 每 50ms tick 向目标靠近的比例（模拟机械臂跟随）
STATE_NOISE = 1e-4  # 状态读数噪声（弧度/毫米），防止数值完全静止


class FakePiperDriver(BaseRobotDriver):
    """软件在环 Piper 驱动：一阶跟随动作目标，无硬件。"""

    @property
    def actuator_order(self) -> list[str]:
        return list(ACTUATOR_ORDER)

    def __init__(self, seed: int = 0) -> None:
        self._pos = np.zeros(len(ACTUATOR_ORDER), dtype=np.float64)
        self._target = np.zeros(len(ACTUATOR_ORDER), dtype=np.float64)
        self._rng = np.random.default_rng(seed)

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def get_state(self) -> RobotState:
        # 位置模式：向目标一阶跟随（每次 get_state 前积分一步）
        self._pos += FOLLOW_GAIN * (self._target - self._pos)
        self._pos += self._rng.normal(0.0, STATE_NOISE, size=self._pos.size)
        return RobotState(
            actuators={
                name: ActuatorValue(
                    value=float(self._pos[i]),
                    mode="position",
                    unit="millimeters" if name == "gripper" else "radians",
                )
                for i, name in enumerate(ACTUATOR_ORDER)
            }
        )

    def set_actuators(self, action: RobotAction) -> None:
        # 位置模式：记录目标；gripper 限位 0..105 mm（真机同款安全口径）
        for name, av in action.actuators.items():
            if name not in ACTUATOR_ORDER:
                continue
            idx = ACTUATOR_ORDER.index(name)
            value = float(av.value)
            if name == "gripper":
                value = min(max(value, 0.0), 105.0)
            self._target[idx] = value


def main() -> int:
    debug = "--debug" in sys.argv[1:]
    return run_dora_robot_node(
        FakePiperDriver(seed=0),
        actuator_order=ACTUATOR_ORDER,
        is_follower=True,
        debug=debug,
    )


if __name__ == "__main__":
    sys.exit(main())
