#!/usr/bin/env python3
"""启动插值回零（参考 real_deploy 定型口径：main_real_xvla.py `_run_startup_zero_motion`
+ forge robot/base.py `BaseTaskRobot.move_to_zero_position`）。

历史口径是连接后立刻下发一次性全 0 安全位（PiperDriver._connect_follower_mode 的
move_to_home_position，真机上电时机械臂可能在任意姿态，直接冲零有风险）。参考项目
已将该行为替换为：连接时不下发安全位（auto_set_safety_position=False），改为读当前
反馈位置，以 cubic smoothstep 插值回零——两端速度为零，全程匀速渐进。

本模块按单臂节点拓扑实现同一语义（kai0_runner 双臂为两个独立 dora 节点，各自插值、
并行执行；参考实现由 task robot 统一插值双臂，拓扑不同、轨迹语义一致）。
"""

from __future__ import annotations

import logging
import math
import time

from forge_robots_msgs import ActuatorValue, RobotAction
from robot_core import RobotDriver

logger = logging.getLogger(__name__)


def run_startup_zero_motion(driver: RobotDriver, *, duration_s: float, rate_hz: float) -> None:
    """把驱动所有执行器从当前反馈位置 smoothstep 插值到 0 位，完成后才返回。

    与参考实现一致：
    - 参数校验：duration/rate 必须 > 0（否则拒绝执行）；
    - 读一次当前反馈作为起点，反馈缺失或非有限直接拒绝（不知道位置不动臂）；
    - cubic smoothstep（alpha = p^2(3-2p)）保证起点/终点速度为零；
    - 以 1/rate_hz 为步长逐点下发；调用方须保证此期间没有其它动作来源（
      piper_follower 在进入 dora 节点循环前调用，天然满足）。
    """
    if duration_s <= 0:
        raise ValueError("startup_zero_duration_s must be greater than 0.")
    if rate_hz <= 0:
        raise ValueError("startup_zero_rate_hz must be greater than 0.")

    order = list(driver.actuator_order)
    state = driver.get_state()
    missing = [name for name in order if name not in state.actuators]
    if missing:
        raise RuntimeError(f"Cannot move to zero: feedback is missing for {missing}.")
    start = {name: float(state.actuators[name].value) for name in order}
    invalid = [name for name, value in start.items() if not math.isfinite(value)]
    if invalid:
        raise RuntimeError(f"Cannot move to zero: feedback is not finite for {invalid}.")

    steps = max(1, int(duration_s * rate_hz))
    logger.info(
        "[StartupZero] Moving %s to zero over %.1fs at %gHz (%d steps, smoothstep). "
        "Policy actions are not applied yet.",
        order, duration_s, rate_hz, steps,
    )

    step_delay = 1.0 / rate_hz
    for step_idx in range(1, steps + 1):
        progress = step_idx / steps
        # Cubic smoothstep gives zero trajectory velocity at both ends.
        alpha = progress * progress * (3.0 - 2.0 * progress)
        action = RobotAction(
            actuators={
                name: ActuatorValue(
                    value=start[name] * (1.0 - alpha),
                    mode=state.actuators[name].mode,
                    unit=state.actuators[name].unit,
                )
                for name in order
            }
        )
        driver.set_actuators(action)
        time.sleep(step_delay)

    logger.info("[StartupZero] Zero-position trajectory completed; entering node loop.")
