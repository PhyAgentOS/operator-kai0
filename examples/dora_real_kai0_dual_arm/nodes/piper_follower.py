#!/usr/bin/env python3
"""真机 Piper 从站节点（vendor 驱动）：行为与上游 forge_robots/packages/piper/main.py
一致（config 解析 + PiperDriver + robot_core 标准节点循环），仅依赖本仓库 vendor 层
（见 vendor/robots_piper/VENDOR.md，驱动逻辑与上游逐字节一致）。

- 订阅 {robot_id}/action（task_robot 输出，JointCommand 格式，夹爪毫米）→ CAN 下发；
- tick 上报 {robot_id}/state（JointState 格式，夹爪毫米）；
- 启动回零（参考 real_deploy 定型口径，startup_return_to_zero=true 时）：
  连接后不一次性冲零（auto_set_safety_position=False），改为从当前反馈位置
  smoothstep 插值回零（8s / 25Hz），完成前不进入节点循环。
"""

from __future__ import annotations

import sys

from robot_core import run_dora_robot_node
from robots_piper import PiperDriver, load_config
from startup_zero import run_startup_zero_motion


def main() -> int:
    config_path = None
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--config" and i + 1 < len(args):
            config_path = args[i + 1]
    config = load_config(config_path=config_path)
    driver = PiperDriver(
        port=config.port,
        is_follower=config.is_follower,
        auto_connect=True,
        # 参考 real_deploy：startup_return_to_zero 开启时关闭连接即下发的一次性
        # 安全位命令，改由下方插值回零接管（避免先冲零再插值）。
        auto_set_safety_position=not config.startup_return_to_zero,
    )
    if config.startup_return_to_zero:
        run_startup_zero_motion(
            driver,
            duration_s=config.startup_zero_duration_s,
            rate_hz=config.startup_zero_rate_hz,
        )
    return run_dora_robot_node(
        driver,
        actuator_order=driver.actuator_order,
        is_follower=config.is_follower,
        debug=config.debug,
    )


if __name__ == "__main__":
    sys.exit(main())
