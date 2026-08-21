# VENDOR：robots_piper（Piper 硬件驱动）

vendor 自框架工作区 `forge_robots/packages/piper`（commit `9c69078`），
**驱动逻辑（driver.py 268 行）逐字节一致**，仅两处 import 重定向：

1. `driver.py`：`from forge_msgs import ActuatorValue, RobotAction, RobotState`
   → `from forge_robots_msgs import ...`（上游 forge_msgs 从未发布这些类型，
   见 forge_robots_msgs/VENDOR.md）；
2. `driver.py`：`from config import PIPER_ACTUATOR_ORDER`
   → `from .config import ...`（pip 包内相对导入，源码模式为同目录解析）。

`config.py` 逐字节一致。真机节点 `nodes/piper_follower.py` 基于本包构造驱动。

## kai0 增量（与上游的差异，参考 real_deploy 定型口径）

启动回零安全改造（上游同步此改造后可删除）：

- `driver.py`：`PiperDriver` 新增 `auto_set_safety_position: bool = True` 参数，
  为 False 时 `_connect_follower_mode` 跳过连接后两次 `move_to_home_position()`
  （历史口径的连接即一次性冲零）；
- `config.py`：新增 `startup_return_to_zero` / `startup_zero_duration_s` /
  `startup_zero_rate_hz` 三个配置键（默认 8s / 25Hz，与参考配置一致）；
- 插值回零本体在 `nodes/startup_zero.py`（非 vendor），由 `nodes/piper_follower.py`
  在连接后、进入节点循环前调用：读当前反馈 → cubic smoothstep 插值到 0 位，
  反馈缺失/非有限时拒绝执行。

## 同步方法

上游发布机器人消息类型且 robot_core 可装后，删除本包，真机 dataflow 的 piper
节点改回框架工作区 `forge_robots/packages/piper/main.py`（README「节点以打包
二进制运行后此说明作废」同口径）。
