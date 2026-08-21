# VENDOR：机器人消息类型（forge-robots-msgs）

上游 `forge_robots`（commit `9c69078`）与 `forge-msgs`（1.0.0 与 dev 分支）存在
**跨仓库不同步**：`robot_core` / `piper/driver.py` 引用的 `forge_msgs.value`、
`ActuatorValue`、`RobotAction`、`RobotState` **在任何 forge-msgs 版本中都不存在**
（已核对 git 历史，从未发布过）。真机示例因此 vendor 本包补齐这些类型。

## 类型与 dora 边界

- `ActuatorValue` / `RobotAction` / `RobotState`：API 与 `robot_core` 使用面完全一致
  （`action.actuators[name].value`、`state.actuators`、`RobotAction.from_arrow(value, order)`、
  `state.to_arrow(order)`），**驱动层零逻辑改动**；
- `value.py`：`ensure_record_batch` 别名（上游引用 `forge_msgs.value`，实为 `forge_msgs.arrow`）；
- **dora 边界统一到 forge-msgs 1.0.0 契约**：`RobotState.to_arrow` 输出
  `JointState` 格式（name/position/velocity/effort 列）、`RobotAction.from_arrow`
  解析 `JointCommand` 格式（name/mode/position 列）——与 task_robot 源码
  （输出 JointCommand、消费 JointState）互通。

## 同步方法

上游发布机器人消息类型后，删除本包并把真机示例的
`robot_core` / `robots_piper` vendor 重新指向上游（见 robot_core/VENDOR.md）。
