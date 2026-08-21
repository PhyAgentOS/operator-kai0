# dora_real_kai0_dual_arm

真机 **双臂 14D 装箱** 示例：真实 kai0 推理服务端 + kai0 策略节点 + 真机双臂（Piper）+ 三路
USB 相机。与仿真示例 `dora_sim_kai0_dual_arm` **共用同一策略节点与服务端**，只更换
环境侧（仿真 mujoco → 真机 piper/相机）。格式对齐框架官方 `forge_runtime/examples/real_playback`。

## 文件

| 文件 | 说明 |
|------|------|
| `dataflow.yaml` | 相机×3 / 左右 Piper / task_robot / 单位桥 / kai0_policy / image_viewer |
| `dataflow_mock.yaml` | **无硬件验收版**：同拓扑，piper/usb_camera 换为 mock 节点（`scripts/start_real.sh --mock`） |
| `kai0_rtc.yaml` | 策略节点配置（RTC 异步推理；`rtc_control_rate=20` 与 dataflow tick 一致） |
| `task_robot.yaml` | 左+右共 14 关节，actuator 顺序与 Piper 固定契约一致（夹爪毫米） |
| `piper_slave_*.yaml` | 左右臂 CAN 端口（现场修改）+ 启动插值回零参数（8s/25Hz，见下） |
| `camera_*.yaml` | 顶视/左腕/右腕相机 `/dev/v4l/by-path/...`（现场修改） |
| `nodes/kai0_unit_bridge.py` | 单位桥：夹爪「Forge 米 ↔ Piper 毫米」双向换算（仅夹爪列） |
| `nodes/piper_follower.py` | 真机 Piper 节点（vendor 驱动 + robot_core 标准循环） |
| `nodes/mock_piper.py` | 假 Piper 驱动（同一节点循环 + 一阶跟随，软件在环） |
| `nodes/mock_camera.py` | 假相机（渐变+噪声帧） |
| `verify_config.py` | 配置契约校验（无硬件可跑）：joints/单位/tick/图像映射 |
| `vendor/` | 真机链路 vendor 层（上游 forge_robots 依赖了 forge_msgs 未发布的 API，见各 VENDOR.md） |

## 单位链（为什么有 bridge 节点）

- kai0 契约：夹爪**服务端毫米 ↔ Forge 米**（`inference/contract.py`），策略节点输出/消费
  Forge 米——仿真 mujoco 与真机同一口径；
- Piper 驱动固定毫米（`PIPER_ACTUATOR_ORDER`：两指总开口 0..105 mm），`task_robot` 只
  透传不换算；
- 故 `kai0_policy` 与 `task_robot` 之间由 `kai0_unit_bridge` 做双向换算（仅夹爪列
  index 6/13，关节透传）。

## vendor 层说明

真机节点链（`robot_core` 节点循环 / `PiperDriver` 驱动）引用 `forge_msgs` **从未发布过**
的 API（`ActuatorValue` / `RobotAction` / `RobotState` / `forge_msgs.value`，1.0.0 与
dev 分支均不存在，已核对 git 历史）——上游跨仓库不同步。真机示例因此 vendor
`vendor/` 三包：驱动与节点循环逻辑与上游**逐字节一致**，仅 import 重定向 + dora
边界统一到 forge-msgs 1.0.0 契约（JointCommand/JointState）；上游修复后可删除
（同步方法见 `vendor/robot_core/VENDOR.md`）。

## 启动插值回零（参考 real_deploy 定型口径）

历史口径是连接后立刻下发一次性全 0 安全位——真机上电瞬间机械臂可能在任意姿态，直接
冲零有风险。本示例已改为参考项目的定型做法（`piper_slave_*.yaml`）：

- `startup_return_to_zero: true`：连接时不下发安全位（`PiperDriver
  auto_set_safety_position=False`），改为读**当前反馈位置**，以 cubic smoothstep
  插值回零（8s / 25Hz，两端速度为零），完成前节点不进入循环、不接收策略动作；
- 反馈缺失或非有限时**拒绝执行**并报错退出（不知道位置不动臂）；
- 实现：`nodes/startup_zero.py`（轨迹）+ `nodes/piper_follower.py`（调用点），
  配置键与默认值同参考项目；双臂为两个独立节点、并行插值。

## 上真机前必改参数（逐项核对）

以下参数是**真机接入的全部必改项**，按「文件:行号」修改（行号以当前版本为准，修改前
先对照字段名）；可选参数按场景决定。全部改完后再走「无硬件验收（mock）」→「运行」。

### 必改：CAN 端口（`piper_slave_*.yaml` 第 5 行 `port:`）

| 文件 | 行 | 字段 | 默认示例值 | 现场动作 |
|---|---|---|---|---|
| `piper_slave_left.yaml` | 9 | `port` | `can_follower_l` | 改为**左臂**实际 CAN 口（如 `can0`） |
| `piper_slave_right.yaml` | 9 | `port` | `can_follower_r` | 改为**右臂**实际 CAN 口 |

端口与机械臂的对应**必须让机械臂动才能确认**（不要靠猜）：在示例目录 venv 下逐口运行
夹爪开合循环，观察哪只臂在动：

```bash
cd examples/dora_real_kai0_dual_arm
uv run python <框架根>/forge_robots/packages/piper/piper_ports.py gripper-cycle can0   # 15 秒夹爪开合
uv run python <框架根>/forge_robots/packages/piper/piper_ports.py gripper-cycle can1
```

也可先 `list-ports` 看有哪些 CAN 口：`uv run python <框架根>/forge_robots/packages/piper/piper_ports.py list-ports`。
⚠️ 左右臂接反 = 策略把左臂动作发给右臂，真机安全风险，务必逐口确认。

### 必改：相机设备路径（`camera_*.yaml` 的 `device:`）

| 文件 | 行 | 字段 | 默认示例值 | 现场动作 |
|---|---|---|---|---|
| `camera_top.yaml` | 4 | `device` | `/dev/v4l/by-path/pci-0000:00:0d.0-...`（顶视） | 改为**顶视相机**路径 |
| `camera_left.yaml` | 5 | `device` | `/dev/v4l/by-path/pci-0000:80:14.0-...-0:6:...`（左腕） | 改为**左腕相机**路径 |
| `camera_right.yaml` | 5 | `device` | `/dev/v4l/by-path/pci-0000:80:14.0-...-0:5:...`（右腕） | 改为**右腕相机**路径 |

确认方法：

```bash
v4l2-ctl --list-devices        # 列出设备名 → /dev/v4l/by-path/... 的对应关系
```

把**物理位置**与列表对号入座（顶视 / 左腕 / 右腕）。⚠️ 相机映射与策略观测**一一对应**
（`kai0_rtc.yaml` 第 56-58 行 `image_inputs`：顶视→`top_head`、左腕→`hand_left`、
右腕→`hand_right`），左右腕接反会把错误的画面发给策略。默认 640×480@30、JPEG 无需
改动；`image_format: jpeg`（第 7 行）若训练数据为其它编码需与数据采集口径核对。

### 可选：服务端地址 / 启动方式（`kai0_rtc.yaml`）

| 行 | 字段 | 默认值 | 何时改 |
|---|---|---|---|
| 38 | `server_url` | `ws://127.0.0.1:8000` | **客户端与服务端分机部署**时改为服务端地址（`ws://<服务端IP>:8000`）；同机部署不改。分机时启动脚本须同步 `SERVER_HOST=<服务端IP>`（`start_real.sh` 的 healthz 检查用），改端口则 `SERVER_PORT` 与 `server_url` 一并改 |
| 37 | `auto_start` | `true` | 生产接入 gateway（`policy_command` 驱动）时置 `false` |
| 42 | `rtc_control_rate` | `20` | 控制率与 `dataflow.yaml` 的 `kai0_policy` tick（第 100 行）必须一致（默认 50ms=20Hz）；改动需同时改 tick |

## 前置

1. 推理服务端已在 8000 端口以 RTC 启动（仓库根执行）：

   ```bash
   bash scripts/start_server.sh --ckpt ckpt/<任务名>/<step> --rtc
   ```

2. Forge 框架工作区就位（`forge_robots` / `forge_runtime` / `devices`，默认布局
   `<框架根>/policies/embodied/kai0_runner`；非默认布局用 `FORGE_ROOT` 覆盖）。
3. **构建 usb_camera 相机节点**（Rust，首次一次性；`devices/camera/usb_camera`）：

   ```bash
   cd <框架根>/devices/camera/usb_camera && cargo build --release --bins
   ```

4. 按现场修改本目录配置：CAN 端口与相机设备路径等**全部必改项见上文「上真机前必改
   参数」节**（精确到文件:行号与确认命令）。
5. 环境解析：首次运行 `dora build --uv` 时按本目录 `pyproject.toml` 创建节点环境
   （内含 kai0_runner editable + vendor 真机层），需能访问 Forge 依赖仓库。

## 无硬件验收（mock 模式，推荐先跑）

mock 末端（假 Piper 驱动 + 假相机）与真机 dataflow **同拓扑**，复用同一
`task_robot` / 单位桥 / 策略节点：

```bash
# 仓库根执行；前置只需服务端已启动（usb_camera 二进制可不构建）
bash scripts/start_real.sh --mock

# 或先跑配置契约校验（无服务端也可）：
uv run --project examples/dora_real_kai0_dual_arm python examples/dora_real_kai0_dual_arm/verify_config.py
```

验收判据：`dora_run_mock.log` 中无 Arrow schema 错误、kai0_policy 持续 `[RTC] request=N`
（2Hz）、task_robot 的 msg/s 统计显示状态/动作 20Hz 与图像 30Hz 全流动。

## 安全提醒

- 启动前清空机械臂周围的人员和障碍物；急停按钮在手边且可用。
- 确认左右机械臂、左右腕相机**没有接反**（画面分布与 `image_inputs` 契约对应：
  顶视→`top_head`、左腕→`hand_left`、右腕→`hand_right`）。
- 程序运行异常时，先按急停，再在终端 `Ctrl+C`。
- 首次启动前建议先跑无真机冒烟：`uv run kai0 infer-once --config config/inference/dummy.yaml`。

## 运行

```bash
# 仓库根执行：启动脚本会做安全确认（输入 RUN 继续）+ /healthz 检查 + dora build/run
bash scripts/start_real.sh
```

结束：`Ctrl+C`（脚本 trap 会清理残留节点进程）。

## 验收

- `image_viewer` 窗口显示三路画面（顶视/左腕/右腕）；
- 机械臂按策略动作执行（装箱行为），终端无 action schema 错误日志；
- 服务端侧可见 RTC 请求/响应日志（`serve_policy` 进程）。

## 约定

- `len(joints) == 14`（左 7 + 右 7），顺序与训练一致，与 `kai0_rtc.yaml` / 仿真示例相同；
- 真机控制回读 20 Hz（`rtc_control_rate=20`，真机部署定型值），dataflow 各 tick 均为 50 ms；
- 启动插值回零 8s / 25Hz（参考 real_deploy 定型口径，连接不冲零、smoothstep 插值）；
- 生产接入 gateway（`policy_command` 驱动）时，将 `kai0_rtc.yaml` 的 `auto_start` 置 `false`。
