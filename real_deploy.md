# kai0_runner 真机部署指南

本文档为 kai0_runner 真机部署的操作手册,按章节顺序执行即可完成配置与启动。

**适用范围**:同机部署——推理服务端与真机数据流运行于同一主机,该主机具备 NVIDIA GPU,
并经 CAN 连接机械臂、经 USB 连接三路相机。服务端独立部署的差异见
[附录 A](#附录-a服务端独立部署);离线/受限网络见[附录 B](#附录-b离线与受限网络)。
文中 `<框架根>` 指 Forge 框架工作区根目录。

**执行顺序**:环境安装(第 2 节)→ 配置契约校验(第 3 节)→ 权重放置与服务端启动
(第 4-5 节)→ 无硬件验收(第 6 节)→ 现场硬件参数确认(第 7 节)→ 真机启动(第 8 节)。
软件链路问题在无硬件验收阶段先行暴露,真机阶段剩余风险集中于硬件层,便于定位。

## 0. 交付内容与目录布局

交付内容包括三部分:

| 交付物 | 内容 | 说明 |
|---|---|---|
| 代码包 | 本仓库(kai0_runner) | 客户端、服务端代码、启动脚本、真机示例 |
| 权重包 | `ckpt/` 下的 checkpoint 与 norm_stats | 由部署人员单独分发,不随 git 提交 |
| Forge 工作区 | forge_runtime / forge_robots / devices | 提供 dora、task_robot、piper 工具、相机节点 |

默认布局:本仓库位于框架工作区内(下文命令均以仓库根为工作目录):

```text
<框架根>/                        # 示例: /home/robot/forge
├── forge_runtime/               # dora、task_robot、image_viewer
├── forge_robots/                # piper_ports.py(CAN 确认工具)
├── devices/camera/usb_camera/   # 相机节点(Rust,使用前须构建)
└── policies/embodied/
    └── kai0_runner/             # 本仓库
        ├── scripts/start_server.sh   # 推理服务端启动
        ├── scripts/start_real.sh     # 真机启动(--mock 为无硬件验收)
        ├── server/                   # 服务端代码与 setup.sh 一键安装
        ├── ckpt/                     # 权重放置位置(约定见第 4 节)
        ├── config/inference/         # 仓库级客户端配置
        └── examples/dora_real_kai0_dual_arm/   # 真机示例
            ├── dataflow.yaml / dataflow_mock.yaml   # 真机 / 无硬件验收数据流
            ├── kai0_rtc.yaml         # 策略节点配置(server_url 在此)
            ├── piper_slave_left.yaml / piper_slave_right.yaml   # 左右臂 CAN 端口(第 7 节修改)
            ├── camera_top.yaml / camera_left.yaml / camera_right.yaml  # 相机设备(第 7 节修改)
            ├── task_robot.yaml       # 任务机器人配置
            ├── verify_config.py      # 配置契约校验(第 3 节)
            └── README.md             # 示例详细说明,必改项精确到文件与行号
```

仓库不在默认位置时,启动脚本支持 `FORGE_ROOT` 环境变量指向框架根;但 dataflow 内节点
相对路径约定不变,**建议保持默认布局**,避免两套路径口径。

## 1. 前置条件

启动前逐项确认以下条件:

```bash
# 1) Linux x86_64
uname -m
#    正常输出: x86_64

# 2) Python 3.12(客户端与服务端均需要)
python3.12 --version
#    正常输出: Python 3.12.x

# 3) uv(客户端依赖安装工具)
uv --version
#    未安装时: curl -LsSf https://astral.sh/uv/install.sh | sh
#    安装后重开终端,或 export PATH="$HOME/.local/bin:$PATH"

# 4) NVIDIA GPU 与驱动(服务端 JAX/CUDA)
nvidia-smi
#    正常输出: 显卡型号、显存与 CUDA Version: 12.x
#    实测推理显存峰值约 12.8GB,建议 16GB 及以上显卡
```

硬件连接确认:双臂 CAN 线已接入(`ip link show | grep can` 可见 can0/can1 等接口)、
三路 USB 相机已接入、机械臂上电。

网络要求(安装阶段一次性;运行阶段无外网依赖,服务端首次启动除外):

| 依赖 | 需访问 | 阶段 |
|---|---|---|
| 客户端 `uv sync` | 内网 `gitlab.ex-ai.cn`(forge 包源) | 安装 |
| 服务端 `server/setup.sh` | 官方 PyPI 与 github.com | 安装 |
| tokenizer | `gs://big_vision` | 服务端首次启动(离线方案见附录 B) |

## 2. 一次性安装

每台机器执行一次,日常启动不涉及:

```bash
cd <框架根>/policies/embodied/kai0_runner

# 1) 客户端环境(需内网 GitLab 可达)
uv sync --extra dev

# 2) 服务端环境(jax/cuda12 等重依赖,需 PyPI 与 github 可达,约数分钟)
bash server/setup.sh
#    正常结束输出类似: jax x.x.x | openpi / openpi_client / chex / pytest OK
#    (jax 版本以实际安装为准;脚本已按实测兼容组合锁定 ml-dtypes/tensorstore)

# 3) 相机节点构建(一次性)
cd <框架根>/devices/camera/usb_camera
cargo build --release --bins
cd <框架根>/policies/embodied/kai0_runner

# 4) 客户端冒烟:不连服务端、不涉硬件
uv run kai0 infer-once --config config/inference/dummy.yaml
#    正常退出、无报错(dummy 策略返回固定安全动作,不发网络请求)
```

## 3. 配置契约校验

不依赖服务端与硬件,校验真机示例配置的关节顺序、单位链、actuator 顺序、tick 频率、
图像映射与启动回零参数。环境安装完成后即执行:

```bash
uv run --project examples/dora_real_kai0_dual_arm python examples/dora_real_kai0_dual_arm/verify_config.py
```

通过判据:7 项全部 `[OK ]`,末尾输出「全部校验通过」。存在 FAIL 时先修正再继续。
校验项与硬件设备具体值无关,第 7 节修改 CAN/相机参数后可随时重跑。

## 4. 权重放置

权重按以下结构放置(目录名可自定义,结构固定):

```text
ckpt/fold_box_0721_0722_from_10000/   # 任务目录(示例名)
├── norm_stats.json                   # 归一化统计,位于 step 目录的父目录
└── 6000/                             # step 目录(启动时 --ckpt 指向此目录)
    └── params/                       # 模型权重
```

权重位于其他路径时,`--ckpt` 直接指向包含 `params/` 的 step 目录即可,无需移动文件。

## 5. 启动推理服务端

```bash
# RTC 模式(推荐,客户端配置默认口径);--ckpt 按实际目录替换
bash scripts/start_server.sh --ckpt ckpt/fold_box_0721_0722_from_10000/6000 --rtc
```

正常启动日志关键行:

```text
[INFO] CUDA_VISIBLE_DEVICES=0
[INFO] PORT=8000
[INFO] PROMPT_MODE=fixed
[INFO] POLICY_CONFIG=fold_box_normal_subtask_8_12_rtc
[INFO] POLICY_DIR=/.../ckpt/fold_box_0721_0722_from_10000/6000
[INFO] RTC 模式：客户端需设置 policy.rtc_async=true。
```

服务端保持前台运行,后续步骤另开终端执行。就绪验证:

```bash
curl -s http://127.0.0.1:8000/healthz && echo " ← HTTP 2xx 即就绪"
```

启动开关:

| 开关 | 使用条件 |
|---|---|
| `--rtc` | 默认推荐路径,客户端使用 kai0_rtc 配置 |
| (不加 `--rtc`) | 同步模式,调试/回退用,客户端改用 kai0.yaml |
| `--delta` | 权重以相对关节角(use_delta_joint_actions)训练 |
| `--lora` | LoRA 微调权重(params 内含 lora_a/lora_b) |

脚本内置动作空间安全检查:按 norm_stats 判定 checkpoint 的 delta/绝对属性并与开关
比对,不一致时拒绝启动并提示修正,防止量纲错误导致机械臂冲向零位。

## 6. 无硬件验收(mock)

**首次部署须执行**;本机已通过且环境未变更时可跳过。前置条件:服务端已启动(第 5 节)。

```bash
bash scripts/start_real.sh --mock
```

无需机械臂、相机与 usb_camera 二进制,以 mock 末端 + 相同拓扑验证完整软件链路:
真实服务端、策略节点、task_robot、单位桥、image_viewer,覆盖 WS 协议、Arrow schema、
tick 频率与 RTC 请求流。

通过判据(观察 `examples/dora_real_kai0_dual_arm/dora_run_mock.log`):

- 无 Arrow schema 报错;
- `kai0_policy` 持续输出 `[RTC] request=N`,约 2Hz(每 0.5 秒一条);
- `task_robot` 的 msg/s 统计显示状态/动作 20Hz、图像 30Hz 全流动。

注意:

- 首个推理请求包含服务端 RTC 路径 JIT 编译,数分钟无动作属正常(客户端
  `rtc_startup_timeout_s=300` 已覆盖);观察数分钟后 `Ctrl+C` 结束;
- mock 节点不读取 `piper_slave_*.yaml`,CAN 端口占位符在此阶段**不会报错**,
  端口正确性由第 7 节与真机连接验证;
- 验收期间服务端须保持运行。

## 7. 现场硬件参数确认

**左右对应关系不可接反**:左臂 CAN 口写入 `piper_slave_left.yaml`,右臂写入
`piper_slave_right.yaml`;相机同理。接反将导致策略把左臂动作下发右臂、错误画面
进入观测,属真机安全风险。

### 7.1 CAN 端口确认

CAN 端口须通过机械臂实际动作确认,禁止凭猜测。以下命令在示例目录 uv 环境中执行
(该环境已含 piper_sdk):

```bash
cd examples/dora_real_kai0_dual_arm

# 1) 列出 CAN 接口
uv run python <框架根>/forge_robots/packages/piper/piper_ports.py list-ports
#    输出形如 { "ports": [{ "name": "can0", "address": ..., "status": ... }, ...] }

# 2) 激活接口(弹出 pkexec 授权框;无图形界面时由运维预先配置)
uv run python <框架根>/forge_robots/packages/piper/piper_ports.py activate-ports

# 3) 逐口确认:每口执行 15 秒夹爪开合,观察哪只臂动作
uv run python <框架根>/forge_robots/packages/piper/piper_ports.py gripper-cycle can0
uv run python <框架根>/forge_robots/packages/piper/piper_ports.py gripper-cycle can1
#    记录结论,例如: can0=左臂、can1=右臂
```

### 7.2 相机设备路径确认

```bash
v4l2-ctl --list-devices
#    输出示例:
#      USB Camera (usb-cam): /dev/video2
#          /dev/v4l/by-path/pci-0000:00:0d.0-usb-0:2:1.4-video-index0
```

将物理位置(顶视 / 左腕 / 右腕)与 `/dev/v4l/by-path/...` 逐一对应并记录。
`v4l2-ctl` 缺失时执行 `sudo apt install v4l-utils`。

### 7.3 写入配置

修改文件(均在 `examples/dora_real_kai0_dual_arm/` 下):

| 文件 | 行 | 字段 | 修改 |
|---|---|---|---|
| `piper_slave_left.yaml` | 9 | `port` | 占位符 `can_follower_l` → 7.1 确认的左臂接口(如 `can0`) |
| `piper_slave_right.yaml` | 9 | `port` | 占位符 `can_follower_r` → 7.1 确认的右臂接口 |
| `camera_top.yaml` | 4 | `device` | 开发机示例路径 → 本机顶视相机 by-path |
| `camera_left.yaml` | 5 | `device` | → 本机左腕相机 by-path |
| `camera_right.yaml` | 5 | `device` | → 本机右腕相机 by-path |

`server_url` 同机部署保持 `ws://127.0.0.1:8000` 不变(分机部署见附录 A)。

## 8. 真机启动

启动前安全确认:

1. 机械臂周围无人、无障碍物;
2. 急停按钮在手边并可用;
3. 第 7 节左右确认已完成。

```bash
cd <框架根>/policies/embodied/kai0_runner
bash scripts/start_real.sh
#    提示 [SAFETY] 后输入 RUN 回车 → dora build(首次自动创建示例环境)→ dora run
```

启动后正常现象(按顺序):

1. 双臂各自输出 `[StartupZero] Moving ... to zero over 8.0s at 25Hz ...`——8 秒
   smoothstep 插值回零(自当前姿态平滑回零,无连接即冲零),期间策略不下发动作;
2. 输出 `[StartupZero] Zero-position trajectory completed; entering node loop.`
   ——回零完成,节点进入循环;
3. `kai0_policy` 开始输出 `[RTC] request=N`;首请求含 JIT 编译,数分钟属正常;
4. `image_viewer` 显示三路画面(顶视 / 左腕 / 右腕),机械臂开始执行策略动作。

停止与异常:

- 正常停止:`Ctrl+C`。脚本 trap 清理残留节点进程,机械臂断开前自动下休息位(防摔);
- 异常:先按急停,再返回终端 `Ctrl+C`;
- 服务端可保留或停止;重启服务端复用 `cache/jax_compilation` 编译缓存,数秒就绪。

## 9. 常见问题排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `start_real.sh` 报「服务端未就绪」 | 服务端未启动或端口不符 | 见第 5 节,`curl http://127.0.0.1:8000/healthz` 确认 |
| 服务端启动报「动作空间不一致」 | checkpoint delta/绝对属性与 `--delta` 开关不符 | 按提示增删 `--delta` 后重启 |
| 首个动作等待数分钟 | 服务端 RTC 路径 JIT 编译 | 正常;超过 300s 才报超时错误 |
| 首次启动报下载失败 | 无外网,tokenizer 无法下载 | 见附录 B 离线放置 |
| `[ERROR] EnablePiper failed` | CAN 接口未激活或端口配置错误 | 见第 7.1 节重查 |
| 相机节点无画面 | device 路径在本机不存在 | 见第 7.2 节重查 |
| piper 节点报未知配置 `startup_return_to_zero` | 示例环境为旧构建 | 删除 `examples/dora_real_kai0_dual_arm/.venv`,重跑 `start_real.sh` 重建 |
| 报「dora 不存在」/「usb_camera 二进制不存在」 | 工作区非默认布局或相机未构建 | 见脚本头部注释,`FORGE_ROOT` / `DORA_BIN` / `USB_CAMERA_BIN` 覆盖;或补第 2 节构建 |
| `uv` / `v4l2-ctl` 不存在 | 未安装 | uv 见第 1 节;v4l-utils 执行 `sudo apt install v4l-utils` |

## 附录 A:服务端独立部署

客户端(策略节点 + 真机数据流)与服务端分属不同机器时:

1. **服务端机器**(需 GPU):执行第 2 节服务端安装、第 4 节权重放置、第 5 节启动。
   服务端监听 `0.0.0.0:8000`,防火墙须放行 8000 端口。
2. **机械臂机器**(无需 GPU):执行第 2 节客户端安装与相机构建、第 7 节现场确认;
   另将 `examples/dora_real_kai0_dual_arm/kai0_rtc.yaml` 第 38 行 `server_url` 改为
   `ws://<服务端IP>:8000`,启动时:

   ```bash
   SERVER_HOST=<服务端IP> bash scripts/start_real.sh
   ```

   `SERVER_HOST`(脚本 healthz 检查)与 yaml `server_url` 须指向同一地址。
3. 变更端口(默认 8000)须三处同步:服务端 `--port`、客户端 `server_url`、启动脚本
   `SERVER_PORT`。同机部署不改端口则无需处理。

## 附录 B:离线与受限网络

- **镜像源**:客户端/服务端安装时设置 `UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`
  (或内部镜像)。服务端 github 源依赖(lerobot)不可达时,由运维在可达环境安装后整机迁移。
- **tokenizer 离线**:在可达环境启动一次服务端(或手工获取 `paligemma_tokenizer.model`),
  放置到 `<仓库>/cache/openpi/`,启动时跳过下载。
- **缓存迁移**:`cache/openpi`(tokenizer)与 `cache/jax_compilation`(JAX 编译缓存)均
  已 gitignore,可整目录随机器迁移。
