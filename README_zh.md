# kai0 推理

[English](README.md) | [简体中文](README_zh.md)

> **🚀 真机部署请看 [real_deploy.md](real_deploy.md)**——从环境安装、现场硬件确认
> （CAN 端口 / 相机）、无硬件验收（mock）到真机启动的完整操作手册，含命令示例、
> 期望输出与常见问题排查。真机部署一律以该文档为准。

kai0（χ₀）是一个基于 openpi 开发的具身操作策略模型，其权重加载与推理依赖 JAX/openpi
技术栈。与 lerobot 类可以直接嵌入策略节点运行的策略不同，kai0 无法在 Forge 推理节点
所在的进程内直接加载权重：openpi 的依赖体系与 Forge 策略节点的运行环境存在冲突，两者
共存于同一进程会互相干扰。因此节点不直接加载模型，而是通过 WebSocket 与独立运行的推理
服务端通信——这也是 openpi 官方推荐的部署方式。本项目即按此形态实现：`kai0_runner`
作为客户端，负责接收观测、调用服务端完成推理，并将返回的动作交给 Forge 数据流；
`server/` 则以独立进程运行 JAX/openpi 推理，承担权重加载与模型推理，两端经 WebSocket
通信。

## Node、Endpoint 与 Tool API

| 项目 | 当前实现 |
| --- | --- |
| 生产 Node | 1 个推理 Node：`kai0 infer`；Node 打包 entrypoint 为 `kai0_policy`，见 [build/make_node_bundle.sh](build/make_node_bundle.sh)；`infer-once` 是离线 CLI，不另计 Node |
| Tool 支持 | 可选 provider 模式：推理配置 `mode: session_endpoint`；默认 `dora_plain` 不注册 Endpoint |
| Endpoint 数 | provider 模式为 1：`policy.runner`；plain 模式为 0 |
| operation 数 | provider 模式为 1：`serve`（Session，可 stop、可查状态，最大并发 1） |
| 直接 HTTP Tool API | 0；`tool_in/tool_out` 通过 Dora 接入 Gateway |
| Gateway Tool 数 | 由 Skill/Gateway `tools.specs` 决定，仓库不固定外部 Tool ID |

调用映射为 `<tool_id from tools.specs> -> policy.runner / serve -> kai0 infer`。
会话 attach 或启动独立 openpi 推理服务，执行就绪门并按进程所有权处理 stop；外部推理服务不是额外的 Dora Node。

Session 使用 stop，不应替换成 Action cancel。对外调用通过 Gateway
`POST /tools/{tool_id}:invoke`，随后读取 invocation 状态/结果；停止使用
`POST /invocations/{invocation_id}/stop`。HTTP 路由属于 Gateway，Node 内的推理服务接口不算 Tool API。

Kai0 与 LeRobot 使用相同 `policy.runner` Endpoint ID；同一 Gateway 中该 ID 只有一个 current instance，
两者应作为部署候选选择，不能仅用不同 Node ID 就假定能同时成为独立路由。

实现与启用条件：[descriptor](inference/session_endpoint.py)、[runner](inference/session_runner.py)、
[配置校验](inference/config.py)、[推理入口](inference/main.py)。
Gateway 当前源码实现 Session，但历史二进制可能不同，需核对配套版本与依赖锁。

兼容性：当前依赖声明为 `forge-msgs==1.2.0`、`forge-common==1.0.1`、`forge-policy==1.0.1`、`forge-tool[dora]==1.0.0`、`dora-rs==0.4.1`；以 [pyproject.toml](pyproject.toml) 与 [uv.lock](uv.lock) 为准。 这些声明不构成跨版本互操作保证；使用制品时还需核对 build revision 和 Skill lock。

后续修改 Node 入口、Endpoint 或 Tool API 时，请同步更新本节及对应配置/契约，遵循 [Node README 规范](https://github.com/PhyAgentOS/forge/blob/master/CONTRIBUTING.md#node-readme-contract)。

## 系统架构

系统分三层，服务端与客户端之间走 WebSocket 协议，客户端与环境侧之间走 Dora 数据流：

```text
┌─ 推理服务端（独立进程，本仓库 server/）────────────────────┐
│  serve_policy：JAX/openpi 模型推理                         │
│  加载 ckpt/ 权重 + norm_stats，WS 端口 8000                │
└──────────────────────▲─────────────────────────────────────┘
                       │ WebSocket（msgpack 二进制帧）
┌──────────────────────┴───────────────────────────────────────┐
│ 客户端（策略节点，本仓库）                                   │
│  kai0_runner：观测 → 动作（不加载模型）                      │
│  由 config/inference/*.yaml 驱动，以 CLI 或 Dora 节点运行    │
└──────────────────────┬───────────────────────────────────────┘
                       │ Dora 动作（14 维关节位置）
┌──────────────────────┴──────────────────────────────────────────┐
│ 环境侧（可替换）                                                │
│  仿真：本地 MJCF 场景（examples/dora_sim_kai0_dual_arm/）       │
│  真机：examples/dora_real_kai0_dual_arm（Piper 双臂 + USB 相机）│
└─────────────────────────────────────────────────────────────────┘
```

**部署形态**：推理服务端与客户端是**必选**组件（即「kai0 接入 Forge」的本体，任何环境
都要运行）；环境侧**二选一**——仿真（`examples/dora_sim_kai0_dual_arm/`）用于本机演示/验证，
真机（`examples/dora_real_kai0_dual_arm/`）为生产部署。

## 安装

以下为**每台机器一次性**的环境配置；日常运行不需要重复安装，直接走「部署流程」。
客户端与服务端可以装在同一台机器，也可以分机（见「系统架构」部署形态）。
两个环境分开装：**客户端**（uv 依赖）与**推理服务端**（JAX 依赖）。

要求：Linux x86_64、Python 3.12、[uv](https://docs.astral.sh/uv/)；客户端安装时需能访问
公开 Python 包索引（Forge 版本由 `pyproject.toml` / `uv.lock` 固定，无 Forge git 来源）；
服务端安装（`server/setup.sh`）还需访问 github.com（lerobot 依赖）。推理服务端额外需要
NVIDIA GPU（JAX CUDA，实测显存峰值 12.8GB / 16GB 卡、推理 ~255ms）；客户端为轻量
WS 进程、不占显存（dummy infer-once 实测峰值 RSS ≈ 44MB）。**仿真与真机侧**均需
Forge 框架工作区（见「部署流程」第 3 步）。

```bash
# 客户端（节点 + CLI + Dora 示例），在仓库根目录执行
uv sync --extra dev

# 推理服务端（有网环境一键安装，首次数分钟；UV_INDEX_URL 可选指向镜像）
bash server/setup.sh
```

主命令为 `kai0`，子命令 `infer` / `infer-once`（兼容命令 `kai0-infer` / `kai0-infer-once`）。
单元测试不连网：`uv run pytest`。

## 部署流程

> **真机部署完整操作手册见 [real_deploy.md](real_deploy.md)**:从环境安装、现场硬件
> 确认（CAN 端口/相机）、无硬件验收到真机启动的全流程,含命令示例与常见问题排查。
> 以下各节为该手册的机制说明与参数速查。

### 1. 启动推理服务端

权重由部署人员分发，放置约定见 `ckpt/README.md`。启动（仿真与真机部署**同一脚本**，
内置动作空间安全检查）：

```bash
# 通用形式：--ckpt 指向权重步骤目录（须含 params/，其父目录须含 norm_stats.json）
bash scripts/start_server.sh --ckpt 'ckpt/<task>/<step>' --rtc

# 将 task/step 替换为部署时提供的权重目录
bash scripts/start_server.sh --ckpt 'ckpt/<task>/<step>' --rtc
```

参数选取：`--rtc` 表示采用 RTC 模式（默认，客户端用 `kai0_rtc.yaml`）；不加 `--rtc`
则是不采用 RTC 模式（客户端 `kai0.yaml`，调试/回退用）。`--delta` 对应以相对关节角训练的
checkpoint，绝对角训练的 checkpoint 不用加。

首次启动会自动下载 tokenizer 到 `OPENPI_DATA_HOME`（默认 `<仓库>/cache/openpi`）；
无网环境需手工放置 tokenizer 到默认位置（`cache/openpi/big_vision/`），或
`export OPENPI_DATA_HOME` 指向已有位置。

### 2. 预热（建议在启动客户端前先跑）

首次推理请求需 JAX 编译（全新环境约 2.5 分钟），预热一次后进入稳态：

```bash
uv run kai0 infer-once --config config/inference/kai0_rtc.yaml
```

### 3. 启动（仿真 / 真机二选一）

仿真与真机**共用同一推理服务端、策略节点与 Forge 框架工作区**，区别只在环境侧 dataflow
实际驱动的机构：仿真跑本地 MJCF 模拟器，真机跑 Piper 双臂 + USB 相机。两个启动脚本会
连同客户端节点一起拉起——客户端节点嵌在示例 dataflow 中，无需单独启动。

**仿真**（`examples/dora_sim_kai0_dual_arm/`，默认 RTC，90s 后自动停）：

```bash
bash scripts/start_sim.sh 90
```

**真机**（`examples/dora_real_kai0_dual_arm/`，安全确认后运行；无硬件可先 `--mock` 验收）：

```bash
bash scripts/start_real.sh
```

接入真机前，按现场修改 CAN 端口（`piper_slave_left.yaml` / `piper_slave_right.yaml` 的
`port:`）与相机设备（`camera_top/left/right.yaml` 的 `device:`）；完整改动清单与确认命令
见真机示例 README「上真机前必改参数」节。

## 推理配置

| 配置 | adapter | 说明 |
| --- | --- | --- |
| `config/inference/kai0_rtc.yaml` | `kai0_rtc` | 采用 RTC 模式（默认） |
| `config/inference/kai0.yaml` | `kai0` | 不采用 RTC 模式（调试/回退） |
| `config/inference/kai0_libero.yaml` | `kai0_libero` | LIBERO 基准（同步模式 + LIBERO 契约，见 `examples/dora_bench_libero/`） |
| `config/inference/dummy.yaml` | `dummy` | 无服务端调试（不连网） |

字段说明见 `config/inference/README.md`；关节顺序、单位换算等契约见 `inference/contract.py`。

## 项目结构

```text
.
├── cli.py                 # 统一 CLI 入口
├── real_deploy.md         # 真机部署操作手册（环境安装 → 现场硬件确认 → 验收 → 启动）
├── common/                # 共用路径逻辑
├── inference/             # 客户端：Dora 与单次推理实现
│   ├── contract.py        # 关节顺序、单位换算、相机别名契约（双臂 / LIBERO 两套 profile）
│   ├── config.py          # 配置加载与契约校验
│   ├── transport/         # 与服务端通信的 WebSocket 传输层（短连接 / 常驻连接 + msgpack codec）
│   ├── policies/          # kai0（同步）、kai0_rtc（RTC）、kai0_libero（LIBERO）、dummy 策略适配器
│   ├── main.py            # Dora 节点主体
│   └── run_once.py        # infer-once 单次推理
├── config/                # 推理配置模板（见 config/inference/README.md）
├── examples/              # Dora 示例（见 examples/README.md）
│   ├── dora_sim_kai0_dual_arm/   # 仿真示例：本地 MJCF 场景 + 策略节点
│   ├── dora_real_kai0_dual_arm/  # 真机示例：Piper 双臂 + USB 相机 + 单位桥
│   └── dora_bench_libero/        # LIBERO 基准示例：benchmark 节点 + kai0_libero 策略节点
├── scripts/               # 启动脚本：start_server.sh（服务端）/ start_sim.sh（仿真）/ start_real.sh（真机）
├── server/                # 推理服务端：kai0_prompt_schedule/（代码快照）+ SOURCE.md（来源与同步说明）
│   └── setup.sh           # 服务端环境一键安装
├── ckpt/                  # checkpoint 权重放置约定（权重不随仓库分发，见 ckpt/README.md）
├── test/                  # 测试配套：fake_policy_server（协议级假服务端，见 test/fake_policy_server/README.md）
└── tests/                 # 单元测试
```
