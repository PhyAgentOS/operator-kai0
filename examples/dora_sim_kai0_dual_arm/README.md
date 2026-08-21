# 双臂 14 维仿真示例：kai0 策略节点 + 真实推理服务端（源码模式）

## 目标

在仿真中驱动真实 kai0 模型（装箱任务权重）完成推理：

```
真实推理服务端（仓库级 scripts/start_server.sh，与真机部署同一脚本，独立进程）
      ↕ WS（msgpack 字典线格式 codec，见 inference/transport/msgpack_codec.py）
kai0_policy → robot_router → mujoco 仿真（+ image_viewer 三路画面）
```

- 仿真场景为本地 MJCF 变体（`mjcf/`）：在 taskrobots 双臂桌面场景上添加了装箱任务道具
  （纸箱、青色物体、胶带、保护袋）与补光，供模型输出作用于场景物体；
- 模型输出为真实推理结果（非模式化动作），用于验证 14 维动作维度、数值有限性与
  夹爪单位换算（服务端毫米 → Forge 米）；
- 14 维关节顺序与协议正确性由 infer-once 冒烟、仿真/真机（真服务端）验证。

## 运行

```bash
bash scripts/start_sim.sh [秒数]          # 推荐默认：RTC 异步推理（服务端须以 --rtc 启动）
bash scripts/start_sim.sh [秒数] --sync   # 回退/调试：同步模式；需 DISPLAY 可用以便 image_viewer 弹窗
```

前置：真实推理服务端已启动（仓库级 `scripts/start_server.sh`，与真机部署同一脚本，启动方式见
仓库 README「启动推理服务端」；RTC 变体加 `--rtc`）。`start_sim.sh` 先做 `/healthz` 检查 →
`dora build --uv` → `dora run`（RTC 日志 `dora_run_rtc.log` / 同步 `dora_run.log`）→ 超时自动结束。

可选环境变量（非默认布局时）：`FORGE_ROOT`（Forge 框架工作区根）、`DORA_BIN`（dora 可执行）、
`UV_BIN`（uv 不在 PATH 时）、`MUJOCO_SIM_SRC`（mujoco_sim 源码目录）；详细说明见
`scripts/start_sim.sh` 头部注释。

**冷启动注意（实测）**：启动脚本已启用 JAX 编译缓存落盘（`cache/jax_compilation`）——
全新环境首次 RTC 请求编译约 2.5 分钟，期间机械臂不动、服务端无输出，属正常；
编译一次后，**重启服务端也能复用缓存**（首请求约 20s，同进程稳态 ~1s）。建议首次先跑
`uv run kai0 infer-once --config config/inference/kai0_rtc.yaml` 完成预热再跑仿真；
服务端进程不重启则无需再次预热。

> 源码模式说明：`robot_router` / `mujoco_sim` / `image_viewer` 直接指向 Forge 框架工作区
> （dataflow 相对路径上溯 5 级；默认按 `<框架根>/forge_runtime/...` 与
> `<框架根>/simulators/mujoco_sim/...` 推断，非默认布局用 `DORA_BIN` /
> `MUJOCO_SIM_SRC` / `FORGE_ROOT` 环境变量覆盖）。
> 各节点以打包二进制运行后，dataflow 的 path 换成二进制即可，其余不变。

## 目录结构

| 文件 | 说明 |
|---|---|
| `dataflow.yaml` | 四节点编排：kai0_policy（50Hz tick）→ robot_router → mujoco + image_viewer，外加取证节点 frame_capture |
| `kai0.yaml` | 复制自仓库 `config/inference/kai0.yaml`，适配 `auto_start: true`（无 gateway 自主运行）与 `server_url: ws://127.0.0.1:8000` |
| `dataflow_rtc.yaml` / `kai0_rtc.yaml` | RTC 变体（仅 kai0_policy 的 config 换为 kai0_rtc.yaml，其余与同步版一致）；服务端须以 `--rtc` 启动 |
| `robot_router.yaml` | 14 维全局契约 + 拆臂配置（与 forge_studio dual_arm_sim 模板一致） |
| `simulator.yaml` | model_path 指向本目录 `mjcf/` 本地场景变体（含装箱道具与补光），三路相机 640×480 |
| `mjcf/` | 本地仿真场景：taskrobots `Bimanual_PIPER_withTable_aloha` 的拷贝 + 装箱道具（纸箱/青色物体/胶带/保护袋）与补光（XML 内有注释） |
| `nodes/kai0_main.py` | kai0 节点本地包装入口（调用 `kai0_inference.cli`，dora path 必须是 python 文件） |
| `nodes/mujoco_main.py` | mujoco_sim 源码入口包装（src 布局相对导入需包上下文） |
| `nodes/frame_capture.py` | 取证节点：每 0.1s 打印 14 维关节值（SNAPSHOT 行）、每 5s 存三路图像快照到 `out/frames/` |
| `pyproject.toml` | 合并环境：三仓库节点依赖并集 + kai0-inference editable path 依赖 |
| （启动脚本） | 已移至仓库级 `scripts/start_sim.sh`：dora build + dora run + 超时回收，默认 RTC、`--sync` 切同步（先做服务端 /healthz 检查） |
| `verify_acceptance.py` | 验收数值化核查：解析 dora_run*.log 的 SNAPSHOT 行，校验机械臂运动/夹爪量程/0 错误 |

## 验收判据（真实推理，管线健康性）

- **机械臂在运动**：12 个臂关节中至少 6 个幅值 ≥ 0.05 rad（真实推理输出为工作位形，
  幅度与任务/场景有关，不设上限；实测 joint2 幅度可达 ~1.5 rad）；
- **夹爪**：全程在契约量程容差内（0..0.115 m，契约 0.105 m + mujoco 跟踪容差），
  且至少一个夹爪有开合动作；启动瞬态（前 2s）剔除；
- **日志 0 错误**（dora_run.log / dora_run_rtc.log）；
- 数值化核查：`.venv/bin/python verify_acceptance.py dora_run.log`。
  注意：本判据只判管线健康；任务完成度（装箱动作是否到位）受仿真场景还原度限制，
  不在判据内。

## 已知限制

- **任务完成度受场景还原度限制**：模型对画面分布敏感（开发期实测：桌面白度、纸箱
  颜色偏移都会让策略退回「保持」模式）；`camera_style` 节点（仅仿真）做风格补偿后
  已能稳定触发执行，但任务全流程完成仍是概率事件。
- **关节物理限位**：MJCF 双臂有关节限位（如 joint3 下限 -2.967 rad），策略输出的
  位形超出限位时会被物理截断——属环境约束，非映射错误。
