# server/kai0_prompt_schedule —— 代码快照来源说明

本目录是 kai0 推理服务端代码的**快照拷贝**（非 git submodule），用于让 kai0_runner
自包含：客户只需本仓库 + 权重即可跑通推理服务端，无需再获取源项目。

## 来源

- 服务端基于开源 openpi（`Physical-Intelligence/openpi`，Apache-2.0）与 kai0
  （`OpenDriveLab/kai0`）构建，并包含 RTC 引导推理、LoRA/Delta 推理配置等本地定制；
- 快照 commit：`9b6086f`（2026-08-17）。

## 拷贝范围（运行必需子集，76 个文件）

| 拷入 | 说明 |
|---|---|
| `src/openpi/` | 服务端核心（含 Pi0RTC 引导推理） |
| `packages/openpi-client/` | WS 客户端库（uv workspace 成员，pyproject 依赖） |
| `scripts/serve_policy.py` + `scripts/__init__.py` | 服务端入口 |
| `pyproject.toml` / `uv.lock` | 依赖与锁文件（安装用） |
| `LICENSE` / `LICENSE_GEMMA.txt` / `README.md` / `.python-version` | 许可证与说明 |

**未包含**：`docs/`、`assets/`（文档与非运行资产）；`scripts/docker/`（容器化部署，
本仓库用 venv 方式，见 `server/setup.sh`）；`.github/`/`.cursor/`/`.vscode/`（工具配置）；
`policy_server.sh`（旧启动脚本，已被仓库级 `scripts/start_server.sh` 取代）；
`mixed_models/`（权重，见仓库 `ckpt/README.md` 放置约定）；`.venv/`（运行环境，
由 `server/setup.sh` 新建）。

## 公开版差异

本目录代码与源快照保持一致，唯一例外是 `src/openpi/training/config.py` 与
`README.md`：训练/部署示例中的本地数据目录绝对路径与带日期的实验命名已替换为
占位符（`<path_to_dataset>`、`<path_to_checkpoints>` 等），避免泄漏内部机器信息；
服务端运行所需的推理配置名（`fold_box_normal_subtask_8_12` 等）保持不变，
`scripts/start_server.sh` 依赖这些名字。其余文件逐字节一致。

## 同步方法（维护者用）

源项目有更新时：按上表范围重新拷贝，重跑单测与 infer-once 冒烟，并重新应用
`config.py` / `README.md` 的路径脱敏（占位符替换）。
