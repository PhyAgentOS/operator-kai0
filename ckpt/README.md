# Checkpoint 权重放置约定

本仓库自包含推理服务端（`server/kai0_prompt_schedule/`，脚本 `scripts/start_server.sh`）。
权重由部署人员单独分发，**不提交到任何代码仓库**，放置在本目录下：

```text
ckpt/fold_box_0721_0722_from_10000/   ← 本仓库当前放置的权重示例
├── norm_stats.json
└── 6000/
    └── params/
```

- `<step>`（如 `6000/`）为 checkpoint step 目录，必须直接包含 `params/`；
- `norm_stats.json` 位于 step 目录的**父目录**；
- 目录名可自定，仅用于组织，不影响加载；
- 模型位于其它位置时，用 `scripts/start_server.sh` 的 `--ckpt` 指向**直接包含
  `params/` 的 step 目录**即可，无需移动文件。

## 模型来源与许可证

- 基础模型：**Pi0.5**（openpi 系，`Physical-Intelligence/openpi`；服务端快照版本
  0.1.0，见 `server/SOURCE.md`）；
- 部署权重：由部署人员分发、不随仓库提交（本目录只放放置约定）；
- tokenizer：**PaliGemma**（首次启动自动下载到 `OPENPI_DATA_HOME`，见下）；
- 许可证：openpi **Apache-2.0**（`server/kai0_prompt_schedule/LICENSE`）与 Gemma
  **Terms of Use**（`server/kai0_prompt_schedule/LICENSE_GEMMA.txt`）。

## 启动服务端

```bash
# 1) 首次：安装服务端环境（有网环境一键安装，python3.12 + JAX/CUDA 依赖，需数分钟）
bash server/setup.sh

# 2) 启动（--rtc 为推荐默认路径；delta 训练的 checkpoint 加 --delta，LoRA 加 --lora）
bash scripts/start_server.sh --ckpt ckpt/<任务名>/<step> --rtc
```

- 脚本内置动作空间安全检查：读取 `norm_stats.json` 的臂关节动作均值，与配置的
  `_delta` 后缀一致性校验，不一致直接拒绝启动（防止配错后机械臂高速冲向零位）；
- 首次启动会从 `gs://big_vision` 自动下载 tokenizer 到 `OPENPI_DATA_HOME`
  （默认 `<仓库>/cache/openpi`，gitignored）；离线环境可手工放置
  `paligemma_tokenizer.model` 到该目录后重启；
- GPU 需要 NVIDIA 驱动 + CUDA 12 支持（服务端依赖 `jax[cuda12]`）；
- 服务端代码快照来源与同步方法见 `server/SOURCE.md`。

## 客户端接入

服务端就绪后，客户端侧（策略节点）接入步骤见本仓库 README「启动推理服务端」。
