# 推理配置

kai0_runner 推理配置模板，按适配器类型拆分：

| 文件 | `policy.type` | 说明 |
|---|---|---|
| `kai0_rtc.yaml` | `kai0_rtc` | **推荐默认**：采用 RTC 模式（常驻连接 + 后台推理线程 + 动作流裁剪混合）；服务端须以 `--rtc` 启动 |
| `kai0.yaml` | `kai0` | 不采用 RTC 模式（短连接 + chunk 摊销），调试/回退用；服务端不加 `--rtc` |
| `dummy.yaml` | `dummy` | 无服务端链路调试（DummyAdapter，不连网），用作离线冒烟 |

使用：

```bash
uv run kai0 infer-once --config config/inference/<name>.yaml
```

配置分三块（见仓库 README「配置」节与文件内注释）：

| 块 | 说明 |
|---|---|
| `joints` | 14 维动作布局：左臂 6 关节 + 左夹爪、右臂 6 关节 + 右夹爪（顺序即动作顺序） |
| `policy` | 适配器类型 / `server_url` / `auto_start` / `instruction` / `timeout_s`，RTC 参数组（`rtc_*`，`rtc_control_rate` 须与节点 tick 频率一致） |
| `image_inputs` | 哪路 Dora 图像对应服务端哪个相机（`top_head` / `hand_left` / `hand_right`） |

要点：`auto_start` 真机置 `false`（等 gateway 命令），仿真无 gateway 置 `true`；
`instruction` 在 `PROMPT_MODE=fixed` 下传空串（服务端强制用配置 default_prompt）。

单位换算（夹爪服务端毫米 ↔ Forge 米）与相机别名契约集中在 `inference/contract.py`。
