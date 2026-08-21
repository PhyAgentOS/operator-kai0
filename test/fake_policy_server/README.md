# fake-policy-server

WebSocket 策略服务模拟器，与 openpi `websocket_policy_server.py` 协议逐字段对齐。
用于在无 checkpoint 权重、无 GPU 的条件下验证 kai0 链路。

> 定位：本工具是**长期测试资产**，不是一次性脚本：作为 WS Adapter 单测与仿真联调的
> 假服务端，与真实服务端协议逐字段对齐（见下方清单）。

## 安装

```bash
# 独立环境（websockets/msgpack/numpy，与 kai0_runner 主环境分离）
python3 -m venv .venv-fake
.venv-fake/bin/pip install -e test/fake_policy_server
```

## 使用

```bash
.venv-fake/bin/python -m fake_policy_server --port 8000 --mode sine

# 另一个终端，用 openpi 仓库的 smoke 测试验证：
#   python scripts/smoke_test_server.py --url ws://127.0.0.1:8000
```

预期输出：

```text
server metadata keys: ['mode', 'server']
actions shape: (50, 14)
VLA server smoke test passed; no robot command was sent.
```

## CLI 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--host` | 0.0.0.0 | 监听地址 |
| `--port` | 8000 | 监听端口 |
| `--mode` | zero | `zero`=全零动作；`sine`=逐关节正弦（目视验证顺序映射用） |
| `--action-dim` | 14 | 动作维度 |
| `--chunk-size` | 50 | 动作块长度 |
| `--default-task-index` | 无 | 初始 sticky prompt_id（模拟 `DEFAULT_PROMPT_ID` 环境变量） |
| `--infer-delay-ms` | 0 | 每次推理模拟耗时，供时序相关测试使用 |

## 协议对齐清单（对照本仓库 `server/` 快照的 `websocket_policy_server.py`）

| 行为 | 真实服务端 | fake server |
|---|---|---|
| 连接后先发 metadata 帧 | ✓ | ✓（默认含 server/mode 字段） |
| 请求-响应循环（msgpack 帧） | ✓ | ✓ |
| sticky task 值（prompt_id/task_index/subtask） | ✓ | ✓（白盒观测点 `server.last_task_value`） |
| 响应附加 `server_timing.infer_ms` | ✓ | ✓ |
| 第二次请求起附 `prev_total_ms` | ✓ | ✓ |
| 异常发送文本帧 traceback + 1011 关闭 | ✓ | ✓（测试钩子：obs 携带 `trigger_error` 触发） |
| `/healthz` 返回 `200 OK\n` | ✓ | ✓ |
| `compression=None`、`max_size=None` | ✓ | ✓ |
| 响应含 `state` / `actions` / `policy_timing` | ✓ | ✓（state 为空数组占位） |
| RTC 附加字段（prev_action_chunk/inference_delay/execute_horizon） | ✓（传入 policy） | ✓（容忍并忽略） |

## 测试

```bash
.venv-fake/bin/pytest test/fake_policy_server/tests -v
```

测试全部通过真实 socket（websockets 同步客户端 ↔ 异步服务端），不使用 mock。
