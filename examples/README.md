# examples/

kai0_runner 示例。策略节点本体是仓库 `inference/`（模型在独立推理服务端 `server/`，经 WebSocket
访问），示例只演示「策略节点嵌进完整 dataflow」的用法：

| 目录 | 说明 |
|---|---|
| `dora_sim_kai0_dual_arm/` | 双臂仿真示例：真实推理服务端 + kai0 策略节点 + 本地 MJCF 场景（含装箱道具：纸箱/青色物体/胶带/保护袋，风格补偿节点仅仿真） |
| `dora_real_kai0_dual_arm/` | 双臂真机示例：同一策略节点与服务端 + 真机 Piper 双臂 + 三路 USB 相机（夹爪单位桥 nodes/kai0_unit_bridge.py；设备参数现场填写） |

**仿真与真机共用同一策略节点与服务端**，只更换环境侧（仿真 mujoco → 真机 piper/相机）；
单位链差异（kai0 契约 Forge 米 ↔ Piper 毫米）由真机示例的单位桥节点处理，见其 README。
`scripts/start_server.sh` 是仿真与真机部署的同一启动脚本（内置动作空间安全检查）。

启动方式（仓库根执行）：

```bash
bash scripts/start_server.sh --ckpt ckpt/<任务名>/<step> --rtc   # 服务端（RTC，推荐）
bash scripts/start_sim.sh 90                                      # 仿真（默认 RTC）
bash scripts/start_sim.sh 90 --sync                               # 仿真（同步/调试）
bash scripts/start_real.sh                                        # 真机（安全确认后运行）
```
