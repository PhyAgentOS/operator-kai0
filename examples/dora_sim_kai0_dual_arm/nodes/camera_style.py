#!/usr/bin/env python3
"""仿真画面风格补偿节点：对三路画面做逐通道线性增益，对齐训练相机画面分布。

背景（见 diagnosis/诊断记录_仿真机械臂不动.md 与相关探针实验）：
- 策略在真实相机画面分布下训练（中位亮度 ~150、白平衡中性、真实相机饱和度曲线）；
- mujoco 自然渲染无法用灯光/材质复现该分布（灯光整体抬升会填平阴影；亮度、白平衡、
  灯光色比、天空盒、材质逐一实验均无法触发策略——探针判据「50 步动作块臂关节最大
  幅度」始终 ~0.02 rad）；
- 对渲染帧做逐通道线性增益（RGB ×[1.7, 2.125, 1.751]）实测可把策略从「保持」分布
  稳定推向「执行」分布（判据 0.02 → 0.3~0.6 rad，多次重复采样验证）。
- 本节点为仿真示例专用（真实部署机用真实相机，不经过此节点）；不改任何框架/仓库代码，
  策略节点（kai0_runner）保持场景无关。

输入/输出：image/top、image/left、image/right（逐帧透传+增益，输出 id 不变）。
"""

from __future__ import annotations

import sys

import numpy as np
from dora import Node
from forge_msgs import Image

# RGB 通道增益（实测校准值，见诊断记录；存盘帧为 BGR 需注意，管道内为 RGB）
GAINS = np.array([1.7, 2.125, 1.751], dtype=np.float64)


def main() -> int:
    node = Node()
    for event in node:
        if event.get("kind") != "dora" or event["type"] != "INPUT":
            continue
        value = event.get("value")
        if value is None:
            continue
        input_id = event["id"]
        if not input_id.startswith("image/"):
            continue
        frame = Image.from_arrow(value).to_numpy()
        styled = np.clip(frame.astype(np.float64) * GAINS, 0, 255).astype(np.uint8)
        msg = Image.from_numpy(styled, encoding="rgb8")
        node.send_output(input_id, msg.to_arrow())
    return 0


if __name__ == "__main__":
    sys.exit(main())
