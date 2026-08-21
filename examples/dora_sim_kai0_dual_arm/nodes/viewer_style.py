#!/usr/bin/env python3
"""人眼查看降亮节点：仅供 image_viewer 使用，不参与策略数据流。

场景按真实采集画面还原后（纯白桌面+鲜黄纸箱）原始渲染偏亮（top 均值 ~170/255），
人眼观看刺眼；策略依赖该亮度分布（训练分布），不能动场景。故在查看器路径上加
统一降亮（×0.75），与策略输入（camera_style 补偿后画面）完全解耦。
"""

from __future__ import annotations

import sys

import numpy as np
from dora import Node
from forge_msgs import Image

GAIN = 0.75


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
        dim = np.clip(frame.astype(np.float64) * GAIN, 0, 255).astype(np.uint8)
        msg = Image.from_numpy(dim, encoding="rgb8")
        node.send_output(input_id, msg.to_arrow())
    return 0


if __name__ == "__main__":
    sys.exit(main())
