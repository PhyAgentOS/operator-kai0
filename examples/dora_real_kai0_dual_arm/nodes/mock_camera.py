#!/usr/bin/env python3
"""Mock USB 相机节点（无硬件软件在环）：输出固定尺寸的渐变+噪声帧。

与真机 usb_camera 节点同契约：tick 驱动、输出 id 为 `image`（forge_msgs.Image，
rgb8）。用于 mock 模式下验证 task_robot 图像转发与策略图像输入链路。
"""

from __future__ import annotations

import sys

import numpy as np
from dora import Node
from forge_msgs import Image

WIDTH, HEIGHT = 640, 480


def main() -> int:
    seed = 0
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1])
    rng = np.random.default_rng(seed)

    # 渐变底（每路实例 seed 不同 → 三路画面可区分）
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    base = np.stack([x / WIDTH, y / HEIGHT, (x + y) / (WIDTH + HEIGHT)], axis=-1)

    node = Node()
    for event in node:
        if event.get("kind") != "dora" or event["type"] != "INPUT":
            continue
        if event["id"] == "tick":
            noise = rng.integers(0, 12, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
            frame = np.clip(base * 255 + noise, 0, 255).astype(np.uint8)
            node.send_output("image", Image.from_numpy(frame, "rgb8").to_arrow())

    return 0


if __name__ == "__main__":
    sys.exit(main())
