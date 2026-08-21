#!/usr/bin/env python3
"""仿真取证节点：周期记录 14 维关节数值 + 保存三路图像快照。

- proprio_state（robot_router 透传，50Hz）：每 1s 打印一行 14 维关节值
  （SNAPSHOT 前缀），用于数值化验证逐关节正弦频率两两不同（左快右慢判据）；
- 三路图像：每 5s 存 PNG 到 out/frames/（与 image_viewer 同源，目视证据）。
仅本地验收取证用，不改动主数据流（kai0_policy → robot_router → mujoco）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from dora import Node
from forge_msgs import Image, JointState

FRAMES_DIR = Path(__file__).resolve().parents[1] / "out" / "frames"
STATE_EVERY = 5    # 50Hz proprio → 每 0.1s 一行（10Hz 采样，左臂 0.4–0.9s 周期不混叠）
FRAME_EVERY = 250  # 每 5s 一次图像快照


def main() -> int:
    node = Node()
    images: dict[str, np.ndarray] = {}
    state_count = 0
    for event in node:
        if event.get("kind") != "dora" or event["type"] != "INPUT":
            continue
        value = event.get("value")
        if value is None:
            continue
        input_id = event["id"]

        if input_id == "proprio_state":
            state_count += 1
            if state_count % STATE_EVERY == 0:
                state = JointState.from_arrow(value)
                pairs = ", ".join(
                    f"{n}={p:+.4f}" for n, p in zip(state.name, state.position)
                )
                # t 以 0.1s 为单位（state_count // STATE_EVERY）
                print(f"SNAPSHOT t={state_count // STATE_EVERY} | {pairs}", flush=True)
            if state_count % FRAME_EVERY == 0 and len(images) == 3:
                tag = f"t{state_count // FRAME_EVERY * 5:02d}s"
                FRAMES_DIR.mkdir(parents=True, exist_ok=True)
                for topic, frame in images.items():
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(FRAMES_DIR / f"{topic.replace('/', '_')}_{tag}.png"), bgr)
                print(f"SNAPSHOT frames saved to {FRAMES_DIR} ({tag})", flush=True)
        elif input_id.startswith("image/"):
            images[input_id] = Image.from_arrow(value).to_numpy()

    return 0


if __name__ == "__main__":
    sys.exit(main())
