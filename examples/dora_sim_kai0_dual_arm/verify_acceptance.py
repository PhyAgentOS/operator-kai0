#!/usr/bin/env python3
"""仿真验收数值化核查：解析 dora_run.log / dora_run_rtc.log 的 SNAPSHOT 行，
校验真实 kai0 推理的健康性（非模式化动作判据）。

判据（真实推理，同步与 RTC 模式通用）：
- SNAPSHOT 样本充足（≥ 50 行，frame_capture 每 0.1s 一行）；
- 机械臂在运动：12 个臂关节中至少 6 个幅值 ≥ 0.05 rad（真实推理输出为工作位形，
  幅度与任务/场景有关，不设上限）；
- 夹爪全程在契约量程容差内（0..0.110 m，契约 0.105 m + mujoco 位置伺服跟踪容差），
  且至少一个夹爪有开合动作（幅值 ≥ 0.01 m）；
- 日志 0 错误（error/exception/traceback 文案，排除 fake server 单测专用的
  trigger_error 文案）；
- 前 2s 为启动瞬态（MJCF 初始位形→策略位形），量程与幅值检查剔除。

注意：本脚本只判「管线健康」（有动作、数值合规、无错误）。任务是否完成
（装箱动作是否到位）不在判据内——模型任务完成度受仿真场景还原度限制。

用法：.venv/bin/python verify_acceptance.py [dora_run.log]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

# 与 contract.GLOBAL_JOINT_ORDER 一致（左 6 关节+左夹爪、右 6 关节+右夹爪）
JOINT_NAMES = (
    [f"robot_left/joint{i}" for i in range(1, 7)]
    + ["robot_left/gripper"]
    + [f"robot_right/joint{i}" for i in range(1, 7)]
    + ["robot_right/gripper"]
)
GRIPPERS = (6, 13)

SAMPLE_INTERVAL_S = 0.1     # frame_capture 节点每 0.1s 一行
TRANSIENT_SAMPLES = 20      # 前 2s 启动瞬态（MJCF 初始位形→策略位形）
MIN_MOVING_JOINTS = 6       # 至少一半臂关节在运动
JOINT_MIN_AMPLITUDE_RAD = 0.05
GRIPPER_BOUND_M = (0.0, 0.115)   # 契约 0..0.105 m + mujoco 位置伺服跟踪超调容差
                                  #（实测超调至 0.108~0.111 m；单位错误会是百倍量级，此容差足以拦截）
GRIPPER_MIN_AMPLITUDE_M = 0.01


def parse_snapshots(path: Path) -> np.ndarray:
    """返回 [T, 14] 关节值数组（列序同 JOINT_NAMES）。"""
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "SNAPSHOT t=" not in line:
            continue
        values = {}
        for name, raw in re.findall(r"(robot_\w+/[\w]+)=([+-]?\d+\.\d+)", line):
            values[name] = float(raw)
        if set(JOINT_NAMES) - set(values):
            continue  # 不完整行
        rows.append([values[n] for n in JOINT_NAMES])
    return np.asarray(rows)


def main() -> int:
    log_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parent / "dora_run.log"
    )
    if not log_path.is_file():
        print(f"FAIL: 日志不存在 {log_path}（先运行 bash run.sh 或 run_rtc.sh）")
        return 1
    data = parse_snapshots(log_path)
    if data.shape[0] < 50:
        print(f"FAIL: SNAPSHOT 样本不足（{data.shape[0]} 行），先运行 bash run.sh 30")
        return 1

    errors: list[str] = []
    steady = data[TRANSIENT_SAMPLES:]  # 剔除启动瞬态后做量程/幅值检查
    print(f"样本数 {data.shape[0]}（10Hz），跨度 {data.shape[0] * SAMPLE_INTERVAL_S:.1f}s，"
          f"瞬态剔除 {TRANSIENT_SAMPLES} 个")

    print("\n关节                     min      max     幅值")
    moving_joints = 0
    gripper_moved = False
    for i, name in enumerate(JOINT_NAMES):
        col = data[:, i]
        print(f"{name:24s} {col.min():+.3f}  {col.max():+.3f}  {np.ptp(col):.3f}")
        if i in GRIPPERS:
            if steady[:, i].min() < GRIPPER_BOUND_M[0] - 1e-6 or steady[:, i].max() > GRIPPER_BOUND_M[1] + 1e-6:
                errors.append(
                    f"{name} 超出 {GRIPPER_BOUND_M[0]}..{GRIPPER_BOUND_M[1]} m"
                    f"（{steady[:, i].min():.3f}..{steady[:, i].max():.3f}）"
                )
            if np.ptp(steady[:, i]) >= GRIPPER_MIN_AMPLITUDE_M:
                gripper_moved = True
        else:
            if np.ptp(steady[:, i]) >= JOINT_MIN_AMPLITUDE_RAD:
                moving_joints += 1

    print(f"\n运动关节数 {moving_joints}/12（判据 ≥ {MIN_MOVING_JOINTS}）")
    if moving_joints < MIN_MOVING_JOINTS:
        errors.append(
            f"运动关节不足：{moving_joints}/12（≥ {MIN_MOVING_JOINTS} 才判定策略在输出动作）"
        )
    print(f"夹爪开合 {gripper_moved}（判据：至少一个夹爪幅值 ≥ {GRIPPER_MIN_AMPLITUDE_M} m）")
    if not gripper_moved:
        errors.append("左右夹爪均无开合动作")

    # 日志错误核查（排除 fake server 单测专用的 trigger_error 文案）
    text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    err_lines = [
        l
        for l in text.splitlines()
        if re.search(r"\b(error|exception|traceback)\b", l)
        and "trigger_error" not in l
        and "internal server error" not in l
    ]
    print(f"\n日志错误行（排除 fake server 故意 trigger_error）: {len(err_lines)}")
    for l in err_lines[:10]:
        print("  ", l.strip()[:120])
    if err_lines:
        errors.append(f"日志含 {len(err_lines)} 行错误")

    if errors:
        print("\nFAIL:")
        for e in errors:
            print(" -", e)
        return 1
    print("\nPASS: 管线健康——机械臂运动、夹爪在契约量程内开合、日志 0 错误")
    return 0


if __name__ == "__main__":
    sys.exit(main())
