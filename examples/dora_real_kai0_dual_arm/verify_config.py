#!/usr/bin/env python3
"""真机示例配置契约校验（无硬件可跑）：dataflow / task_robot / kai0_rtc / contract 一致性。

校验项：
1. task_robot.yaml 的 14 关节顺序 == kai0_rtc.yaml state_joints == contract.GLOBAL_JOINT_ORDER；
2. 夹爪单位：kai0_rtc（Forge 米）↔ task_robot（Piper 毫米）——单位桥存在的依据；
3. task_robot robots[].actuator_order == Piper 固定契约（joint1..joint6,gripper）；
4. dataflow 的 kai0_policy tick 频率 == kai0_rtc rtc_control_rate（真机 50ms=20Hz）；
5. kai0_rtc image_inputs 键 ⊆ task_robot image_forward 键。

运行：uv run python verify_config.py（示例目录 venv）
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from kai0_inference.inference import contract
from kai0_inference.inference.config import load_config

HERE = Path(__file__).resolve().parent
PIPER_ACTUATOR_ORDER = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = "OK " if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(name)


def main() -> int:
    task_cfg = yaml.safe_load((HERE / "task_robot.yaml").read_text())
    policy_cfg = load_config(config_path=str(HERE / "kai0_rtc.yaml"))

    # 1) 关节顺序三方一致
    task_joints = [j["name"] for j in task_cfg["joints"]]
    check(
        "joints: task_robot == kai0_rtc == contract.GLOBAL_JOINT_ORDER",
        task_joints == policy_cfg.state_joint_order == list(contract.GLOBAL_JOINT_ORDER),
    )

    # 2) 夹爪单位链（单位桥依据）
    def joint_attr(j, key: str):
        return j[key] if isinstance(j, dict) else getattr(j, key)

    def gripper_unit(cfg_joints) -> str:
        for j in cfg_joints:
            if joint_attr(j, "name").endswith("/gripper"):
                return joint_attr(j, "unit")
        raise AssertionError("joints 缺少 gripper")

    check(
        "gripper 单位链：kai0_rtc 米 ↔ task_robot 毫米（bridge 依据）",
        gripper_unit(policy_cfg.joints) == "meters"
        and gripper_unit(task_cfg["joints"]) == "millimeters",
    )

    # 3) actuator_order 与 Piper 固定契约一致
    check(
        "task_robot robots actuator_order == Piper 固定契约",
        all(
            r["actuator_order"] == PIPER_ACTUATOR_ORDER
            for r in task_cfg["robots"]
        ),
        "PIPER_ACTUATOR_ORDER: joint1..joint6, gripper",
    )

    # 4) tick 频率 == rtc_control_rate（20Hz → 50ms）：取 kai0_policy 节点自身的 tick
    flow = yaml.safe_load((HERE / "dataflow.yaml").read_text())
    policy_node = next(n for n in flow["nodes"] if n["id"] == "kai0_policy")
    tick_ms = int(policy_node["inputs"]["tick"].rsplit("/", 1)[1])
    control_rate = float(policy_cfg.policy["rtc_control_rate"])
    check(
        "dataflow kai0_policy tick == rtc_control_rate（50ms == 20Hz）",
        tick_ms == round(1000.0 / control_rate),
        f"tick={tick_ms}ms vs rtc_control_rate={control_rate}Hz",
    )

    # 5) 图像输入 ⊆ task_robot 转发
    image_keys = policy_cfg.image_input_ids
    forwarded = set(task_cfg.get("image_forward", {}))
    check(
        "kai0_rtc image_inputs ⊆ task_robot image_forward",
        image_keys <= forwarded,
        f"缺失: {sorted(image_keys - forwarded)}",
    )

    # 6) 启动回零（参考 real_deploy 定型口径）：双臂均开启插值回零且 duration/rate
    #    为正；开启时 piper_follower 会以 auto_set_safety_position=False 连接
    #    （连接即冲零被关闭，改由 nodes/startup_zero.py 插值接管）。
    for arm in ("piper_slave_left", "piper_slave_right"):
        arm_cfg = yaml.safe_load((HERE / f"{arm}.yaml").read_text())
        check(
            f"{arm} 启动插值回零已开启且 duration/rate 为正",
            bool(arm_cfg.get("startup_return_to_zero"))
            and float(arm_cfg.get("startup_zero_duration_s", 0)) > 0
            and float(arm_cfg.get("startup_zero_rate_hz", 0)) > 0,
            "startup_return_to_zero: true + startup_zero_duration_s/rate_hz > 0",
        )

    if failures:
        print(f"\n共 {len(failures)} 项失败")
        return 1
    print("\n全部校验通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
