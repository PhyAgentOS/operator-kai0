#!/usr/bin/env python3
"""Policy Dora 节点：订阅 proprio_state + 多路 image，输出 JointCommand。

策略适配器经 registry 创建（DummyAdapter / 同步 WS 客户端 / RTC 客户端）。
"""

from __future__ import annotations

import logging
import sys

import numpy as np
from forge_msgs import JointCommand
from forge_policy import run_dora_policy_node

from kai0_inference.inference.config import load_config
from kai0_inference.inference.policies.registry import create_policy_adapter

logger = logging.getLogger(__name__)


def _build_joint_command(action_np, config) -> JointCommand:
    """动作数组 → JointCommand（契约 profile 见 config.contract；关节名顺序 = config.joints）。"""
    names = config.joint_order
    action = config.contract.validate_action(action_np)
    if len(action) != len(names):
        # validate_action 已保证契约维度；此检查兜底 config 与契约脱钩的场景
        raise ValueError(
            f"policy action length={len(action)} does not match action joints={len(names)}"
        )

    # 夹爪量程告警（双臂契约 0..0.105 m；LIBERO 契约不校验夹爪范围）：
    # 不拦截、不静默修改，仅记录便于真机安全排查
    for idx, name, value in config.contract.gripper_range_violations(action):
        logger.warning(
            "action gripper %s=%s out of range %s (meters)",
            name,
            value,
            config.contract.gripper_range_m,
        )

    position = [0.0] * len(names)
    velocity = [0.0] * len(names)
    effort = [0.0] * len(names)

    for i, joint in enumerate(config.joints):
        value = float(action[i])
        mode = joint.mode
        if mode == "velocity":
            velocity[i] = value
        elif mode == "effort":
            effort[i] = value
        else:
            position[i] = value

    return JointCommand(name=names, position=position, velocity=velocity, effort=effort)


def run_infer(args) -> int:
    """执行 Dora 在线推理；args 需含可选 config。"""
    # Dora 节点以独立进程运行；不配置 logging 时 INFO 日志会被 root logger
    # 丢弃（默认 WARNING）。与其它 Forge 节点（robot_router / mujoco_sim）
    # 一致，Dora 模式将 root logger 配为 INFO。
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(config_path=getattr(args, "config", None))
    policy_config = config.runtime_policy_config()
    policy = create_policy_adapter(policy_config)
    image_inputs = config.image_inputs_for(policy.required_image_keys)
    try:
        return run_dora_policy_node(
            policy,
            joint_order=config.state_joint_order,
            image_input_id_to_alias=image_inputs,
            alias_for_cameras=list(image_inputs.values()),
            build_action=lambda action_np: _build_joint_command(action_np, config),
            policy_id=str(policy_config.get("policy_id", "default")),
            auto_start=bool(policy_config.get("auto_start", False)),
            call_lifecycle_hooks=True,
        )
    finally:
        stop = getattr(policy, "stop", None)
        if callable(stop):
            stop()


def main() -> int:
    """兼容旧入口：转发到统一 CLI 的 infer 子命令。"""
    from kai0_inference.cli import main_infer

    return main_infer()


if __name__ == "__main__":
    sys.exit(main())
