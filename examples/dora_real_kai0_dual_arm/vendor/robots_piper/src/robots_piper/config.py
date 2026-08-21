"""Piper 节点配置加载与校验。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# 与 driver 中关节/夹爪顺序一致，固定不可配置
PIPER_ACTUATOR_ORDER = [
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"
]


@dataclass
class PiperNodeConfig:
    """Piper 节点配置。"""

    """与 task_robot 中 robots[].id 对应，用于 dataflow 中 input/output 的语义一致。"""
    port: str
    is_follower: bool
    debug: bool = False
    # kai0 新增（参考 real_deploy 定型口径）：启动回零——连接后不一次性冲零
    # （PiperDriver auto_set_safety_position=False），改为从当前反馈位置 smoothstep
    # 插值回零，完成前不进入节点循环。默认值与参考配置一致（8s / 25Hz）。
    startup_return_to_zero: bool = False
    startup_zero_duration_s: float = 8.0
    startup_zero_rate_hz: float = 25.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PiperNodeConfig":
        """从字典解析配置。"""
        port = data.get("port", "can0")
        is_follower = data.get("is_follower", True)
        debug = data.get("debug", False)
        return cls(
            port=port,
            is_follower=is_follower,
            debug=debug,
            startup_return_to_zero=bool(data.get("startup_return_to_zero", False)),
            startup_zero_duration_s=float(data.get("startup_zero_duration_s", 8.0)),
            startup_zero_rate_hz=float(data.get("startup_zero_rate_hz", 25.0)),
        )

    @classmethod
    def from_yaml_path(cls, path: str | Path) -> "PiperNodeConfig":
        """从 YAML 文件加载配置。"""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在: {p}")
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data:
            raise ValueError(f"配置文件为空: {p}")
        return cls.from_dict(data)


def load_config(
    config_path: str | Path | None = None,
    dataflow_config: dict[str, Any] | None = None,
) -> PiperNodeConfig:
    """
    加载配置，支持多种来源（优先级从高到低）：
    1. dataflow_config 中的 config_path 或内联 config
    2. config_path 参数
    3. 环境变量 PIPER_NODE_CONFIG

    Args:
        config_path: 可选，YAML 配置文件路径
        dataflow_config: 可选，dora 节点 config 字段

    Returns:
        PiperNodeConfig
    """
    data: dict[str, Any] | None = None

    if dataflow_config:
        if "config" in dataflow_config:
            data = dataflow_config["config"]
        elif "config_path" in dataflow_config:
            return PiperNodeConfig.from_yaml_path(dataflow_config["config_path"])

    path = config_path
    if path is None and data is None:
        path = os.environ.get("PIPER_NODE_CONFIG")

    if path:
        return PiperNodeConfig.from_yaml_path(path)

    if data:
        return PiperNodeConfig.from_dict(data)

    raise ValueError(
        "未找到配置。请设置 PIPER_NODE_CONFIG 环境变量、"
        "或通过 --config 指定配置文件、或使用 dataflow config。"
    )
