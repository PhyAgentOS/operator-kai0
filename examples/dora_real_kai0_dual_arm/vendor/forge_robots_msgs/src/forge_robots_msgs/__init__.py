"""VENDOR：机器人消息类型（ActuatorValue / RobotAction / RobotState）。

API 对齐上游 robot_core 使用面；dora 边界统一到 forge-msgs 1.0.0 契约
（JointState / JointCommand 格式），见 VENDOR.md。
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from forge_msgs.arrow import ensure_record_batch

_ARM_UNIT = "radians"
_GRIPPER_UNIT = "millimeters"


class ActuatorValue:
    """单个执行器的值 + 元数据（与上游 forge_msgs 机器人消息同构）。"""

    __slots__ = ("value", "mode", "unit")

    def __init__(self, value: float, mode: str = "position", unit: str = "") -> None:
        self.value = float(value)
        self.mode = str(mode)
        self.unit = str(unit)

    def __repr__(self) -> str:  # pragma: no cover
        return f"ActuatorValue(value={self.value:.4f}, mode={self.mode}, unit={self.unit})"


def _unit_for(name: str) -> str:
    return _GRIPPER_UNIT if name == "gripper" else _ARM_UNIT


class RobotAction:
    """动作：actuator 名 -> ActuatorValue。dora 边界解析 JointCommand 格式。"""

    def __init__(self, actuators: dict[str, ActuatorValue]) -> None:
        self.actuators: dict[str, ActuatorValue] = dict(actuators)

    @classmethod
    def from_arrow(cls, data: Any, actuator_order: list[str]) -> "RobotAction":
        """解析 task_robot 输出的 JointCommand Arrow（name/mode/position/... 列）。"""
        batch = ensure_record_batch(data)
        names = list(batch["name"][0].as_py() or [])
        mode = batch["mode"][0].as_py() if "mode" in batch.schema.names else "position"
        field = "velocity" if mode == "velocity" else ("effort" if mode == "effort" else "position")
        values = batch[field][0].as_py() if field in batch.schema.names else []
        by_name = dict(zip(names, values, strict=False))
        actuators = {}
        for name in actuator_order:
            actuators[name] = ActuatorValue(
                value=by_name.get(name, 0.0),
                mode=mode,
                unit=_unit_for(name),
            )
        return cls(actuators=actuators)


class RobotState:
    """状态：actuator 名 -> ActuatorValue。dora 边界序列化为 JointState 格式。"""

    def __init__(self, actuators: dict[str, ActuatorValue]) -> None:
        self.actuators: dict[str, ActuatorValue] = dict(actuators)

    def to_arrow(self, order: list[str]) -> pa.RecordBatch:
        """输出为 task_robot 可解析的 JointState 格式（name/position/velocity/effort 列）。"""
        names = list(order)
        position = [self.actuators[n].value for n in names]
        return pa.RecordBatch.from_pydict(
            {
                "name": pa.array([names], type=pa.list_(pa.string())),
                "position": pa.array([position], type=pa.list_(pa.float64())),
                "velocity": pa.array([[]], type=pa.list_(pa.float64())),
                "effort": pa.array([[]], type=pa.list_(pa.float64())),
            }
        )

    @classmethod
    def from_arrow(cls, data: Any, actuator_order: list[str]) -> "RobotState":
        """解析 JointState 格式 Arrow（master_state 场景）。"""
        batch = ensure_record_batch(data)
        names = list(batch["name"][0].as_py() or [])
        values = batch["position"][0].as_py() if "position" in batch.schema.names else []
        by_name = dict(zip(names, values, strict=False))
        actuators = {}
        for name in actuator_order:
            actuators[name] = ActuatorValue(
                value=by_name.get(name, 0.0), mode="position", unit=_unit_for(name)
            )
        return cls(actuators=actuators)
