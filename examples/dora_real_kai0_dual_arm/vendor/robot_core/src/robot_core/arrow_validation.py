"""VENDOR 修复：RobotAction / RobotState 控制类 Arrow 输入的 schema 校验。

上游（forge_robots commit 9c69078）按「每 actuator 一列 + mode/unit 元列」的
RobotAction 格式校验；本 vendor 的 dora 边界统一到 forge-msgs 1.0.0 契约
（task_robot 输出 JointCommand：name/mode/position/... 列），故校验规则改为：

- 必须存在 ``name`` 列，且 ``actuator_order`` 中每个执行器名都在 name 值中出现；
- 必须存在 ``mode`` 列（JointCommand 输出固定带）。

其余语义（避免静默填 0、错误提示可排查）保持不变。
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from forge_robots_msgs.value import ensure_record_batch

_META_COLUMNS = frozenset({"mode", "unit"})


class RobotArrowSchemaError(ValueError):
    """Arrow RecordBatch 不满足当前机器人 actuator_order 约定时抛出。"""


def validate_robot_control_arrow(
    value: Any,
    actuator_order: list[str],
    *,
    strict_extra_columns: bool = False,
) -> pa.RecordBatch:
    """
    将 dora 传入的 payload 规范为 RecordBatch，并校验列名。

    要求存在 ``name``、``mode`` 列，且 ``actuator_order`` 中每个执行器名都在
    name 值中出现（JointCommand 格式，task_robot 输出）。

    Args:
        value: ``RecordBatch`` / ``Table`` / ``StructArray`` / ``bytes``（与 ``from_arrow`` 一致）。
        actuator_order: 本节点期望的执行器顺序与名称。
        strict_extra_columns: 为 True 时，除 ``name``、``mode`` 及值列外不允许多余列。

    Returns:
        规范化后的 ``RecordBatch``（调用方可传给 ``from_arrow``，其内部会再次 ``ensure``）。

    Raises:
        RobotArrowSchemaError: 列不满足约定时。
    """
    if not actuator_order:
        raise RobotArrowSchemaError("actuator_order 不能为空")

    try:
        batch = ensure_record_batch(value)
    except TypeError as e:
        raise RobotArrowSchemaError(str(e)) from e

    available = frozenset(batch.schema.names)
    if "name" not in available:
        raise RobotArrowSchemaError(
            f"缺少必需列 name（JointCommand 格式）；available={sorted(available)!r}"
        )
    if "mode" not in available:
        raise RobotArrowSchemaError(
            f"缺少必需列 mode（JointCommand 格式）；available={sorted(available)!r}"
        )

    names = list(batch["name"][0].as_py() or [])
    missing = [n for n in actuator_order if n not in names]
    if missing:
        raise RobotArrowSchemaError(
            f"name 值缺少执行器 {missing}；actuator_order={list(actuator_order)!r}"
        )

    if strict_extra_columns:
        value_columns = {"position", "velocity", "effort", "kp", "kd"} | _META_COLUMNS | {"name"}
        extra = available - value_columns
        if extra:
            raise RobotArrowSchemaError(f"存在未声明的列 {sorted(extra)!r}")

    return batch
