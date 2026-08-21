"""kai0 策略契约集中定义（防屎山原则）。

动作布局、关节命名、夹爪量程、相机 alias、图像尺寸、state 维度、单位换算等
契约只在此定义一次；adapter / build_action / 配置 / 测试一律引用本模块，
禁止在其它文件中散落字面量。

多契约（D2 修订版，2026-08-20）：自「单一 14 维双臂常量」演进为契约 profile——
- `DUAL_ARM`：原双臂契约（值、语义、行为全部不变；模块级常量与函数保留为
  其向后兼容别名，既有调用点与单测不受影响）；
- `LIBERO`：π0.5-libero 契约（7 维动作、8 维 state、两路相机、夹爪 [-1,1]
  原生域不做米↔毫米换算）。
新接入的契约场景（policy.type=kai0_libero）通过 `get_contract` 选择 profile，
校验下沉为 `PolicyContract` 方法。

契约来源：
- 14 维动作布局与夹爪索引（%7==6）：kai0 服务端数据契约（技术方案 §5）
- 关节全局名与顺序：robot_router 14 维全局契约（双臂仿真数据流验证）
- 夹爪量程 0..0.105 m（两指总开口）：robot_router / 双臂 MJCF（仿真数据流验证）
- 相机 alias：openpi 服务端配置（top_head / hand_left / hand_right）
- 图像尺寸 224×224：openpi 模型输入契约；缩放由服务端 transforms 负责，节点原样透传
- LIBERO 契约：openpi LiberoInputs/LiberoOutputs（state 8D = eef_pos(3)+axis_angle(3)+
  gripper_qpos(2)；action 7D delta，夹爪 q01/q99 = ±1.0 原生域；相机键
  observation/image 与 observation/wrist_image）——证据见项目文档
  `docs/契约复核结论.md` §4 与官方 norm_stats（sha256 见规划 §4.4）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# —— 双臂动作契约（原 14 维常量，DUAL_ARM profile 的组成部分）——
ACTION_DIM = 14
_ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
# 顺序：左 6 关节 + 左夹爪、右 6 关节 + 右夹爪（与 kai0 服务端、robot_router 一致）
LEFT_JOINT_NAMES = tuple(f"robot_left/{n}" for n in _ARM_JOINT_NAMES) + ("robot_left/gripper",)
RIGHT_JOINT_NAMES = tuple(f"robot_right/{n}" for n in _ARM_JOINT_NAMES) + ("robot_right/gripper",)
GLOBAL_JOINT_ORDER = LEFT_JOINT_NAMES + RIGHT_JOINT_NAMES

# 夹爪维度索引：左臂 6、右臂 13（满足 idx % 7 == 6，与服务端布局一致）
GRIPPER_INDICES = frozenset(
    i for i, name in enumerate(GLOBAL_JOINT_ORDER) if name.endswith("/gripper")
)
# 夹爪两指总开口量程（米），robot_router 契约
GRIPPER_RANGE_M = (0.0, 0.105)

# DummyAdapter 固定动作值：关节 0、夹爪开 0.03 m（在量程内）
DUMMY_JOINT_POSITION = 0.0
DUMMY_GRIPPER_OPEN_M = 0.03

# —— 双臂服务端单位契约（kai0 训练数据域，与 Forge 侧不同）——
# 服务端夹爪维度单位为毫米（两指总开口 0..105 mm，旧客户端数据采集域；
# 佐证：old forge piper driver gripper actuator max_value=105.0 mm）；
# Forge 侧夹爪单位为米（0..0.105 m，robot_router / agilex_piper driver 契约）。
# 节点在组包（观测→服务端）与拆包（服务端动作块→JointCommand）时按此换算；
# 其余 12 个关节维度单位为弧度，两侧一致、原样透传。
SERVER_GRIPPER_UNIT = "millimeters"
GRIPPER_MM_PER_M = 1000.0

# —— 双臂观测契约 ——
# kai0 服务端识别的三路相机 alias
CAMERA_ALIASES = ("top_head", "hand_left", "hand_right")
# 模型图像输入尺寸 224×224（openpi 契约）。缩放由服务端 transforms 负责：
# Policy.infer 应用训练输入 transforms（ResizeImages(224, 224) + resize_with_pad），
# 节点侧不做 resize，任意分辨率原样透传。
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
# 仿真 dataflow 的 dora 图像输入 id → kai0 alias 默认映射
DEFAULT_IMAGE_INPUT_ID_TO_ALIAS = {
    "image/top": "top_head",
    "image/left": "hand_left",
    "image/right": "hand_right",
}

# —— LIBERO 契约常量（π0.5-libero，openpi LiberoInputs/LiberoOutputs）——
LIBERO_ACTION_DIM = 7
LIBERO_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
LIBERO_GRIPPER_INDEX = 6  # action 第 7 维 = 夹爪（robosuite 原生域 [-1, 1]）
# 服务端 payload 中的两路相机键（LiberoInputs：observation/image → base_0_rgb、
# observation/wrist_image → left_wrist_0_rgb；右腕零填充）
LIBERO_CAMERA_ALIASES = ("image", "wrist_image")
LIBERO_STATE_DIM = 8  # eef_pos(3) + axis_angle(3) + gripper_qpos(2)


@dataclass(frozen=True)
class PolicyContract:
    """一个推理契约 profile：动作/观测/相机/单位换算的完整口径。"""

    name: str
    action_dim: int
    joint_names: tuple[str, ...]
    gripper_indices: frozenset[int]
    camera_aliases: tuple[str, ...]
    state_dim: int
    # 夹爪量程（动作侧校验与告警）；None = 不校验（如 LIBERO [-1,1] 原生域）
    gripper_range_m: tuple[float, float] | None
    # 组包（state→服务端）乘、拆包（服务端动作→Forge）除；1.0 = 原样透传
    gripper_scale_to_server: float
    # 服务端 payload 形态："images_dict"（双臂：{"state","images":{alias:...},"prompt"}）
    # 或 "flat_observation_keys"（LIBERO：{"observation/image",...} 平铺键，LiberoInputs 直读）
    server_payload_style: str

    def validate_action(self, action) -> np.ndarray:
        """校验并返回 1-D 动作数组：维度必须为 action_dim 且数值有限。"""
        arr = np.asarray(action, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"policy action must be 1-D, got shape={arr.shape}")
        if len(arr) != self.action_dim:
            raise ValueError(
                f"policy action length={len(arr)} does not match "
                f"ACTION_DIM={self.action_dim}（contract {self.name}）"
            )
        if not np.isfinite(arr).all():
            raise ValueError("policy action must contain only finite values")
        return arr

    def gripper_range_violations(self, action) -> list[tuple[int, str, float]]:
        """返回超出夹爪量程的 (索引, 关节名, 值) 列表；空列表表示合规。"""
        if self.gripper_range_m is None:
            return []
        arr = np.asarray(action)
        low, high = self.gripper_range_m
        return [
            (i, self.joint_names[i], float(arr[i]))
            for i in self.gripper_indices
            if not (low <= float(arr[i]) <= high)
        ]

    def scale_grippers(self, value, factor: float) -> np.ndarray:
        """复制数组并将夹爪维度乘以 factor；1-D 或 2-D（末维为动作维）。"""
        arr = np.asarray(value).copy()
        if arr.ndim == 1:
            for i in self.gripper_indices:
                arr[i] *= factor
            return arr
        if arr.ndim == 2:
            for i in self.gripper_indices:
                arr[:, i] *= factor
            return arr
        raise ValueError(f"expected 1-D or 2-D action array, got shape={arr.shape}")

    def to_server_state(self, state) -> np.ndarray:
        """组包发送观测 state：维度校验 + 有限性校验 + 单位换算。"""
        arr = np.asarray(state, dtype=np.float64).reshape(-1)
        if arr.shape != (self.state_dim,):
            raise ValueError(
                f"observation.state must be [{self.state_dim}], got shape={arr.shape}"
            )
        if not np.isfinite(arr).all():
            raise ValueError("observation.state must contain only finite values")
        return self.scale_grippers(arr, self.gripper_scale_to_server).astype(np.float32)

    def from_server_actions(self, actions) -> np.ndarray:
        """拆包服务端动作块：单位换算（夹爪乘 1/scale）。宽度校验见 payloads.parse_actions。"""
        if self.gripper_scale_to_server == 1.0:
            return np.asarray(actions, dtype=np.float64)
        return self.scale_grippers(actions, 1.0 / self.gripper_scale_to_server)

    def required_image_keys(self) -> frozenset[str]:
        """runner 观测 dict 中本契约需要的图像键（node_runner 以 observation.images.<alias> 提供）。"""
        return frozenset(f"observation.images.{alias}" for alias in self.camera_aliases)


DUAL_ARM = PolicyContract(
    name="dual_arm",
    action_dim=ACTION_DIM,
    joint_names=GLOBAL_JOINT_ORDER,
    gripper_indices=GRIPPER_INDICES,
    camera_aliases=CAMERA_ALIASES,
    state_dim=ACTION_DIM,
    gripper_range_m=GRIPPER_RANGE_M,
    gripper_scale_to_server=GRIPPER_MM_PER_M,
    server_payload_style="images_dict",
)

LIBERO = PolicyContract(
    name="libero",
    action_dim=LIBERO_ACTION_DIM,
    joint_names=LIBERO_JOINT_NAMES,
    gripper_indices=frozenset({LIBERO_GRIPPER_INDEX}),
    camera_aliases=LIBERO_CAMERA_ALIASES,
    state_dim=LIBERO_STATE_DIM,
    gripper_range_m=None,  # robosuite 原生域 [-1, 1]，不做米制范围校验
    gripper_scale_to_server=1.0,  # 夹爪域两侧一致，原样透传（不做米↔毫米换算）
    server_payload_style="flat_observation_keys",
)

_CONTRACTS: dict[str, PolicyContract] = {
    DUAL_ARM.name: DUAL_ARM,
    LIBERO.name: LIBERO,
}


def get_contract(name: str) -> PolicyContract:
    """按 profile 名取契约；未知名称直接报错（防拼写错误的静默掉口径）。"""
    try:
        return _CONTRACTS[name]
    except KeyError:
        raise ValueError(
            f"unknown policy contract: {name!r}; known contracts: {sorted(_CONTRACTS)}"
        ) from None


# —— 向后兼容别名：模块级函数一律委托 DUAL_ARM（既有调用点行为不变）——


def validate_action(action) -> np.ndarray:
    """校验并返回 1-D 动作数组：维度必须为 ACTION_DIM 且数值有限。（双臂口径）"""
    return DUAL_ARM.validate_action(action)


def gripper_range_violations(action) -> list[tuple[int, str, float]]:
    """返回超出夹爪量程的 (索引, 关节名, 值) 列表；空列表表示合规。（双臂口径）"""
    return DUAL_ARM.gripper_range_violations(action)


def _with_grippers_scaled(value, factor: float) -> np.ndarray:
    """复制数组并将夹爪维度（索引 %7==6）乘以 factor；1-D 或 2-D（末维为动作维）。"""
    return DUAL_ARM.scale_grippers(value, factor)


def grippers_to_server_units(value) -> np.ndarray:
    """夹爪 米→毫米（组包发送观测 state 时使用）。"""
    return _with_grippers_scaled(value, GRIPPER_MM_PER_M)


def grippers_from_server_units(value) -> np.ndarray:
    """夹爪 毫米→米（拆包服务端动作块时使用）。"""
    return _with_grippers_scaled(value, 1.0 / GRIPPER_MM_PER_M)
