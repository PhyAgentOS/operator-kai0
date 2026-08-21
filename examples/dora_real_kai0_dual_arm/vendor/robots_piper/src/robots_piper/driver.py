"""Piper 机械臂硬件驱动，实现 robot_core.RobotDriver 协议。"""

from __future__ import annotations

import math
import time
from typing import Literal

from forge_robots_msgs import ActuatorValue, RobotAction, RobotState
from piper_sdk import C_PiperInterface_V2
from robot_core import BaseRobotDriver

from .config import PIPER_ACTUATOR_ORDER
from forge_common import get_logger

logger = get_logger(__name__)

# 关节/夹爪限位，用于 set_actuators 前裁剪
_ACTUATOR_LIMITS: dict[str, tuple[float, float]] = {
    "joint1": (-2.618, 2.618),
    "joint2": (0, 3.140),
    "joint3": (-2.967, 0),
    "joint4": (-1.745, 1.745),
    "joint5": (-1.320, 1.320),
    "joint6": (-2.094, 2.094),
    "gripper": (0.0, 105.0),
}


def _get_safe_action(action: RobotAction) -> RobotAction:
    """将 action 中各执行器值裁剪到限位内。"""
    safe_actuators = {}
    for name, act_val in action.actuators.items():
        limits = _ACTUATOR_LIMITS.get(name)
        if limits is None:
            continue
        lo, hi = limits
        clipped = max(min(act_val.value, hi), lo)
        safe_actuators[name] = ActuatorValue(
            value=clipped,
            mode=act_val.mode,
            unit=act_val.unit,
        )
    return RobotAction(actuators=safe_actuators)


class PiperDriver(BaseRobotDriver):
    """
    Piper 机械臂硬件驱动。

    通过 CAN 与 piper-sdk 通信。主从：role="slave"（从站）连接后使能；
    role="master"（主站）仅连接不使能。单位：关节弧度、夹爪毫米。
    """

    @property
    def actuator_order(self) -> list[str]:
        return list(PIPER_ACTUATOR_ORDER)

    ANGLE_SCALE = 1000.0
    GRIPPER_SCALE = 1000.0

    _LINKAGE_MASTER = 0xFA
    _LINKAGE_SLAVE = 0xFC

    def __init__(
        self,
        port: str = "can0",
        role: Literal["master", "slave"] = "slave",
        is_follower: bool | None = None,
        auto_connect: bool = True,
        # kai0 新增（参考 real_deploy 定型口径）：False 时连接后不下发一次性全 0
        # 安全位，由上层在连接后自行插值回零（nodes/startup_zero.py）。默认 True
        # 保持上游历史行为。
        auto_set_safety_position: bool = True,
    ):
        if role == "master":
            self.is_follower = False
        elif is_follower is not None:
            self.is_follower = is_follower
        else:
            self.is_follower = True
        self.role: Literal["master", "slave"] = (
            "slave" if self.is_follower else "master"
        )
        self.port = port
        self.auto_set_safety_position = auto_set_safety_position
        self.bus: C_PiperInterface_V2 | None = None
        # Leader 模式下 get_state 不是总有数据，全 0 视为无效，用缓存
        self._state_cache: RobotState | None = None

        if auto_connect:
            self.connect()

    def connect(self) -> None:
        if self.bus is not None:
            logger.warning("Robot is already connected.")
            return

        self.bus = C_PiperInterface_V2(self.port)
        logger.info("Connecting to Piper robot at port %s...", self.port)
        self.bus.ConnectPort()

        linkage = self._LINKAGE_SLAVE if self.is_follower else self._LINKAGE_MASTER
        self.bus.MasterSlaveConfig(linkage, 0, 0, 0)

        if self.is_follower:
            self._connect_follower_mode()
            logger.info("Successfully connected and enabled the robot (slave).")
        else:
            self._connect_master_mode()
            logger.info("Successfully connected (master, not enabled).")

    # EnablePiper 等待超时时间（秒），超时后尝试 ResetPiper 再重试一次
    _ENABLE_PIPER_TIMEOUT_SEC = 2.0

    def _connect_follower_mode(self) -> None:
        logger.info("Enabling the robot in follower mode...")

        def wait_enable() -> bool:
            """等待 EnablePiper 成功，超时返回 False。"""
            deadline = time.monotonic() + self._ENABLE_PIPER_TIMEOUT_SEC
            while not self.bus.EnablePiper():
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.01)
            self.bus.MotionCtrl_2(0x01, 0x01, 30)
            return True

        if not wait_enable():
            logger.warning(
                "EnablePiper timed out after %.1fs, calling ResetPiper and retrying once...",
                self._ENABLE_PIPER_TIMEOUT_SEC,
            )
            self.bus.ResetPiper()
            time.sleep(0.5)
            if not wait_enable():
                raise RuntimeError(
                    "EnablePiper failed after ResetPiper retry. Check robot state and connection."
                )

        # kai0 修改（参考 real_deploy 定型口径）：auto_set_safety_position=False 时
        # 跳过连接即下发的一次性安全位，由上层在连接后插值回零（避免先冲零再插值）。
        if self.auto_set_safety_position:
            self.move_to_home_position()
            time.sleep(0.1)
            self.move_to_home_position()

    def move_to_home_position(self) -> None:
        """将关节与夹爪移动到 Home 位（全 0）。"""
        if self.bus is None:
            raise RuntimeError("Robot is not connected. Call connect() first.")
        logger.info("Moving robot to home position (all 0)...")
        self.bus.JointCtrl(0, 0, 0, 0, 0, 0)
        self.bus.GripperCtrl(0, 1000, 0x01, 0)

    def move_to_rest_position(self) -> None:
        """将关节与夹爪移动到休息位（防摔，j5=0.3 rad）。"""
        if self.bus is None:
            raise RuntimeError("Robot is not connected. Call connect() first.")
        logger.info("Moving robot to rest position (j5=0.3 rad)...")
        self.bus.JointCtrl(0, 0, 0, 0, int(math.degrees(0.3) * self.ANGLE_SCALE), 0)
        self.bus.GripperCtrl(0, 1000, 0x01, 0)

    def _connect_master_mode(self) -> None:
        logger.info("Robot configured in master mode.")

    def disconnect(self) -> None:
        if self.bus is None:
            logger.warning("Robot is not connected.")
            return

        if self.is_follower:
            logger.info("Safety sequence: Moving to rest position before disable...")
            try:
                self.move_to_rest_position()
            except Exception as e:
                logger.error("Failed to move to rest position: %s", e)

        logger.info("Disabling and disconnecting from the robot...")
        self.bus.DisconnectPort()
        self.bus = None
        logger.info("Disconnected from the robot.")

    def reset(self) -> None:
        if self.bus is None:
            raise RuntimeError("Robot is not connected. Call connect() first.")
        logger.info("Starting homing sequence...")
        self.bus.ResetPiper()
        logger.info("Homing command sent.")

    def get_state(self) -> RobotState:
        """从硬件读取状态，转为 RobotState（弧度/毫米）。Leader 模式下全 0 视为无效，返回缓存。"""
        if self.bus is None:
            raise RuntimeError("Robot is not connected. Call connect() first.")

        if self.is_follower:
            joints = self.bus.GetArmJointMsgs().joint_state
            gripper = self.bus.GetArmGripperMsgs().gripper_state
        else:
            joints = self.bus.GetArmJointCtrl().joint_ctrl
            gripper = self.bus.GetArmGripperCtrl().gripper_ctrl
            # Leader 下全 0 表示无有效数据，用缓存
            if (
                joints.joint_1 == 0
                and joints.joint_2 == 0
                and joints.joint_3 == 0
                and joints.joint_4 == 0
                and joints.joint_5 == 0
                and joints.joint_6 == 0
                and gripper.grippers_angle == 0
            ):
                if self._state_cache is not None:
                    return self._state_cache
                # 无缓存时仍返回全 0 的 state，避免调用方报错

        actuator_values = {
            "joint1": ActuatorValue(
                value=math.radians(joints.joint_1 / self.ANGLE_SCALE),
                mode="position",
                unit="radians",
            ),
            "joint2": ActuatorValue(
                value=math.radians(joints.joint_2 / self.ANGLE_SCALE),
                mode="position",
                unit="radians",
            ),
            "joint3": ActuatorValue(
                value=math.radians(joints.joint_3 / self.ANGLE_SCALE),
                mode="position",
                unit="radians",
            ),
            "joint4": ActuatorValue(
                value=math.radians(joints.joint_4 / self.ANGLE_SCALE),
                mode="position",
                unit="radians",
            ),
            "joint5": ActuatorValue(
                value=math.radians(joints.joint_5 / self.ANGLE_SCALE),
                mode="position",
                unit="radians",
            ),
            "joint6": ActuatorValue(
                value=math.radians(joints.joint_6 / self.ANGLE_SCALE),
                mode="position",
                unit="radians",
            ),
            "gripper": ActuatorValue(
                value=gripper.grippers_angle / self.GRIPPER_SCALE,
                mode="position",
                unit="millimeters",
            ),
        }
        state = RobotState(actuators=actuator_values)
        if not self.is_follower:
            self._state_cache = state
        return state

    def set_actuators(self, action: RobotAction) -> None:
        """下发 RobotAction（弧度/毫米），内部会裁剪到限位。"""
        if self.bus is None:
            raise RuntimeError("Robot is not connected. Call connect() first.")

        safe_action = _get_safe_action(action)
        action_dict = {name: act.value for name, act in safe_action.actuators.items()}

        j1 = int(math.degrees(action_dict.get("joint1", 0.0)) * self.ANGLE_SCALE)
        j2 = int(math.degrees(action_dict.get("joint2", 0.0)) * self.ANGLE_SCALE)
        j3 = int(math.degrees(action_dict.get("joint3", 0.0)) * self.ANGLE_SCALE)
        j4 = int(math.degrees(action_dict.get("joint4", 0.0)) * self.ANGLE_SCALE)
        j5 = int(math.degrees(action_dict.get("joint5", 0.0)) * self.ANGLE_SCALE)
        j6 = int(math.degrees(action_dict.get("joint6", 0.0)) * self.ANGLE_SCALE)
        gripper = int(action_dict.get("gripper", 0.0) * self.GRIPPER_SCALE)

        self.bus.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.bus.JointCtrl(j1, j2, j3, j4, j5, j6)
        self.bus.GripperCtrl(abs(gripper), 1000, 0x01, 0)
