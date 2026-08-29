"""Policy 节点配置加载与校验。

契约 profile（D2 修订版）：校验不再面向全局双臂常量，而是按 `policy.type`
选择契约（`kai0_libero` → contract.LIBERO，其余 → contract.DUAL_ARM）；
也可用顶层 `contract:` 键显式覆盖（拼写错误立即报错，防静默掉口径）。
"""

from __future__ import annotations

import os
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from kai0_inference.common.paths import resolve_user_path
from kai0_inference.inference import contract

# policy.type → 契约 profile（未显式指定 contract 键时的默认选择）
_CONTRACT_BY_POLICY_TYPE: dict[str, contract.PolicyContract] = {
    "kai0_libero": contract.LIBERO,
    "kai0_rtc": contract.DUAL_ARM,
    "kai0": contract.DUAL_ARM,
    "dummy": contract.DUAL_ARM,
}


@dataclass(frozen=True)
class JointConfig:
    name: str
    mode: str = "position"
    unit: str = "radians"
    type: str = "actuator"


@dataclass(frozen=True)
class SessionConfig:
    """session_endpoint 模式的会话段（kai0 推理服务 attach-or-launch 生命周期）。

    - server_url / weights_dir / norm_stats_sha256：就绪门必需；
    - weights_dir 相对路径按 launch cwd（skill 工作区）解析；start 参数可覆盖；
      省略时按 PAOS 环境推导默认路径（见 _default_weights_dir）；
    - server_venv 为部署形态必需（PyInstaller 单文件内无 .venv）：显式配置优先，
      省略时按 PAOS 环境推导默认路径（见 _default_server_venv），无 PAOS 环境
      且未显式配置时为 None（独立部署须显式提供）；
    - 其余键为可选运行期细节（start 脚本/工作区），仓库内无绝对路径。
    """

    server_url: str
    weights_dir: str
    norm_stats_sha256: str
    readiness_timeout_s: float = 600.0
    policy_config: str = "pi05_libero"
    start_script: str | None = None
    server_venv: str | None = None
    workspace_dir: str | None = None
    stop_grace_s: float = 30.0


_SESSION_KEYS = frozenset(SessionConfig.__dataclass_fields__)


@dataclass
class PolicyNodeConfig:
    """Policy 节点完整配置：joints、policy、image_inputs（input_id -> alias）。"""

    joints: list[JointConfig]
    policy: dict[str, Any]
    auto_start: bool = False
    image_input_id_to_alias: dict[str, str] = field(default_factory=dict)
    state_joints: list[str] = field(default_factory=list)
    contract: contract.PolicyContract = contract.DUAL_ARM
    mode: str = "dora_plain"  # dora_plain | session_endpoint（过渡开关）
    session: SessionConfig | None = None

    @property
    def joint_order(self) -> list[str]:
        """动作关节顺序（与 JointCommand / task_robot 一致）。"""
        return [j.name for j in self.joints]

    @property
    def state_joint_order(self) -> list[str]:
        """观测状态顺序；未单独配置时沿用动作关节顺序。"""
        return self.state_joints or self.joint_order

    @property
    def image_input_ids(self) -> set[str]:
        return set(self.image_input_id_to_alias.keys())

    @property
    def alias_for_cameras(self) -> list[str]:
        """相机别名顺序；由 YAML image_inputs 定义顺序决定。"""
        return list(self.image_input_id_to_alias.values())

    def image_inputs_for(
        self,
        required_image_keys: Collection[str] | None,
    ) -> dict[str, str]:
        """Return only runtime image inputs consumed by the loaded policy."""
        if required_image_keys is None:
            return dict(self.image_input_id_to_alias)
        required = set(required_image_keys)
        return {
            input_id: alias
            for input_id, alias in self.image_input_id_to_alias.items()
            if f"observation.images.{alias}" in required
        }

    def runtime_policy_config(self) -> dict[str, Any]:
        """生成策略运行时配置，补齐可由节点配置推断的字段。"""
        return {
            **self.policy,
            "auto_start": self.auto_start,
            "contract": self.contract.name,
            "camera_names": self.alias_for_cameras,
            "joint_count": len(self.joints),
            "state_joint_count": len(self.state_joint_order),
            "action_joint_count": len(self.joints),
            "state_joint_names": list(self.state_joint_order),
            "action_joint_names": list(self.joint_order),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyNodeConfig:
        joints_raw = data.get("joints")
        if not joints_raw:
            raise ValueError("joints 不能为空")

        policy = data.get("policy", {})
        if not policy or not policy.get("type"):
            raise ValueError("policy 必须包含 type（kai0 / kai0_libero / kai0_rtc / dummy）")

        # 契约 profile 选择：显式 contract 键优先，否则按 policy.type 默认映射
        profile = _select_contract(data.get("contract"), str(policy.get("type")))

        joints = _parse_action_joints(joints_raw)
        # 契约校验：关节名与顺序必须与所选契约的 joint_names 完全一致
        if [j.name for j in joints] != list(profile.joint_names):
            raise ValueError(
                f"joints 必须与契约 contract.{profile.name} 的 joint_names 顺序一致："
                f"{list(profile.joint_names)}"
            )

        state_joints_raw = data.get("state_joints")
        state_joints = (
            _parse_state_joints(state_joints_raw)
            if state_joints_raw is not None
            else [joint.name for joint in joints]
        )

        image_inputs = data.get("image_inputs")
        if not isinstance(image_inputs, dict) or not image_inputs:
            raise ValueError("image_inputs 不能为空，且必须为 dict[str, str]")

        image_input_id_to_alias = {
            str(input_id).strip(): str(alias).strip() for input_id, alias in image_inputs.items()
        }
        if any(
            not input_id or not alias
            for input_id, alias in image_input_id_to_alias.items()
        ):
            raise ValueError("image_inputs 的 input ID 和 alias 均不能为空")
        aliases = list(image_input_id_to_alias.values())
        if len(set(aliases)) != len(aliases):
            raise ValueError("image_inputs 的 alias 必须唯一，不能让多路输入覆盖同一相机")
        # 契约校验：alias 集合必须恰好是所选契约识别的相机 alias
        if set(aliases) != set(profile.camera_aliases):
            raise ValueError(
                f"image_inputs 的 alias 集合必须与契约 contract.{profile.name} 的 "
                f"camera_aliases 一致：{set(profile.camera_aliases)}"
            )

        mode = str(data.get("mode", "dora_plain")).strip()
        if mode not in ("dora_plain", "session_endpoint"):
            raise ValueError(f"mode 必须为 dora_plain 或 session_endpoint：{mode!r}")
        session = _parse_session(data.get("session"))
        if mode == "session_endpoint" and session is None:
            raise ValueError("mode=session_endpoint 需要 session 段（server_url/weights_dir 等）")

        return cls(
            joints=joints,
            policy=policy,
            state_joints=state_joints,
            auto_start=_as_bool(policy.get("auto_start", data.get("auto_start", False))),
            image_input_id_to_alias=image_input_id_to_alias,
            contract=profile,
            mode=mode,
            session=session,
        )

    @classmethod
    def from_yaml_path(cls, path: str | Path) -> PolicyNodeConfig:
        p = resolve_user_path(path)
        if not p.is_file():
            raise FileNotFoundError(f"配置文件不存在: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not data:
            raise ValueError(f"配置文件为空: {p}")
        return cls.from_dict(data)


def _parse_session(raw: Any) -> SessionConfig | None:
    """解析可选 session 段；未知键/缺键 fail fast（防静默掉口径）。"""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("session 必须为 dict")
    unknown = sorted(set(raw) - _SESSION_KEYS)
    if unknown:
        raise ValueError(f"session 存在未知键: {unknown}（可用: {sorted(_SESSION_KEYS)}）")
    missing = [
        name
        for name in ("server_url", "norm_stats_sha256")
        if not str(raw.get(name, "")).strip()
    ]
    if missing:
        raise ValueError(f"session 缺少必需键: {missing}")
    weights_dir = str(raw.get("weights_dir", "")).strip() or _default_weights_dir()
    if not weights_dir:
        raise ValueError(
            "session 缺少 weights_dir：既无显式配置，也无法从 PAOS 环境"
            "（PAOS_SKILL_NAME / PAOS_SKILL_VERSION）推导默认权重路径"
        )
    timeout = float(raw.get("readiness_timeout_s", 600.0))
    if timeout <= 0:
        raise ValueError("session.readiness_timeout_s 必须为正数")
    grace = float(raw.get("stop_grace_s", 30.0))
    if grace <= 0:
        raise ValueError("session.stop_grace_s 必须为正数")
    return SessionConfig(
        server_url=str(raw["server_url"]).strip(),
        weights_dir=weights_dir,
        norm_stats_sha256=str(raw["norm_stats_sha256"]).strip(),
        readiness_timeout_s=timeout,
        policy_config=str(raw.get("policy_config", "pi05_libero")).strip(),
        start_script=_optional_str(raw.get("start_script")),
        server_venv=_optional_str(raw.get("server_venv")) or _default_server_venv(),
        workspace_dir=_optional_str(raw.get("workspace_dir")),
        stop_grace_s=grace,
    )


def _default_weights_dir() -> str:
    """PAOS 环境下的确定性权重落盘路径。

    skill 的 start.sh 将大权重下载到 `~/.PhyAgentOS/cache/weights/<skill>/<version>/`
    （见 skill bundle 的 start.sh 与 SKILL.md 下载时长说明）；PhyAgentOS 流启动时
    注入 PAOS_SKILL_NAME / PAOS_SKILL_VERSION，节点据此推导同一点，配置无需硬编码。
    独立部署（无 PAOS 环境）时返回空串，由调用方报缺键错误。
    """
    name = os.environ.get("PAOS_SKILL_NAME", "").strip()
    version = os.environ.get("PAOS_SKILL_VERSION", "").strip()
    if not name or not version:
        return ""
    return str(Path.home() / ".PhyAgentOS" / "cache" / "weights" / name / version)


def _default_server_venv() -> str | None:
    """PAOS 环境下的服务端 venv 默认路径。

    PyInstaller 单文件内无 .venv，部署形态必须指向外部环境；约定由节点仓库的
    安装步骤（server/setup.sh，支持 SERVER_VENV 覆盖）安装到
    `~/.PhyAgentOS/cache/server/kai0_prompt_schedule/.venv`，节点在 PAOS 环境
    （PAOS_SKILL_NAME / PAOS_SKILL_VERSION 注入）下推导同一点，配置无需硬编码
    机器路径。无 PAOS 环境时返回 None（独立部署须显式配置 server_venv）。
    """
    name = os.environ.get("PAOS_SKILL_NAME", "").strip()
    version = os.environ.get("PAOS_SKILL_VERSION", "").strip()
    if not name or not version:
        return None
    return str(
        Path.home() / ".PhyAgentOS" / "cache" / "server" / "kai0_prompt_schedule" / ".venv"
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _select_contract(explicit: Any, policy_type: str) -> contract.PolicyContract:
    """契约 profile 选择：显式 `contract:` 键优先，否则按 policy.type 默认映射。"""
    if explicit is not None:
        return contract.get_contract(str(explicit).strip())
    known = _CONTRACT_BY_POLICY_TYPE.get(policy_type)
    if known is None:
        # registry 会给出更准确的报错；此处按双臂契约先行加载，避免配置层报错过早
        return contract.DUAL_ARM
    return known


def _parse_action_joints(items: Any) -> list[JointConfig]:
    if not isinstance(items, list) or not items:
        raise ValueError("joints 必须为非空列表")
    joints: list[JointConfig] = []
    for item in items:
        if isinstance(item, str):
            joints.append(JointConfig(name=item))
        elif isinstance(item, dict) and "name" in item:
            joints.append(
                JointConfig(
                    name=str(item["name"]),
                    mode=str(item.get("mode", "position")),
                    unit=str(item.get("unit", "radians")),
                    type=str(item.get("type", item.get("joint_type", "actuator"))),
                )
            )
        else:
            raise ValueError(f"joint 项须为字符串或包含 name 的字典: {item!r}")
    return joints


def _parse_state_joints(items: Any) -> list[str]:
    if not isinstance(items, list) or not items:
        raise ValueError("state_joints 必须为非空列表")
    names: list[str] = []
    for item in items:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict) and "name" in item:
            name = str(item["name"])
        else:
            raise ValueError(f"state_joints 项须为字符串或包含 name 的字典: {item!r}")
        if not name:
            raise ValueError("state_joints 中的关节名不能为空")
        names.append(name)
    return names


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def load_config(
    config_path: str | Path | None = None,
    dataflow_config: dict[str, Any] | None = None,
) -> PolicyNodeConfig:
    """
    加载配置（优先级同 pick_and_place）：
    1. dataflow 内联 config / config_path
    2. config_path 参数
    3. 环境变量 KAI0_INFERENCE_CONFIG
    """
    data: dict[str, Any] | None = None

    if dataflow_config:
        if "config" in dataflow_config:
            data = dataflow_config["config"]
        elif "config_path" in dataflow_config:
            return PolicyNodeConfig.from_yaml_path(dataflow_config["config_path"])

    path = config_path
    if path is None and data is None:
        path = os.environ.get("KAI0_INFERENCE_CONFIG")

    if path:
        return PolicyNodeConfig.from_yaml_path(path)

    if data:
        return PolicyNodeConfig.from_dict(data)

    raise ValueError(
        "未找到配置。请设置 KAI0_INFERENCE_CONFIG、"
        "或通过 --config 指定配置文件、或使用 dataflow config。"
    )
