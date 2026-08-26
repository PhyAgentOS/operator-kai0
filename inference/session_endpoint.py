"""kai0 推理服务会话端点（forge SessionToolEndpoint SPI）。

会话语义 = 常驻 openpi 推理服务（serve_policy）的生命周期：

- `start()` attach-or-launch：先探测 /healthz，200 则附接（owned=False，不启进程）；
  否则经 `scripts/start_server.sh --ckpt <weights> --policy-config pi05_libero`
  拉起进程（start_new_session，owned=True），日志落会话工作区。
- 三段就绪门（顺序）：/healthz 200（默认 600s）→ `kai0 infer-once` 成功且
  action_dim==7 → norm_stats.json sha256 匹配配置。
  任一失败：owned 则先 teardown 进程组，再抛 ToolEndpointError("READINESS_FAIL",
  retryable=True)；重复 execution key 幂等返回当前状态，不重启。
- `stop()` owned 时 SIGTERM 进程组（30s 宽限）→ SIGKILL（GPU 释放 = 进程组消亡）；
  attached 不杀进程。终态 stopped + ToolResult(status="stopped")。

运行期路径解析：权重/工作区相对路径按 launch cwd（skill 工作区）解析；
start 脚本优先取 _MEIPASS（PyInstaller 单文件内嵌副本），回退仓库根/launch cwd。
仓库内不出现绝对机器路径。
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge_tool import (
    TOOL_ENDPOINT_PROTOCOL,
    ToolAccepted,
    ToolContext,
    ToolControlResponse,
    ToolEndpointDescriptor,
    ToolEndpointError,
    ToolError,
    ToolEvent,
    ToolEventEmitter,
    ToolExecutionKey,
    ToolExecutionStatus,
    ToolOperationDescriptor,
    ToolRequest,
    ToolResult,
    ToolResultResponse,
)

from kai0_inference.common.paths import resolve_user_path

ENDPOINT_ID = "policy.runner"
OPERATION = "serve"
DESCRIPTOR = ToolEndpointDescriptor(
    protocol_version=TOOL_ENDPOINT_PROTOCOL,
    endpoint_id=ENDPOINT_ID,
    operations=(
        ToolOperationDescriptor(
            name=OPERATION,
            semantics="session",
            stoppable=True,
            status_supported=True,
            max_concurrency=1,
        ),
    ),
)

# 就绪门第 3 段锚点：pi05_libero 官方 norm_stats.json（LIBERO 布局
# <weights>/assets/<repo_id>/norm_stats.json）的 sha256。
DEFAULT_NORM_STATS_SHA256 = (
    "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84"
)
DEFAULT_POLICY_CONFIG = "pi05_libero"
DEFAULT_READINESS_TIMEOUT_S = 600.0
DEFAULT_STOP_GRACE_S = 30.0
DEFAULT_POLL_INTERVAL_S = 1.0
DEFAULT_HEALTH_TIMEOUT_S = 5.0
_START_ARGUMENT_KEYS = frozenset(("server_url", "weights_dir"))

logger = logging.getLogger(__name__)

PopenFactory = Callable[[Sequence[str], Mapping[str, str], Path], "subprocess.Popen[bytes]"]
HealthProbe = Callable[[str, float], bool]
InferOnceSmoke = Callable[[], Mapping[str, Any]]


def _endpoint_error(
    code: str, message: str, *, retryable: bool = False, details: Mapping[str, Any] | None = None
) -> ToolEndpointError:
    return ToolEndpointError(
        ToolError(code=code, message=message, retryable=retryable, details=dict(details or {}))
    )


def _group_alive(pgid: int) -> bool:
    """进程组是否存活（0 信号探测）；组不存在 = 已消亡。"""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        if error.errno == errno.ESRCH:
            return False
        raise
    return True


def _signal_group(pgid: int, sig: int) -> None:
    """向整个进程组发信号；组已消亡视为成功（幂等 teardown）。"""
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        logger.warning("process group %s no longer signalable (sig=%s)", pgid, sig)
    except OSError as error:
        if error.errno != errno.ESRCH:
            raise


def _http_url_for(server_url: str) -> str:
    """ws(s):// 推理服务地址 → 同端口 HTTP 地址（/healthz 探测用）。"""
    if server_url.startswith("wss://"):
        return "https://" + server_url[len("wss://") :]
    if server_url.startswith("ws://"):
        return "http://" + server_url[len("ws://") :]
    return server_url


def _port_for(server_url: str) -> str | None:
    for scheme in ("wss://", "ws://", "https://", "http://"):
        if server_url.startswith(scheme):
            rest = server_url[len(scheme) :].split("/", 1)[0]
            return rest.rsplit(":", 1)[1] if ":" in rest else None
    return None


def default_health_probe(server_url: str, timeout_s: float) -> bool:
    """GET <server>/healthz → 200 视为就绪（urllib：不受 ALL_PROXY 影响）。"""
    url = _http_url_for(server_url).rstrip("/") + "/healthz"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:  # noqa: S310
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def default_popen_factory(
    argv: Sequence[str], env: Mapping[str, str], log_path: Path
) -> "subprocess.Popen[bytes]":
    """拉起 start_server.sh：新会话（进程组 = pid），日志重定向到会话工作区。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")  # noqa: SIM115 - 进程生命周期长于本函数
    return subprocess.Popen(
        list(argv),
        env=dict(env),
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def _default_start_script() -> Path | None:
    """start_server.sh 解析顺序：_MEIPASS（单文件内嵌）→ 仓库根 → launch cwd。"""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        bundled = Path(meipass) / "scripts" / "start_server.sh"
        if bundled.is_file():
            return bundled
    package_root = Path(__file__).resolve().parents[2]
    repo_script = package_root / "scripts" / "start_server.sh"
    if repo_script.is_file():
        return repo_script
    return resolve_user_path("scripts/start_server.sh")


@dataclass(slots=True)
class _Execution:
    key: ToolExecutionKey
    arguments_json: str
    emitter: ToolEventEmitter
    server_url: str
    pid: int | None
    owned: bool
    process: "subprocess.Popen[bytes] | None"
    readiness: dict[str, Any] | None = None
    phase: str = "accepted"
    result: ToolResult | None = None


@dataclass(frozen=True, slots=True)
class ShutdownOutcome:
    active_key: ToolExecutionKey | None
    owned_process_terminated: bool


class Kai0ServeSessionEndpoint:
    """policy.runner/serve 会话端点：openpi 服务 attach-or-launch + 就绪门 + stop。"""

    def __init__(
        self,
        *,
        server_url: str,
        weights_dir: str | None,
        norm_stats_sha256: str = DEFAULT_NORM_STATS_SHA256,
        readiness_timeout_s: float = DEFAULT_READINESS_TIMEOUT_S,
        policy_config: str = DEFAULT_POLICY_CONFIG,
        start_script: str | None = None,
        inference_config_path: str | None = None,
        server_venv: str | None = None,
        workspace_dir: str | None = None,
        stop_grace_s: float = DEFAULT_STOP_GRACE_S,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        health_timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
        health_probe: HealthProbe | None = None,
        infer_once: InferOnceSmoke | None = None,
        popen_factory: PopenFactory | None = None,
    ) -> None:
        if not server_url:
            raise ValueError("server_url must not be empty")
        if readiness_timeout_s <= 0 or stop_grace_s <= 0 or poll_interval_s <= 0:
            raise ValueError("timeouts and intervals must be positive")
        self._server_url = server_url
        self._weights_dir = weights_dir
        self._norm_stats_sha256 = norm_stats_sha256.lower()
        self._readiness_timeout_s = readiness_timeout_s
        self._policy_config = policy_config
        self._start_script_arg = start_script
        self._inference_config_path = inference_config_path
        self._server_venv = server_venv
        self._workspace_dir = workspace_dir
        self._stop_grace_s = stop_grace_s
        self._poll_interval_s = poll_interval_s
        self._health_timeout_s = health_timeout_s
        self._clock = clock
        self._health_probe = health_probe or default_health_probe
        self._infer_once = infer_once
        self._popen_factory = popen_factory or default_popen_factory
        self._executions: dict[ToolExecutionKey, _Execution] = {}
        self._active: ToolExecutionKey | None = None
        self._accepting = True

    # —— SessionToolEndpoint SPI ——

    async def start(
        self,
        request: ToolRequest,
        context: ToolContext,
        events: ToolEventEmitter,
    ) -> ToolAccepted:
        if context.operation != OPERATION:
            raise _endpoint_error(
                "FORGE_PROTOCOL_UNKNOWN_OPERATION",
                f"unknown operation {context.operation!r}",
            )
        if not self._accepting:
            raise _endpoint_error(
                "FORGE_ENDPOINT_STOPPING",
                "policy.runner session endpoint is shutting down and no longer accepts starts",
                retryable=True,
            )
        # session 无 deadline（gateway 豁免）；显式传入时仍按时钟校验，防误用。
        if context.deadline_ms is not None and int(self._clock() * 1000) >= context.deadline_ms:
            raise _endpoint_error(
                "FORGE_DEADLINE_EXCEEDED",
                "the Tool invocation deadline has expired",
            )

        key = context.execution_key
        arguments_json = json.dumps(request.arguments, sort_keys=True, separators=(",", ":"))
        existing = self._executions.get(key)
        if existing is not None:
            if existing.arguments_json != arguments_json:
                raise _endpoint_error(
                    "FORGE_EXECUTION_CONFLICT",
                    "execution key was already used with different session arguments",
                )
            # 幂等：返回当前状态，绝不重启服务。
            return ToolAccepted(details=self._state_details(existing, idempotent=True))
        if self._active is not None:
            raise _endpoint_error(
                "FORGE_ENDPOINT_BUSY",
                f"policy.runner session '{self._active.invocation_id}' is already active",
                retryable=True,
            )

        overrides = _parse_start_arguments(request.arguments)
        server_url = str(overrides.get("server_url") or self._server_url)
        weights_dir = str(overrides.get("weights_dir") or self._weights_dir or "")

        owned, pid, process, readiness = await self._run_readiness_gate(
            key, server_url, weights_dir
        )
        execution = _Execution(
            key=key,
            arguments_json=arguments_json,
            emitter=events,
            server_url=server_url,
            pid=pid,
            owned=owned,
            process=process,
            readiness=readiness,
            phase="running",
        )
        self._executions[key] = execution
        self._active = key
        return ToolAccepted(details=self._state_details(execution, idempotent=False))

    async def stop(
        self, key: ToolExecutionKey, reason: str | None = None
    ) -> ToolControlResponse:
        execution = self._executions.get(key)
        if execution is None:
            return ToolControlResponse(
                command="stop",
                status="rejected",
                error=ToolError(
                    code="FORGE_EXECUTION_NOT_FOUND",
                    message="execution key is unknown",
                ),
            )
        if execution.result is not None:
            return ToolControlResponse(command="stop", status="terminal")
        if execution.phase != "stopping":
            execution.phase = "stopping"
            if execution.owned and execution.pid is not None:
                await self._terminate_owned_process(execution.pid, execution.process)
            execution.phase = "stopped"
            execution.result = ToolResult(
                status="stopped",
                outputs={
                    "server_url": execution.server_url,
                    "pid": execution.pid,
                    "owned": execution.owned,
                    "reason": reason,
                },
            )
            if self._active == execution.key:
                self._active = None
            try:
                await execution.emitter.emit(
                    ToolEvent(
                        type="stopped",
                        data={
                            "server_url": execution.server_url,
                            "pid": execution.pid,
                            "owned": execution.owned,
                        },
                    )
                )
            except Exception:
                logger.exception("failed to emit policy.runner stopped event")
        return ToolControlResponse(
            command="stop",
            status="accepted",
            details={"server_url": execution.server_url, "reason": reason},
        )

    async def status(self, key: ToolExecutionKey) -> ToolExecutionStatus:
        execution = self._executions.get(key)
        if execution is None:
            return ToolExecutionStatus(
                phase="unknown",
                error=ToolError(
                    code="FORGE_EXECUTION_NOT_FOUND",
                    message="execution key is unknown",
                ),
            )
        return ToolExecutionStatus(
            phase=execution.phase,  # type: ignore[arg-type]
            details=self._state_details(execution),
        )

    async def result(self, key: ToolExecutionKey) -> ToolResultResponse:
        execution = self._executions.get(key)
        if execution is None:
            return ToolResultResponse(status="not_found")
        if execution.result is None:
            return ToolResultResponse(status="pending")
        return ToolResultResponse(status="available", result=execution.result)

    async def shutdown(self) -> ShutdownOutcome:
        """节点退出兜底：停止准入；owned 进程随之消亡，避免 GPU 泄漏。"""
        self._accepting = False
        active = self._executions.get(self._active) if self._active is not None else None
        if active is None or active.result is not None:
            return ShutdownOutcome(self._active, False)
        terminated = False
        if active.owned and active.pid is not None:
            await self._terminate_owned_process(active.pid, active.process)
            terminated = True
        return ShutdownOutcome(active.key, terminated)

    # —— 就绪门 ——

    async def _run_readiness_gate(
        self, key: ToolExecutionKey, server_url: str, weights_dir: str
    ) -> tuple[bool, int | None, "subprocess.Popen[bytes] | None", dict[str, Any]]:
        """attach-or-launch + 三段就绪门；失败（owned 先 teardown）抛 READINESS_FAIL。"""
        owned = False
        pid: int | None = None
        process: "subprocess.Popen[bytes] | None" = None
        readiness: dict[str, Any] = {}
        try:
            # 段 1a：attach 探测（单次快速 GET）。
            if not self._health_probe(server_url, self._health_timeout_s):
                pid, process = self._launch_server(key, server_url, weights_dir)
                owned = True
                # 段 1b：launch 后等待 /healthz（默认 600s）。
                if not await self._wait_healthz(server_url, self._readiness_timeout_s):
                    raise _GateFailure(
                        "healthz", f"policy server not healthy within {self._readiness_timeout_s}s"
                    )
            readiness["healthz"] = True
            # 段 2：kai0 infer-once 冒烟（action_dim==7）。
            smoke = await self._run_infer_once_smoke(server_url)
            action_dim = smoke.get("action_dim")
            if action_dim != 7:
                raise _GateFailure("infer_once", f"smoke action_dim={action_dim!r}, expected 7")
            readiness["infer_once"] = dict(smoke)
            # 段 3：norm_stats.json sha256 锚点。
            stats_path = _norm_stats_path(Path(weights_dir))
            if stats_path is None:
                raise _GateFailure("norm_stats", "norm_stats.json not found under <weights>/assets/")
            digest = hashlib.sha256(stats_path.read_bytes()).hexdigest()
            if digest != self._norm_stats_sha256:
                raise _GateFailure(
                    "norm_stats",
                    f"sha256 mismatch: got {digest}, expected {self._norm_stats_sha256}",
                )
            readiness["norm_stats_sha256"] = digest
        except _GateFailure as failure:
            if owned and pid is not None:
                logger.warning("readiness gate failed at %s; tearing down owned process", failure.stage)
                await self._terminate_owned_process(pid, process)
            raise _endpoint_error(
                "READINESS_FAIL",
                f"policy server readiness gate failed at stage {failure.stage}: {failure.message}",
                retryable=True,
                details={
                    "stage": failure.stage,
                    "server_url": server_url,
                    "owned": owned,
                    "pid": pid,
                },
            ) from None
        return owned, pid, process, readiness

    async def _wait_healthz(self, server_url: str, timeout_s: float) -> bool:
        deadline = self._clock() + timeout_s
        while self._clock() < deadline:
            if self._health_probe(server_url, self._health_timeout_s):
                return True
            await asyncio.sleep(self._poll_interval_s)
        return self._health_probe(server_url, self._health_timeout_s)

    async def _run_infer_once_smoke(self, server_url: str) -> Mapping[str, Any]:
        if self._infer_once is not None:
            return self._infer_once()
        # 服务端地址以会话生效值为准（start args 覆盖 server_url 时冒烟同目标）。
        from kai0_inference.inference.config import load_config
        from kai0_inference.inference.run_once import run_infer_once_smoke

        config_path = self._resolve_inference_config()
        if config_path is None:
            raise _GateFailure(
                "infer_once",
                "session_endpoint readiness requires a file-backed inference config "
                "(KAI0_INFERENCE_CONFIG / --config / bundled default)",
            )

        def _smoke() -> Mapping[str, Any]:
            config = load_config(config_path=config_path)
            policy = dict(config.policy)
            policy["server_url"] = server_url
            config_dict = {
                "joints": [{"name": j.name, "mode": j.mode, "unit": j.unit} for j in config.joints],
                "state_joints": config.state_joints,
                "policy": policy,
                "image_inputs": config.image_input_id_to_alias,
            }
            smoke_config = load_config(dataflow_config={"config": config_dict})
            return run_infer_once_smoke(smoke_config)

        try:
            return await asyncio.to_thread(_smoke)
        except Exception as error:
            raise _GateFailure("infer_once", f"kai0 infer-once smoke failed: {error}") from error

    def _resolve_inference_config(self) -> str | None:
        if self._inference_config_path:
            return self._inference_config_path
        env_path = os.environ.get("KAI0_INFERENCE_CONFIG")
        if env_path:
            return env_path
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            bundled = Path(meipass) / "config" / "inference" / "kai0_libero.yaml"
            if bundled.is_file():
                return str(bundled)
        return None

    # —— 进程生命周期 ——

    def _launch_server(
        self, key: ToolExecutionKey, server_url: str, weights_dir: str
    ) -> tuple[int, "subprocess.Popen[bytes]"]:
        weights = resolve_user_path(weights_dir)
        if not (weights / "params").is_dir():
            raise _GateFailure(
                "ckpt", f"weights dir has no params/: {weights}"
            )
        script = self._resolve_start_script()
        if script is None or not script.is_file():
            raise _GateFailure(
                "start_script",
                "scripts/start_server.sh not found (repo root / bundle / config session.start_script)",
            )
        session_workspace = self._session_workspace() / f"serve-{key.invocation_id}-{key.attempt_id}"
        env = dict(os.environ)
        port = _port_for(server_url)
        if port:
            env["PORT"] = port
        if self._server_venv:
            env["SERVER_VENV"] = self._server_venv
        cache_root = session_workspace / "cache"
        env.setdefault("OPENPI_DATA_HOME", str(cache_root / "openpi"))
        env.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_root / "jax_compilation"))
        argv = [str(script), "--ckpt", str(weights), "--policy-config", self._policy_config]
        logger.info(
            "launching policy server: %s (pid will be process-group leader)", " ".join(argv)
        )
        process = self._popen_factory(argv, env, session_workspace / "server.log")
        return process.pid, process

    def _resolve_start_script(self) -> Path | None:
        if self._start_script_arg:
            candidate = resolve_user_path(self._start_script_arg)
            return candidate if candidate.is_file() else None
        return _default_start_script()

    def _session_workspace(self) -> Path:
        if self._workspace_dir:
            root = resolve_user_path(self._workspace_dir)
        else:
            root = resolve_user_path("session_workspace")
        root.mkdir(parents=True, exist_ok=True)
        return root

    async def _terminate_owned_process(
        self, pid: int, process: "subprocess.Popen[bytes] | None"
    ) -> None:
        """owned 会话 teardown：SIGTERM 进程组 → 宽限 → SIGKILL（可选 nvidia-smi 确认）。"""
        _signal_group(pid, signal.SIGTERM)
        deadline = self._clock() + self._stop_grace_s
        while _group_alive(pid) and self._clock() < deadline:
            await asyncio.sleep(self._poll_interval_s)
        if _group_alive(pid):
            logger.warning("policy server group %s survived SIGTERM grace; sending SIGKILL", pid)
            _signal_group(pid, signal.SIGKILL)
        if process is not None:
            try:
                process.wait(timeout=self._stop_grace_s)
            except subprocess.TimeoutExpired:
                logger.warning("policy server process %s did not exit after group kill", pid)
        self._confirm_gpu_release(pid)

    def _confirm_gpu_release(self, pid: int) -> None:
        """可选确认：nvidia-smi 可用时检查已杀 pid 不在计算进程列表（仅日志，不拦截）。"""
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            return
        try:
            output = subprocess.run(
                [nvidia_smi, "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            logger.warning("nvidia-smi query failed; skipping GPU release confirm")
            return
        leftover = str(pid) in output.split()
        if leftover:
            logger.warning("pid %s still listed by nvidia-smi after group kill", pid)
        else:
            logger.info("nvidia-smi confirms pid %s released", pid)

    # —— 状态快照 ——

    def _state_details(self, execution: _Execution, idempotent: bool | None = None) -> dict[str, Any]:
        details: dict[str, Any] = {
            "server_url": execution.server_url,
            "pid": execution.pid,
            "owned": execution.owned,
            "phase": execution.phase,
        }
        if execution.readiness is not None:
            details["readiness"] = execution.readiness
        if idempotent is not None:
            details["idempotent"] = idempotent
        return details


class _GateFailure(Exception):
    """就绪门内部失败信号（带阶段名）；由 _run_readiness_gate 转为 READINESS_FAIL。"""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        self.message = message
        super().__init__(f"{stage}: {message}")


def _parse_start_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(arguments) - _START_ARGUMENT_KEYS)
    if unknown:
        raise _endpoint_error(
            "POLICY_RUNNER_INVALID_ARGUMENTS",
            f"unexpected session arguments: {unknown} "
            f"(allowed: {sorted(_START_ARGUMENT_KEYS)})",
        )
    overrides = {}
    for name in ("server_url", "weights_dir"):
        if name in arguments:
            value = arguments[name]
            if not isinstance(value, str) or not value.strip():
                raise _endpoint_error(
                    "POLICY_RUNNER_INVALID_ARGUMENTS",
                    f"{name} must be a non-empty string",
                )
            overrides[name] = value
    return overrides


def _norm_stats_path(weights_dir: Path) -> Path | None:
    """LIBERO 布局 <weights>/assets/<repo_id>/norm_stats.json 的首个匹配。"""
    assets = weights_dir / "assets"
    if not assets.is_dir():
        return None
    matches = sorted(assets.glob("*/norm_stats.json"))
    return matches[0] if matches else None


__all__ = [
    "DEFAULT_NORM_STATS_SHA256",
    "DEFAULT_POLICY_CONFIG",
    "DESCRIPTOR",
    "ENDPOINT_ID",
    "Kai0ServeSessionEndpoint",
    "OPERATION",
    "ShutdownOutcome",
    "_group_alive",
    "_signal_group",
    "default_health_probe",
    "default_popen_factory",
]
