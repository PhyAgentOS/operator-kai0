"""policy.runner 会话端点单测（GPU-free）：attach/launch/就绪门/teardown/幂等/stop/相位迁移。

不依赖真实 GPU 或 openpi 服务端：healthz 探测、infer-once 冒烟、Popen、
进程组信号全部注入/打桩；工具面集成测试走真实 forge_tool handler + binding
线路径（ToolMessage 编解码），验证 forge-tool-v0.2.0 与 forge-msgs 1.1.0 的配套。
"""

from __future__ import annotations

import asyncio
import hashlib
import signal
from pathlib import Path
from typing import Any

import pytest
from forge_msgs import ToolMessage
from forge_tool import (
    ToolAccepted,
    ToolContext,
    ToolEndpointError,
    ToolError,
    ToolExecutionKey,
    ToolRequest,
    control_response_from_payload,
    invoke_response_from_payload,
    make_control_request_envelope,
    make_invoke_request_envelope,
    make_result_request_envelope,
    make_status_request_envelope,
    result_response_from_payload,
    status_response_from_payload,
)
from forge_tool.dora import (
    DoraToolEndpointBinding,
    tool_envelope_to_message,
    tool_message_to_envelope,
)
from forge_tool.handler import ToolEndpointHandler

from kai0_inference.inference import session_endpoint as se
from kai0_inference.inference.session_endpoint import (
    DESCRIPTOR,
    ENDPOINT_ID,
    OPERATION,
    Kai0ServeSessionEndpoint,
)

INSTANCE_ID = "instance-1"


@pytest.fixture(autouse=True)
def _stub_nvidia_smi(monkeypatch):
    """GPU 释放确认仅日志；测试跳过 nvidia-smi（无 GPU 环境也不起子进程）。"""
    real_which = se.shutil.which

    def which(name: str):
        if name == "nvidia-smi":
            return None
        return real_which(name)

    monkeypatch.setattr(se.shutil, "which", which)


class FakeEmitter:
    """执行绑定事件汇：记录 emit 到的事件。"""

    def __init__(self) -> None:
        self.events: list = []

    async def emit(self, event) -> None:
        self.events.append(event)


class FakeProcess:
    """Popen 假实现：记录 wait 调用（不真等待）。"""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.waits = 0

    def wait(self, timeout: float | None = None) -> int:
        self.waits += 1
        return 0


def make_weights(tmp_path: Path, content: bytes = b'{"norm_stats": {}}') -> tuple[Path, str]:
    """LIBERO 布局假权重：<dir>/params/ + <dir>/assets/<repo>/norm_stats.json。"""
    weights = tmp_path / "weights" / "pi05_libero"
    (weights / "params").mkdir(parents=True)
    stats = weights / "assets" / "pi05_libero" / "norm_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_bytes(content)
    return weights, hashlib.sha256(content).hexdigest()


def make_context(invocation_id: str = "inv-1", attempt_id: str = "att-1") -> ToolContext:
    return ToolContext(
        execution_key=ToolExecutionKey(invocation_id, attempt_id),
        tool_id="kai0_libero.policy",
        implementation_id="impl-1",
        endpoint_id=ENDPOINT_ID,
        operation=OPERATION,
    )


def make_request(**arguments: Any) -> ToolRequest:
    return ToolRequest(arguments=arguments)


def make_endpoint(
    tmp_path: Path,
    *,
    health_probe=None,
    infer_once=None,
    popen_factory=None,
    **overrides: Any,
) -> Kai0ServeSessionEndpoint:
    weights, digest = make_weights(tmp_path)
    if infer_once is None:
        infer_once = lambda: {"action_dim": 7}  # noqa: E731 - 就绪门第 2 段默认通过
    kwargs: dict[str, Any] = {
        "server_url": "ws://127.0.0.1:8000",
        "weights_dir": str(weights),
        "norm_stats_sha256": digest,
        "readiness_timeout_s": 0.5,
        "stop_grace_s": 0.05,
        "poll_interval_s": 0.01,
        "health_timeout_s": 0.1,
        "health_probe": health_probe,
        "infer_once": infer_once,
        "popen_factory": popen_factory,
    }
    kwargs.update(overrides)
    return Kai0ServeSessionEndpoint(**kwargs)


def make_probe_sequence(values: list[bool]):
    calls: list[tuple[str, float]] = []
    index = 0

    def probe(server_url: str, timeout_s: float) -> bool:
        nonlocal index
        calls.append((server_url, timeout_s))
        result = values[min(index, len(values) - 1)]
        index += 1
        return result

    return probe, calls


class RecordingPopen:
    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.calls: list[dict[str, Any]] = []
        self.process = FakeProcess(pid)

    def __call__(self, argv, env, log_path: Path):
        self.calls.append({"argv": list(argv), "env": dict(env), "log_path": Path(log_path)})
        return self.process


# —— 描述符 ——


def test_descriptor_contract():
    assert isinstance(DESCRIPTOR, se.ToolEndpointDescriptor)
    assert DESCRIPTOR.endpoint_id == "policy.runner"
    assert len(DESCRIPTOR.operations) == 1
    op = DESCRIPTOR.operations[0]
    assert op.name == "serve"
    assert op.semantics == "session"
    assert op.stoppable is True
    assert op.status_supported is True
    assert op.max_concurrency == 1


# —— attach / launch ——


async def test_attach_ready_server_no_process(tmp_path):
    probe, calls = make_probe_sequence([True])
    popen = RecordingPopen()
    endpoint = make_endpoint(tmp_path, health_probe=probe, popen_factory=popen)
    accepted = await endpoint.start(make_request(), make_context(), FakeEmitter())
    assert isinstance(accepted, ToolAccepted)
    assert accepted.details["owned"] is False
    assert accepted.details["pid"] is None
    assert accepted.details["server_url"] == "ws://127.0.0.1:8000"
    assert accepted.details["readiness"]["healthz"] is True
    assert accepted.details["readiness"]["infer_once"]["action_dim"] == 7
    assert "norm_stats_sha256" in accepted.details["readiness"]
    assert accepted.details["idempotent"] is False
    assert popen.calls == []
    assert calls == [("ws://127.0.0.1:8000", 0.1)]


async def test_launch_when_unhealthy(tmp_path):
    probe, calls = make_probe_sequence([False, True])  # attach 探测失败 → launch → wait 成功
    popen = RecordingPopen(pid=4242)
    endpoint = make_endpoint(tmp_path, health_probe=probe, popen_factory=popen)
    accepted = await endpoint.start(make_request(), make_context("inv-2"), FakeEmitter())
    assert accepted.details["owned"] is True
    assert accepted.details["pid"] == 4242
    assert len(popen.calls) == 1
    argv = popen.calls[0]["argv"]
    assert argv[1:3] == ["--ckpt", str(Path(endpoint._weights_dir).resolve())]
    assert argv[3:5] == ["--policy-config", "pi05_libero"]
    assert str(argv[0]).endswith("scripts/start_server.sh")
    assert popen.calls[0]["log_path"].name == "server.log"
    assert calls[0] == ("ws://127.0.0.1:8000", 0.1)


async def test_launch_with_default_popen_real_start_new_session(tmp_path, monkeypatch):
    """默认 popen_factory 以 start_new_session=True 拉真 subprocess 壳（服务端不真跑）。"""
    probe, _ = make_probe_sequence([False, True])
    launched: list[dict] = []

    def fake_popen(argv, env=None, **kwargs):
        launched.append({"argv": argv, "kwargs": kwargs})
        return FakeProcess(9999)

    monkeypatch.setattr(se.subprocess, "Popen", fake_popen)
    endpoint = make_endpoint(tmp_path, health_probe=probe)
    accepted = await endpoint.start(make_request(), make_context("inv-p"), FakeEmitter())
    assert launched[0]["kwargs"]["start_new_session"] is True
    assert accepted.details["pid"] == 9999


async def test_launch_healthz_timeout_teardown(tmp_path, monkeypatch):
    probe, _ = make_probe_sequence([False])  # 一直不健康
    popen = RecordingPopen()
    signals: list[int] = []
    monkeypatch.setattr(se, "_signal_group", lambda pgid, sig: signals.append(sig))
    monkeypatch.setattr(se, "_group_alive", lambda pgid: False)
    endpoint = make_endpoint(
        tmp_path, health_probe=probe, popen_factory=popen, readiness_timeout_s=0.05
    )
    with pytest.raises(ToolEndpointError) as caught:
        await endpoint.start(make_request(), make_context("inv-t"), FakeEmitter())
    assert caught.value.error.code == "READINESS_FAIL"
    assert caught.value.error.retryable is True
    assert caught.value.error.details["stage"] == "healthz"
    assert signals == [signal.SIGTERM]  # 组已消亡，无需 SIGKILL
    assert popen.process.waits >= 1


# —— 就绪门失败 / teardown ——


async def test_readiness_failure_infer_once_teardown(tmp_path, monkeypatch):
    probe, _ = make_probe_sequence([False, True])
    popen = RecordingPopen()
    signals: list[int] = []
    monkeypatch.setattr(se, "_signal_group", lambda pgid, sig: signals.append(sig))
    monkeypatch.setattr(se, "_group_alive", lambda pgid: False)
    endpoint = make_endpoint(
        tmp_path,
        health_probe=probe,
        popen_factory=popen,
        infer_once=lambda: {"action_dim": 8},
    )
    with pytest.raises(ToolEndpointError) as caught:
        await endpoint.start(make_request(), make_context(), FakeEmitter())
    assert caught.value.error.code == "READINESS_FAIL"
    assert caught.value.error.details["stage"] == "infer_once"
    assert signals == [signal.SIGTERM]


async def test_readiness_failure_norm_stats_sigterm_then_sigkill(tmp_path, monkeypatch):
    probe, _ = make_probe_sequence([False, True])
    popen = RecordingPopen()
    signals: list[int] = []
    monkeypatch.setattr(se, "_signal_group", lambda pgid, sig: signals.append(sig))
    monkeypatch.setattr(se, "_group_alive", lambda pgid: True)  # 宽限后仍存活 → SIGKILL
    endpoint = make_endpoint(
        tmp_path,
        health_probe=probe,
        popen_factory=popen,
        norm_stats_sha256="0" * 64,  # 必然不匹配
        stop_grace_s=0.02,
    )
    with pytest.raises(ToolEndpointError) as caught:
        await endpoint.start(make_request(), make_context(), FakeEmitter())
    assert caught.value.error.code == "READINESS_FAIL"
    assert caught.value.error.details["stage"] == "norm_stats"
    assert signals == [signal.SIGTERM, signal.SIGKILL]


async def test_readiness_failure_norm_stats_missing_file(tmp_path, monkeypatch):
    probe, _ = make_probe_sequence([False, True])
    signals: list[int] = []
    monkeypatch.setattr(se, "_signal_group", lambda pgid, sig: signals.append(sig))
    monkeypatch.setattr(se, "_group_alive", lambda pgid: False)
    weights = tmp_path / "weights" / "no_stats"
    (weights / "params").mkdir(parents=True)
    endpoint = make_endpoint(
        tmp_path,
        health_probe=probe,
        popen_factory=RecordingPopen(),
        weights_dir=str(weights),
    )
    with pytest.raises(ToolEndpointError) as caught:
        await endpoint.start(make_request(), make_context(), FakeEmitter())
    assert caught.value.error.code == "READINESS_FAIL"
    assert caught.value.error.details["stage"] == "norm_stats"


async def test_readiness_failure_attached_never_kills(tmp_path, monkeypatch):
    probe, _ = make_probe_sequence([True])  # attach：非 owned
    signals: list[int] = []
    monkeypatch.setattr(se, "_signal_group", lambda pgid, sig: signals.append(sig))
    endpoint = make_endpoint(
        tmp_path, health_probe=probe, infer_once=lambda: {"action_dim": 14}
    )
    with pytest.raises(ToolEndpointError) as caught:
        await endpoint.start(make_request(), make_context(), FakeEmitter())
    assert caught.value.error.code == "READINESS_FAIL"
    assert signals == []  # attached 不杀进程


async def test_launch_missing_weights_params_fails_fast(tmp_path, monkeypatch):
    probe, _ = make_probe_sequence([False, True])
    signals: list[int] = []
    monkeypatch.setattr(se, "_signal_group", lambda pgid, sig: signals.append(sig))
    endpoint = make_endpoint(
        tmp_path, health_probe=probe, weights_dir=str(tmp_path / "weights" / "empty")
    )
    with pytest.raises(ToolEndpointError) as caught:
        await endpoint.start(make_request(), make_context(), FakeEmitter())
    assert caught.value.error.code == "READINESS_FAIL"
    assert caught.value.error.details["stage"] == "ckpt"
    assert signals == []  # 尚未 launch，无进程可 teardown


# —— 幂等 / 冲突 / 忙 / 参数 ——


async def test_duplicate_key_idempotent_no_relaunch(tmp_path):
    probe, _ = make_probe_sequence([True])
    popen = RecordingPopen()
    endpoint = make_endpoint(tmp_path, health_probe=probe, popen_factory=popen)
    context = make_context("inv-dup")
    first = await endpoint.start(make_request(), context, FakeEmitter())
    assert first.details["idempotent"] is False
    second = await endpoint.start(make_request(), context, FakeEmitter())
    assert second.details["idempotent"] is True
    assert second.details["owned"] is first.details["owned"]
    assert second.details["readiness"] == first.details["readiness"]
    assert popen.calls == []


async def test_duplicate_key_different_arguments_conflict(tmp_path):
    endpoint = make_endpoint(tmp_path, health_probe=lambda url, t: True)
    context = make_context("inv-conf")
    await endpoint.start(make_request(), context, FakeEmitter())
    with pytest.raises(ToolEndpointError) as caught:
        await endpoint.start(
            make_request(server_url="ws://other:9000"), context, FakeEmitter()
        )
    assert caught.value.error.code == "FORGE_EXECUTION_CONFLICT"


async def test_busy_second_execution(tmp_path):
    endpoint = make_endpoint(tmp_path, health_probe=lambda url, t: True)
    await endpoint.start(make_request(), make_context("inv-a"), FakeEmitter())
    with pytest.raises(ToolEndpointError) as caught:
        await endpoint.start(make_request(), make_context("inv-b"), FakeEmitter())
    assert caught.value.error.code == "FORGE_ENDPOINT_BUSY"
    assert caught.value.error.retryable is True


async def test_start_arguments_validation(tmp_path):
    endpoint = make_endpoint(tmp_path, health_probe=lambda url, t: True)
    with pytest.raises(ToolEndpointError) as caught:
        await endpoint.start(make_request(foo="bar"), make_context(), FakeEmitter())
    assert caught.value.error.code == "POLICY_RUNNER_INVALID_ARGUMENTS"
    with pytest.raises(ToolEndpointError) as caught2:
        await endpoint.start(
            make_request(server_url=""), make_context("inv-x"), FakeEmitter()
        )
    assert caught2.value.error.code == "POLICY_RUNNER_INVALID_ARGUMENTS"


async def test_start_arguments_server_url_override(tmp_path):
    seen: list[str] = []

    def probe(server_url: str, timeout_s: float) -> bool:
        seen.append(server_url)
        return True

    endpoint = make_endpoint(tmp_path, health_probe=probe)
    accepted = await endpoint.start(
        make_request(server_url="ws://127.0.0.1:9000"), make_context(), FakeEmitter()
    )
    assert seen[0] == "ws://127.0.0.1:9000"
    assert accepted.details["server_url"] == "ws://127.0.0.1:9000"


# —— stop 与相位迁移 ——


async def _started_owned(tmp_path, monkeypatch, invocation="inv-stop") -> tuple:
    # [False, True] x2：首启（attach 失败 → launch → 就绪）与测试内可能的二次 start
    # 各自消耗一对，防止序列耗尽后退化为 attach 路径。
    probe, _ = make_probe_sequence([False, True, False, True])
    popen = RecordingPopen(pid=5555)
    signals: list[int] = []
    monkeypatch.setattr(se, "_signal_group", lambda pgid, sig: signals.append(sig))
    monkeypatch.setattr(se, "_group_alive", lambda pgid: False)
    endpoint = make_endpoint(tmp_path, health_probe=probe, popen_factory=popen)
    emitter = FakeEmitter()
    accepted = await endpoint.start(make_request(), make_context(invocation), emitter)
    assert accepted.details["owned"] is True
    return endpoint, emitter, signals, popen


async def test_stop_owned_sigterm_then_sigkill(tmp_path, monkeypatch):
    probe, _ = make_probe_sequence([False, True])
    popen = RecordingPopen(pid=5555)
    signals: list[int] = []
    monkeypatch.setattr(se, "_signal_group", lambda pgid, sig: signals.append(sig))
    monkeypatch.setattr(se, "_group_alive", lambda pgid: True)  # SIGTERM 宽限内不消亡
    endpoint = make_endpoint(
        tmp_path, health_probe=probe, popen_factory=popen, stop_grace_s=0.02
    )
    context = make_context("inv-kill")
    key = context.execution_key
    await endpoint.start(make_request(), context, FakeEmitter())
    response = await endpoint.stop(key, "operator requested")
    assert response.command == "stop"
    assert response.status == "accepted"
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    result = await endpoint.result(key)
    assert result.status == "available"
    assert result.result is not None
    assert result.result.status == "stopped"
    assert dict(result.result.outputs) == {
        "server_url": "ws://127.0.0.1:8000",
        "pid": 5555,
        "owned": True,
        "reason": "operator requested",
    }


async def test_stop_owned_sigterm_suffices_grace(tmp_path, monkeypatch):
    endpoint, emitter, signals, popen = await _started_owned(tmp_path, monkeypatch)
    key = make_context("inv-stop").execution_key
    response = await endpoint.stop(key)
    assert response.status == "accepted"
    assert signals == [signal.SIGTERM]  # 组及时消亡，无 SIGKILL
    assert popen.process.waits >= 1
    result = await endpoint.result(key)
    assert result.status == "available"
    assert result.result is not None
    assert result.result.status == "stopped"
    assert result.result.outputs["owned"] is True
    assert len(emitter.events) == 1
    assert emitter.events[0].type == "stopped"
    assert dict(emitter.events[0].data) == {
        "server_url": "ws://127.0.0.1:8000",
        "pid": 5555,
        "owned": True,
    }
    # stop 后并发释放：可再启动新会话
    accepted = await endpoint.start(make_request(), make_context("inv-2nd"), FakeEmitter())
    assert accepted.details["owned"] is True


async def test_stop_attached_never_kills(tmp_path, monkeypatch):
    probe, _ = make_probe_sequence([True])
    signals: list[int] = []
    monkeypatch.setattr(se, "_signal_group", lambda pgid, sig: signals.append(sig))
    endpoint = make_endpoint(tmp_path, health_probe=probe)
    key = make_context("inv-attach").execution_key
    await endpoint.start(make_request(), make_context("inv-attach"), FakeEmitter())
    response = await endpoint.stop(key, "attached stop")
    assert response.status == "accepted"
    assert signals == []
    result = await endpoint.result(key)
    assert result.status == "available"
    assert result.result is not None and result.result.status == "stopped"
    assert result.result.outputs["owned"] is False


async def test_stop_unknown_rejected_and_second_stop_terminal(tmp_path):
    endpoint = make_endpoint(tmp_path, health_probe=lambda url, t: True)
    key = make_context("inv-t").execution_key
    await endpoint.start(make_request(), make_context("inv-t"), FakeEmitter())
    unknown = await endpoint.stop(make_context("inv-missing").execution_key)
    assert unknown.status == "rejected"
    assert unknown.error is not None
    assert unknown.error.code == "FORGE_EXECUTION_NOT_FOUND"
    first = await endpoint.stop(key)
    assert first.status == "accepted"
    second = await endpoint.stop(key)
    assert second.status == "terminal"


async def test_phase_transitions_and_status_details(tmp_path, monkeypatch):
    endpoint, emitter, signals, popen = await _started_owned(tmp_path, monkeypatch)
    key = make_context("inv-stop").execution_key
    running = await endpoint.status(key)
    assert running.phase == "running"
    assert running.details["server_url"] == "ws://127.0.0.1:8000"
    assert running.details["pid"] == 5555
    assert running.details["owned"] is True
    assert running.details["readiness"]["infer_once"]["action_dim"] == 7
    pending = await endpoint.result(key)
    assert pending.status == "pending"
    await endpoint.stop(key)
    stopped = await endpoint.status(key)
    assert stopped.phase == "stopped"
    available = await endpoint.result(key)
    assert available.status == "available"
    unknown = await endpoint.status(make_context("inv-none").execution_key)
    assert unknown.phase == "unknown"
    assert unknown.error is not None
    assert unknown.error.code == "FORGE_EXECUTION_NOT_FOUND"
    missing = await endpoint.result(make_context("inv-none").execution_key)
    assert missing.status == "not_found"


async def test_shutdown_terminates_owned(tmp_path, monkeypatch):
    endpoint, emitter, signals, popen = await _started_owned(
        tmp_path, monkeypatch, "inv-shut"
    )
    outcome = await endpoint.shutdown()
    assert outcome.active_key is not None
    assert outcome.owned_process_terminated is True
    assert signals == [signal.SIGTERM]
    # 关闭后拒绝新 start
    with pytest.raises(ToolEndpointError) as caught:
        await endpoint.start(make_request(), make_context("inv-late"), FakeEmitter())
    assert caught.value.error.code == "FORGE_ENDPOINT_STOPPING"


# —— 配置：仓库内打包 YAML 的 session 段 ——


def test_config_bundled_yaml_session_section():
    from kai0_inference.inference.config import PolicyNodeConfig

    config = PolicyNodeConfig.from_yaml_path(
        Path(__file__).resolve().parents[1] / "config" / "inference" / "kai0_libero.yaml"
    )
    assert config.mode == "session_endpoint"
    assert config.session is not None
    assert config.session.server_url == "ws://127.0.0.1:8000"
    assert config.session.weights_dir == "weights/pi05_libero"
    assert config.session.norm_stats_sha256 == se.DEFAULT_NORM_STATS_SHA256
    assert config.session.readiness_timeout_s == 600.0


def test_norm_stats_path_matches_ocdbt_two_level_layout(tmp_path):
    """norm_stats 锚点须匹配 start.sh 的 GCS ocdbt 落盘布局（assets/<org>/<repo>/）。"""
    nested = tmp_path / "assets" / "physical-intelligence" / "libero" / "norm_stats.json"
    nested.parent.mkdir(parents=True)
    nested.write_text('{"state": {}}', encoding="utf-8")
    assert se._norm_stats_path(tmp_path) == nested


def test_norm_stats_path_flat_layout_and_missing(tmp_path):
    flat = tmp_path / "assets" / "libero" / "norm_stats.json"
    flat.parent.mkdir(parents=True)
    flat.write_text("{}", encoding="utf-8")
    assert se._norm_stats_path(tmp_path) == flat
    assert se._norm_stats_path(tmp_path / "no-assets") is None


# —— 工具面集成：真实 handler + binding 线路径 ——


def _attach_endpoint(tmp_path) -> Kai0ServeSessionEndpoint:
    return make_endpoint(tmp_path, health_probe=lambda url, t: True)


def _decode_batch(batch) -> Any:
    """RecordBatch → ToolMessage → ToolEnvelope。"""
    return tool_message_to_envelope(ToolMessage.from_arrow(batch))


async def _dispatch(binding: DoraToolEndpointBinding, envelope) -> tuple:
    """ToolEnvelope → Arrow 载体 → binding.dispatch_input。

    session invoke：响应与缓冲事件一律经 event_sink 发布，返回空元组；
    其余请求：响应元组直接返回。
    """
    message = tool_envelope_to_message(envelope)
    if envelope.message_type == "tool.invoke.request":
        result = await binding.dispatch_input(message.to_arrow())
        assert result == ()
        return ()
    return tuple(await binding.dispatch_input(message.to_arrow()))


async def test_binding_invoke_accepted_stop_and_result(tmp_path):
    endpoint = _attach_endpoint(tmp_path)
    handler = ToolEndpointHandler(
        DESCRIPTOR, endpoint_instance_id=INSTANCE_ID, operations={OPERATION: endpoint}
    )
    published: list = []

    async def sink(batch) -> None:
        published.append(batch)

    binding = DoraToolEndpointBinding(handler, event_sink=sink)
    context = make_context("inv-wire")

    # 1) invoke → accepted（响应经 event_sink 发布）
    invoke = make_invoke_request_envelope(
        make_request(), context, request_id="req-1", endpoint_instance_id=INSTANCE_ID
    )
    await _dispatch(binding, invoke)
    assert len(published) == 1
    accepted = invoke_response_from_payload(_decode_batch(published[0]).payload)
    assert isinstance(accepted, ToolAccepted)
    assert accepted.details["owned"] is False

    # 2) 重复 invoke → 幂等回放 accepted（handler 短路，不再触达 endpoint.start）
    published.clear()
    await _dispatch(binding, invoke)
    assert len(published) == 1
    replay = invoke_response_from_payload(_decode_batch(published[0]).payload)
    assert isinstance(replay, ToolAccepted)
    assert replay.details["owned"] is False

    # 3) stop → accepted（owned=False，不杀进程）
    stop_envelope = make_control_request_envelope(
        "stop",
        context,
        request_id="req-2",
        endpoint_instance_id=INSTANCE_ID,
        reason="wire stop",
    )
    (stop_batch,) = await _dispatch(binding, stop_envelope)
    stop_response = control_response_from_payload(_decode_batch(stop_batch).payload)
    assert stop_response.command == "stop"
    assert stop_response.status == "accepted"
    assert dict(stop_response.details) == {
        "server_url": "ws://127.0.0.1:8000",
        "reason": "wire stop",
    }

    # 4) status → stopped；result → available stopped
    (status_batch,) = await _dispatch(
        binding,
        make_status_request_envelope(
            context, request_id="req-3", endpoint_instance_id=INSTANCE_ID
        ),
    )
    status = status_response_from_payload(_decode_batch(status_batch).payload)
    assert status.phase == "stopped"

    (result_batch,) = await _dispatch(
        binding,
        make_result_request_envelope(
            context, request_id="req-4", endpoint_instance_id=INSTANCE_ID
        ),
    )
    result = result_response_from_payload(_decode_batch(result_batch).payload)
    assert result.status == "available"
    assert result.result is not None
    assert result.result.status == "stopped"
    assert dict(result.result.outputs)["owned"] is False


async def test_binding_invoke_rejected_on_readiness_failure(tmp_path):
    """就绪门失败经 handler 转 rejected invoke 响应（ToolError READINESS_FAIL）。"""
    endpoint = _attach_endpoint(tmp_path)
    endpoint._infer_once = lambda: {"action_dim": 14}
    handler = ToolEndpointHandler(
        DESCRIPTOR, endpoint_instance_id=INSTANCE_ID, operations={OPERATION: endpoint}
    )
    published: list = []

    async def sink(batch) -> None:
        published.append(batch)

    binding = DoraToolEndpointBinding(handler, event_sink=sink)
    context = make_context("inv-reject")
    invoke = make_invoke_request_envelope(
        make_request(), context, request_id="req-9", endpoint_instance_id=INSTANCE_ID
    )
    await _dispatch(binding, invoke)
    assert len(published) == 1
    envelope = _decode_batch(published[0])
    assert envelope.message_type == "tool.invoke.response"
    rejected = invoke_response_from_payload(envelope.payload)
    assert isinstance(rejected, ToolError)
    assert rejected.code == "READINESS_FAIL"
    assert rejected.retryable is True
