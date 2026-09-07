"""Kai0SessionPolicyRunner 工具 worker 回归：就绪门不阻塞 dora 事件线程。

lease 缺陷修复回归（serve 就绪门最长 600s 曾占住事件线程导致租约过期）：
- serve invoke 的门在 worker 线程执行，事件线程在门进行期间照常驱动 lease
  与 tick（以 elapsed 上界 + 首轮注册发出断言）；
- STOP 经 cancel_active_gate 中止门、teardown owned 进程，join 有界返回；
- worker 的 invoke 响应经 outbox 回事件线程发出（消息面断言）。
"""

from __future__ import annotations

import hashlib
import signal
import time
from pathlib import Path
from typing import Any

import pytest
from forge_msgs import ToolMessage
from forge_tool import ToolContext, ToolExecutionKey, ToolRequest, make_invoke_request_envelope
from forge_tool.dora import tool_envelope_to_message, tool_message_to_envelope

from kai0_inference.inference import session_endpoint as se
from kai0_inference.inference.session_endpoint import Kai0ServeSessionEndpoint
from kai0_inference.inference.session_runner import Kai0SessionPolicyRunner

RUNNER_INSTANCE = "runner-inst-1"


@pytest.fixture(autouse=True)
def _stub_nvidia_smi(monkeypatch):
    """GPU 释放确认仅日志；测试跳过 nvidia-smi（GPU-free，不起子进程）。"""
    real_which = se.shutil.which

    def which(name: str):
        if name == "nvidia-smi":
            return None
        return real_which(name)

    monkeypatch.setattr(se.shutil, "which", which)


class FakeNode:
    """事件线程输出汇：记录 send_output（worker 不得直接调用它）。"""

    def __init__(self) -> None:
        self.sent: list[tuple[str, Any]] = []

    def send_output(self, output_id: str, batch: Any) -> None:
        self.sent.append((output_id, batch))


class FakePolicy:
    """数据面桩：本文件不触达策略推理，仅满足 runner 构造。"""

    def is_observation_needed(self) -> bool:
        return False


class FakeProcess:
    """Popen 假实现：记录 wait 调用（不真等待）。"""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.waits = 0

    def wait(self, timeout: float | None = None) -> int:
        self.waits += 1
        return 0


class RecordingPopen:
    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.calls: list[dict[str, Any]] = []
        self.process = FakeProcess(pid)

    def __call__(self, argv, env, log_path: Path):
        self.calls.append({"argv": list(argv), "env": dict(env), "log_path": Path(log_path)})
        return self.process


def make_weights(tmp_path: Path, content: bytes = b'{"norm_stats": {}}') -> tuple[Path, str]:
    """LIBERO 布局假权重：<dir>/params/ + <dir>/assets/<repo>/norm_stats.json。"""
    weights = tmp_path / "weights" / "pi05_libero"
    (weights / "params").mkdir(parents=True)
    stats = weights / "assets" / "pi05_libero" / "norm_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_bytes(content)
    return weights, hashlib.sha256(content).hexdigest()


def make_slow_endpoint(
    tmp_path: Path, monkeypatch, signals: list[int]
) -> tuple[Kai0ServeSessionEndpoint, RecordingPopen]:
    """永远不健康的 healthz → attach 失败 → launch → 600s 门等待（只有取消能终止）。"""
    weights, digest = make_weights(tmp_path)
    monkeypatch.setattr(se, "_signal_group", lambda pgid, sig: signals.append(sig))
    monkeypatch.setattr(se, "_group_alive", lambda pgid: False)
    popen = RecordingPopen()
    endpoint = Kai0ServeSessionEndpoint(
        server_url="ws://127.0.0.1:8000",
        weights_dir=str(weights),
        norm_stats_sha256=digest,
        readiness_timeout_s=600.0,
        stop_grace_s=0.05,
        poll_interval_s=0.01,
        health_timeout_s=0.1,
        health_probe=lambda url, t: False,
        infer_once=lambda: {"action_dim": 7},
        popen_factory=popen,
    )
    return endpoint, popen


def make_runner(
    tmp_path: Path, monkeypatch
) -> tuple[Kai0SessionPolicyRunner, FakeNode, RecordingPopen, list[int]]:
    signals: list[int] = []
    endpoint, popen = make_slow_endpoint(tmp_path, monkeypatch, signals)
    node = FakeNode()
    runner = Kai0SessionPolicyRunner(
        node,
        policy=FakePolicy(),
        endpoint=endpoint,
        joint_order=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
        image_input_id_to_alias={},
        build_action=lambda a: None,
        endpoint_instance_id=RUNNER_INSTANCE,
    )
    return runner, node, popen, signals


def make_events(runner: Kai0SessionPolicyRunner):
    """serve invoke（门挂起在 worker）→ 若干 tick → STOP 的事件序列。"""

    def events():
        context = ToolContext(
            execution_key=ToolExecutionKey("inv-1", "att-1"),
            tool_id="kai0_libero.policy",
            implementation_id="impl-1",
            endpoint_id="policy.runner",
            operation="serve",
        )
        invoke = make_invoke_request_envelope(
            ToolRequest(arguments={}),
            context,
            request_id="req-1",
            endpoint_instance_id=runner.endpoint_instance_id,
        )
        yield {
            "type": "INPUT",
            "id": "tool_in",
            "value": tool_envelope_to_message(invoke).to_arrow(),
        }
        for _ in range(10):
            yield {"type": "INPUT", "id": "tick", "value": None}
        yield {"type": "STOP"}

    return events()


def _tool_message_types(node: FakeNode) -> list[str]:
    return [
        tool_message_to_envelope(ToolMessage.from_arrow(batch)).message_type
        for out_id, batch in node.sent
        if out_id == "tool_out"
    ]


def test_gate_does_not_block_event_loop_and_stop_tears_down(tmp_path, monkeypatch):
    """600s 门在 worker 上：事件线程跑完全部事件（含 STOP）耗时远小于门长。"""
    runner, node, popen, signals = make_runner(tmp_path, monkeypatch)
    started = time.monotonic()
    rc = runner.run(make_events(runner))
    elapsed = time.monotonic() - started
    assert rc == 0
    assert elapsed < 10  # 若门占住事件线程，本断言需等满 600s
    # 门被取消 → launch 的进程已 teardown（SIGTERM 且 wait 回收）。
    assert signals == [signal.SIGTERM]
    assert len(popen.calls) == 1
    assert popen.process.waits >= 1


def test_worker_response_delivered_via_outbox(tmp_path, monkeypatch):
    """worker 的响应（含门取消后的 rejected invoke）经 outbox 由事件线程发出。"""
    runner, node, popen, signals = make_runner(tmp_path, monkeypatch)
    rc = runner.run(make_events(runner))
    assert rc == 0
    types = _tool_message_types(node)
    # lease 首轮注册（run() 入口即发出）+ 门取消后的 rejected invoke 响应。
    assert any("register" in t for t in types)
    assert "tool.invoke.response" in types
