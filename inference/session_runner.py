"""session_endpoint 模式的 Dora 运行器：工具面 + 数据面同节点。

- 工具面：ToolEndpointHandler + DoraToolEndpointBinding（tool_in → tool_out，
  event_sink 发布响应与缓冲事件）+ EndpointLease 周期注册/注销。
- 数据面：与 forge_policy.run_dora_policy_node 同构（policy_command/proprio_state/
  images → 20ms tick 门控 action 输出），过渡开关 mode=dora_plain 时仍走原路径。
- 工具 dispatch 在专职 worker 线程串行执行：serve 就绪门最长 600s，若占住
  dora 事件线程，lease 续期停摆、网关按 TTL 过期判 ambiguous（"pinned
  provider is no longer current"）。worker 的响应经 outbox 队列回事件线程统一
  send_output（dora 节点输出不跨线程）。STOP/退出时事件线程先置取消标志中止
  进行中的门，再排队 endpoint.shutdown()，teardown 顺序由 worker 队列保证。
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from typing import Any

import pyarrow as pa
from forge_msgs import PolicyCommand, ToolMessage
from forge_policy import (
    PolicyRuntimeState,
    apply_policy_command,
    build_policy_observation,
)
from forge_tool import ToolEndpointHandler, ToolEnvelope
from forge_tool.dora import (
    DoraToolEndpointBinding,
    tool_envelope_to_message,
    tool_message_to_envelope,
)

from kai0_inference.inference.endpoint_lease import EndpointLease
from kai0_inference.inference.session_endpoint import DESCRIPTOR, OPERATION, Kai0ServeSessionEndpoint

logger = logging.getLogger(__name__)

ActionBuilder = Callable[[Any], Any]

_WORKER_SHUTDOWN = object()
# STOP 兜底 join 上限：门取消 ≤ 1 个 poll 间隔 + healthz 探测 ≤ 5s +
# SIGTERM 宽限(默认 30s) + SIGKILL 路径余量。
_WORKER_JOIN_TIMEOUT_S = 90.0


class Kai0SessionPolicyRunner:
    def __init__(
        self,
        node: Any,
        *,
        policy: Any,
        endpoint: Kai0ServeSessionEndpoint,
        joint_order: list[str],
        image_input_id_to_alias: dict[str, str],
        build_action: ActionBuilder,
        policy_id: str = "default",
        alias_for_cameras: list[str] | None = None,
        auto_start: bool = False,
        emit_command_status: bool = True,
        call_lifecycle_hooks: bool = True,
        endpoint_instance_id: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.node = node
        self.policy = policy
        self.endpoint = endpoint
        self.joint_order = joint_order
        self.image_input_id_to_alias = image_input_id_to_alias
        self.build_action = build_action
        self.camera_aliases = alias_for_cameras or list(image_input_id_to_alias.values())
        self.emit_command_status = emit_command_status
        self.call_lifecycle_hooks = call_lifecycle_hooks
        self.endpoint_instance_id = endpoint_instance_id or str(uuid.uuid4())
        self.clock = clock
        self.handler = ToolEndpointHandler(
            DESCRIPTOR,
            endpoint_instance_id=self.endpoint_instance_id,
            operations={OPERATION: self.endpoint},
        )
        self.binding = DoraToolEndpointBinding(self.handler, event_sink=self._send_batch)
        self.lease = EndpointLease(DESCRIPTOR, self.endpoint_instance_id)
        # 工具 worker：dispatch 串行执行（端点状态单线程化，start/stop/status/
        # result 与就绪门天然互斥）；响应经 outbox 回事件线程发送。
        self._tool_requests: queue.Queue[Any] = queue.Queue()
        self._tool_outbox: queue.Queue[pa.RecordBatch] = queue.Queue()
        self._tool_worker: threading.Thread | None = None
        # 数据面状态（与 forge_policy.run_dora_policy_node 同构）
        self.state = PolicyRuntimeState(
            policy_id=policy_id,
            phase="running" if auto_start else "idle",
        )
        self.cached_proprio: Any | None = None
        self.cached_images: dict[str, Any] = {}

    # —— 输出 ——

    async def _send_batch(self, batch: pa.RecordBatch) -> None:
        # event_sink 回调发生在 worker 线程：只入 outbox，由事件线程统一发送。
        self._tool_outbox.put(batch)

    def _send_envelope(self, envelope: ToolEnvelope) -> None:
        self.node.send_output("tool_out", tool_envelope_to_message(envelope).to_arrow())

    def _send_command_status(self, result: Any) -> None:
        try:
            self.node.send_output(
                "policy_command_status", result.to_status(self.state.policy_id).to_arrow()
            )
        except Exception:
            logger.exception("failed to send policy_command_status")

    # —— 工具面 ——

    def _drain_tool_outbox(self) -> None:
        while True:
            try:
                batch = self._tool_outbox.get_nowait()
            except queue.Empty:
                return
            self.node.send_output("tool_out", batch)

    def _start_tool_worker(self) -> None:
        if self._tool_worker is not None:
            return
        self._tool_worker = threading.Thread(
            target=self._tool_worker_main, name="kai0-tool-worker", daemon=True
        )
        self._tool_worker.start()

    def _tool_worker_main(self) -> None:
        runner = asyncio.Runner()
        try:
            while True:
                item = self._tool_requests.get()
                if item is _WORKER_SHUTDOWN:
                    self._run_endpoint_shutdown(runner)
                    return
                try:
                    for batch in runner.run(self.binding.dispatch_input(item)):
                        self._tool_outbox.put(batch)
                except Exception:
                    logger.exception("tool dispatch failed")
        finally:
            runner.close()

    def _run_endpoint_shutdown(self, runner: asyncio.Runner) -> None:
        try:
            outcome = runner.run(self.endpoint.shutdown())
        except Exception:
            logger.exception("session endpoint shutdown failed")
            return
        if outcome.active_key is not None and not outcome.owned_process_terminated:
            logger.warning(
                "policy.runner shutdown leaves session %s to gateway stop; "
                "owned process termination is not claimed",
                outcome.active_key.invocation_id,
            )

    def _stop_tool_worker(self) -> None:
        if self._tool_worker is None:
            return
        # 先取消进行中的就绪门（事件线程置标志，worker 在下一个 poll 间隔内
        # 中止并 teardown owned 进程），再排队 shutdown；顺序由 worker 队列保证。
        self.endpoint.cancel_active_gate()
        self._tool_requests.put(_WORKER_SHUTDOWN)
        self._tool_worker.join(timeout=_WORKER_JOIN_TIMEOUT_S)
        if self._tool_worker.is_alive():
            logger.warning(
                "tool worker did not exit within %.0fs; abandoning",
                _WORKER_JOIN_TIMEOUT_S,
            )
        self._tool_worker = None
        self._drain_tool_outbox()

    def _drive_lease(self) -> None:
        request = self.lease.poll(self.clock())
        if request is not None:
            self._send_envelope(request)

    def _handle_tool(self, value: object) -> None:
        message = ToolMessage.from_arrow(value)
        envelope = tool_message_to_envelope(message)
        if envelope.message_type == "endpoint.registry.response":
            self.lease.acknowledge(envelope, self.clock())
            return
        # 工具请求一律交 worker 线程（含 ≤600s 就绪门），事件线程保持空闲以
        # 驱动 lease 与数据面；响应由 worker 入 outbox，事件线程在下一个
        # 事件（20ms tick / 注册回执）时 drain 发出。
        # 入队前清除取消标志（与 cancel_active_gate 配对）：取消是粘滞的，且
        # 只在事件线程串行执行——cancel 之后不会再有请求入队，无竞态窗口。
        self.endpoint.reset_active_gate()
        self._tool_requests.put(value)

    # —— 主循环 ——

    def run(self, events: Iterable[dict[str, Any]]) -> int:
        self._start_tool_worker()
        self._drive_lease()
        try:
            for event in events:
                result = self.process_event(event)
                if result is not None:
                    return result
            return 0
        finally:
            # 正常结束/异常退出也兜底 teardown（owned 服务不残留 GPU）。
            self._stop_tool_worker()

    def process_event(self, event: dict[str, Any]) -> int | None:
        event_type = event.get("type")
        self._drain_tool_outbox()
        self._drive_lease()
        if event_type == "STOP":
            # 节点退出兜底：先注销注册，再取消进行中的门并由 worker 依序
            # teardown（owned 进程随之消亡，GPU 释放 = 进程组消亡）。
            unregister = self.lease.stop()
            if unregister is not None:
                self._send_envelope(unregister)
            self._stop_tool_worker()
            return 0
        if event_type == "ERROR":
            logger.error("Dora node error: %s", event.get("error", "unknown"))
            self._stop_tool_worker()
            return 1
        if event_type != "INPUT":
            return None

        input_id = event.get("id")
        value = event.get("value")
        try:
            if input_id == "tool_in" and value is not None:
                self._handle_tool(value)
                return None

            # —— 数据面（与 forge_policy.run_dora_policy_node 同构）——
            if input_id == "policy_command" and value is not None:
                command = PolicyCommand.from_arrow(value)
                result = apply_policy_command(
                    state=self.state,
                    policy=self.policy,
                    command=command,
                    call_lifecycle_hooks=self.call_lifecycle_hooks,
                )
                if result is None:
                    return None
                if result.reset_observation_cache:
                    self.cached_proprio = None
                    self.cached_images.clear()
                if self.emit_command_status:
                    self._send_command_status(result)
                return None

            if input_id == "proprio_state" and value is not None:
                self.cached_proprio = value
                return None

            if input_id in self.image_input_id_to_alias and value is not None:
                self.cached_images[input_id] = value
                return None

            if input_id != "tick":
                return None

            if not self.state.is_running:
                return None

            if self.policy.is_observation_needed():
                observation = build_policy_observation(
                    proprio_payload=self.cached_proprio,
                    image_payloads=self.cached_images,
                    joint_order=self.joint_order,
                    image_input_id_to_alias=self.image_input_id_to_alias,
                )
                if observation is None:
                    return None
            else:
                observation = {}

            try:
                action_payload = self.policy.generate_action(observation, self.camera_aliases)
                if action_payload is None:
                    return None
                action = self.build_action(action_payload)
                self.node.send_output("action", action.to_arrow())
            except Exception:
                logger.exception("policy tick failed")
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("ignored invalid kai0 policy input: %s", exc)
        return None


__all__ = ["Kai0SessionPolicyRunner"]
