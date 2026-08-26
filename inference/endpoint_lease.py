"""Endpoint 注册租约（Dora 输入活动驱动）。

从 forge 参考实现 gripper_action_policy_node/endpoint_lease.py 迁移：
注册/续租节奏由 gateway 的 lease_ttl_ms 决定（ttl/3 提前续租，10s 租约 ≈ 3.3s 一次）；
响应超时未达则 0.2s 后重发；rejected 且不可重试则 24h 后重试。
"""

from __future__ import annotations

import uuid

from forge_tool import (
    ToolEndpointDescriptor,
    ToolEnvelope,
    make_registration_envelope,
    make_unregister_envelope,
)


class EndpointLease:
    def __init__(
        self,
        descriptor: ToolEndpointDescriptor,
        endpoint_instance_id: str,
        *,
        response_timeout_s: float = 1.0,
    ) -> None:
        self.descriptor = descriptor
        self.endpoint_instance_id = endpoint_instance_id
        self.response_timeout_s = response_timeout_s
        self.pending: ToolEnvelope | None = None
        self.response_deadline = 0.0
        self.next_registration = 0.0
        self.stopped = False

    def poll(self, now: float) -> ToolEnvelope | None:
        if self.stopped:
            return None
        if self.pending is not None and now < self.response_deadline:
            return None
        if self.pending is not None:
            self.pending = None
            self.next_registration = now + 0.2
        if now < self.next_registration:
            return None
        self.pending = make_registration_envelope(
            self.descriptor,
            endpoint_instance_id=self.endpoint_instance_id,
            request_id=str(uuid.uuid4()),
        )
        self.response_deadline = now + self.response_timeout_s
        return self.pending

    def acknowledge(self, response: ToolEnvelope, now: float) -> bool:
        if (
            self.pending is None
            or response.message_type != "endpoint.registry.response"
            or response.request_id != self.pending.request_id
        ):
            return False
        payload = response.payload
        self.pending = None
        if payload.get("status") == "accepted":
            ttl_ms = payload.get("lease_ttl_ms")
            if not isinstance(ttl_ms, int) or ttl_ms < 1:
                return False
            self.next_registration = now + ttl_ms / 3_000
        else:
            error = payload.get("error")
            retryable = isinstance(error, dict) and error.get("retryable") is True
            self.next_registration = now + (0.2 if retryable else 86_400)
        return True

    def stop(self) -> ToolEnvelope | None:
        if self.stopped:
            return None
        self.stopped = True
        return make_unregister_envelope(
            endpoint_id=self.descriptor.endpoint_id,
            endpoint_instance_id=self.endpoint_instance_id,
            request_id=str(uuid.uuid4()),
        )


__all__ = ["EndpointLease"]
