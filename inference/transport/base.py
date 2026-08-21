"""动作块请求传输层抽象。

策略层只依赖本接口，不感知具体连接方式：
- ActionTransport：同步短连接（Kai0SyncAdapter 用，每请求一次连接）；
- AsyncPersistentTransport：异步常驻连接（Kai0RtcAdapter 用，推理线程持有）。

测试通过注入指向真实 socket 的传输层实现；协议正确性由 infer-once 冒烟与
仿真/真机（真服务端）验证。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ActionTransport(ABC):
    """向策略服务端请求动作块的传输层。"""

    @abstractmethod
    def request_actions(self, payload: dict[str, Any]) -> dict[str, Any]:
        """发送 observation payload，返回服务端原始响应 dict（须含 actions 字段）。

        失败（连接不可达 / 服务端错误帧 / 超时）以异常形式抛出。
        """


class AsyncPersistentTransport(ABC):
    """异步常驻连接传输层（RTC 用）：connect 后在同一连接上反复请求。

    适配器在专属 asyncio event loop（后台线程）中调用；实现需保证：
    - connect() 消费服务端 metadata 首帧（并校验为 dict）；
    - request_actions() 依赖已建立的连接，失败以异常形式抛出；
    - close() 幂等、可安全重复调用。
    """

    @abstractmethod
    async def connect(self) -> None:
        """建立连接并消费 metadata 首帧。"""

    @abstractmethod
    async def request_actions(self, payload: dict[str, Any]) -> dict[str, Any]:
        """在常驻连接上发送 payload，返回服务端原始响应 dict。"""

    @abstractmethod
    async def close(self) -> None:
        """断开连接（幂等）。"""
