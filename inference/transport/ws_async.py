"""WS 常驻连接传输层：RTC 模式在一条连接上反复请求（对齐 openpi websocket 协议）。

与 ws_client.WsActionTransport 同协议（metadata 首帧 / 文本帧即异常 / 禁用压缩与
ping），差异仅在连接生命周期：connect() 后 request_actions() 反复复用同一连接，
close() 断开。供 Kai0RtcAdapter 在专属 asyncio event loop（后台线程）中调用。

协议（对照本仓库 server/ 快照的 websocket_policy_server.py）：
- 连接后服务端先发一帧 metadata（msgpack 二进制 dict）；文本帧即异常；
- 客户端发 observation payload（msgpack_codec.packb，openpi-client 字典线格式）；
- 服务端回 {state, actions, policy_timing, server_timing}；文本帧为 traceback 异常信息；
- compression=None、max_size=None（禁用压缩、不限制帧大小）；
- ping_interval=None、ping_timeout=None（服务端推理期间不因 keepalive 超时断开）。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from websockets.asyncio.client import connect

from kai0_inference.inference.transport import msgpack_codec
from kai0_inference.inference.transport.base import AsyncPersistentTransport

logger = logging.getLogger(__name__)


class WsAsyncTransport(AsyncPersistentTransport):
    """常驻连接实现：connect 建立连接并消费 metadata 首帧。"""

    def __init__(self, server_url: str, *, timeout_s: float = 60.0) -> None:
        if not server_url or not str(server_url).strip():
            raise ValueError("server_url must not be empty")
        if not float(timeout_s) > 0:
            raise ValueError("timeout_s must be positive")
        self._server_url = str(server_url).strip()
        self._timeout_s = float(timeout_s)
        self._websocket: Any | None = None

    @property
    def server_url(self) -> str:
        return self._server_url

    async def connect(self) -> None:
        self._websocket = await connect(
            self._server_url,
            compression=None,
            max_size=None,
            ping_interval=None,
            ping_timeout=None,
            open_timeout=self._timeout_s,
            close_timeout=5,
        )
        # 1) metadata 首帧（二进制 msgpack dict；文本帧即异常）
        first = await asyncio.wait_for(self._websocket.recv(), timeout=self._timeout_s)
        if isinstance(first, str):
            raise RuntimeError(f"unexpected text metadata frame from server: {first}")
        metadata = msgpack_codec.unpackb(first)
        if not isinstance(metadata, dict):
            raise RuntimeError(f"server metadata must be a dict, got {type(metadata)}")
        logger.debug("server metadata keys: %s", sorted(metadata))

    async def request_actions(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._websocket is None:
            raise RuntimeError("not connected: call connect() before request_actions()")
        await self._websocket.send(msgpack_codec.packb(payload))
        response = await asyncio.wait_for(self._websocket.recv(), timeout=self._timeout_s)
        if isinstance(response, str):
            # 文本帧为服务端异常 traceback（真实服务端与 fake server 行为一致）
            raise RuntimeError(f"policy server error:\n{response}")
        result = msgpack_codec.unpackb(response)
        if not isinstance(result, dict):
            raise RuntimeError(f"server response must be a dict, got {type(result)}")
        return result

    async def close(self) -> None:
        websocket = self._websocket
        self._websocket = None
        if websocket is None:
            return
        try:
            await websocket.close()
        except Exception:
            logger.debug("error closing websocket", exc_info=True)
