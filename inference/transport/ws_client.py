"""WS 短连接传输层：同步模式每请求一次连接（对齐 openpi websocket 协议）。

协议（对照本仓库 server/ 快照的 websocket_policy_server.py）：
- 连接后服务端先发一帧 metadata（msgpack 二进制 dict）；文本帧即异常；
- 客户端发 observation payload（msgpack_codec.packb，openpi-client 字典线格式）；
- 服务端回 {state, actions, policy_timing, server_timing}；文本帧为 traceback 异常信息；
- compression=None、max_size=None（禁用压缩、不限制帧大小）；
- ping_interval=None、ping_timeout=None（服务端推理期间不因 keepalive 超时断开）。
"""

from __future__ import annotations

import logging
from typing import Any

# 与 openpi-client 内嵌 msgpack_numpy 一致的字典线格式（非 PyPI msgpack-numpy 的
# ExtType 格式；混用会让服务端解码失败，见 msgpack_codec 模块说明）
from kai0_inference.inference.transport import msgpack_codec
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from kai0_inference.inference.transport.base import ActionTransport

logger = logging.getLogger(__name__)


class WsActionTransport(ActionTransport):
    """同步模式：每次 request_actions 建立短连接，请求完成后断开。"""

    def __init__(self, server_url: str, *, timeout_s: float = 60.0) -> None:
        if not server_url or not str(server_url).strip():
            raise ValueError("server_url must not be empty")
        if not float(timeout_s) > 0:
            raise ValueError("timeout_s must be positive")
        self._server_url = str(server_url).strip()
        self._timeout_s = float(timeout_s)

    @property
    def server_url(self) -> str:
        return self._server_url

    def request_actions(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with connect(
                self._server_url,
                compression=None,
                max_size=None,
                ping_interval=None,
                ping_timeout=None,
                open_timeout=self._timeout_s,
                close_timeout=5,
            ) as websocket:
                # 1) metadata 首帧（二进制 msgpack dict）
                first = websocket.recv(timeout=self._timeout_s)
                if isinstance(first, str):
                    raise RuntimeError(f"unexpected text metadata frame from server: {first}")
                metadata = msgpack_codec.unpackb(first)
                if not isinstance(metadata, dict):
                    raise RuntimeError(f"server metadata must be a dict, got {type(metadata)}")
                logger.debug("server metadata keys: %s", sorted(metadata))

                # 2) 发送 observation payload
                websocket.send(msgpack_codec.packb(payload))

                # 3) 接收响应
                response = websocket.recv(timeout=self._timeout_s)
        except ConnectionClosed as exc:
            raise RuntimeError(f"policy server closed the connection: {exc}") from exc
        except (OSError, TimeoutError) as exc:
            raise RuntimeError(
                f"cannot reach policy server at {self._server_url}: {exc}"
            ) from exc

        if isinstance(response, str):
            # 文本帧为服务端异常 traceback（真实服务端与 fake server 行为一致）
            raise RuntimeError(f"policy server error:\n{response}")
        result = msgpack_codec.unpackb(response)
        if not isinstance(result, dict):
            raise RuntimeError(f"server response must be a dict, got {type(result)}")
        return result
