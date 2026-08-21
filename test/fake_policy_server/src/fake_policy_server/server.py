"""与 openpi websocket_policy_server.WebsocketPolicyServer 协议对齐的假服务端。

协议对齐项（逐字段对照本仓库 server/ 快照的 websocket_policy_server.py）：
- 连接建立后先发送一帧 metadata dict（msgpack_numpy.Packer 打包）；
- 请求循环：收 observation（msgpack 帧）→ 应用 sticky task 值（prompt_id/task_index/subtask）
  → policy.infer → 附加 server_timing（infer_ms；第二次请求起附 prev_total_ms）→ 回包；
- 异常时发送文本帧 traceback，并以 1011 INTERNAL_ERROR 关闭连接；
- /healthz 返回 200 "OK\\n"；
- compression=None、max_size=None（禁用压缩、不限制帧大小）。
"""

from __future__ import annotations

import asyncio
import http
import logging
import time
import traceback

try:
    # 与 openpi-client 一致的 numpy 扩展编解码（lebedov/msgpack-numpy 线格式）
    import msgpack_numpy
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "缺少 msgpack-numpy 依赖，请先安装：pip install msgpack-numpy"
    ) from exc

import websockets.asyncio.server as _server
import websockets.frames

from fake_policy_server.policy import FakePolicy

logger = logging.getLogger(__name__)

_TASK_KEYS = ("prompt_id", "task_index", "subtask")

# 测试钩子：observation 携带该键时服务端主动抛异常，用于验证错误帧行为。
_ERROR_TRIGGER_KEY = "trigger_error"


class FakePolicyServer:
    """协议行为与真实 WebsocketPolicyServer 一致，仅 policy 为 FakePolicy。"""

    def __init__(
        self,
        policy: FakePolicy,
        host: str = "0.0.0.0",
        port: int | None = 8000,
        metadata: dict | None = None,
        default_task_index: int | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._default_task_value = default_task_index
        # 白盒观测点：当前生效的 sticky task 值，测试用。
        self.last_task_value: int | None = default_task_index
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        last_task_value = self._default_task_value
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                if _ERROR_TRIGGER_KEY in obs:
                    raise RuntimeError("trigger_error requested by observation")
                obs, last_task_value = _apply_sticky_task_value(obs, last_task_value)
                self.last_task_value = last_task_value

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def _apply_sticky_task_value(obs: dict, last_task_value: int | None) -> tuple[dict, int | None]:
    current_task_value = _extract_task_value(obs)
    if current_task_value is not None:
        return obs, current_task_value

    if last_task_value is None:
        return obs, None

    obs = dict(obs)
    obs["prompt_id"] = last_task_value
    logger.info("Reusing sticky prompt_id=%s for current request", last_task_value)
    return obs, last_task_value


def _extract_task_value(obs: dict) -> int | None:
    for key in _TASK_KEYS:
        if key not in obs:
            continue
        try:
            return _to_scalar_int(obs[key])
        except ValueError:
            logger.warning("Ignoring non-scalar %s=%r", key, obs[key])
            return None
    return None


def _to_scalar_int(value) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected scalar task value, got sequence length {len(value)}")
        return int(value[0])
    return int(value)
