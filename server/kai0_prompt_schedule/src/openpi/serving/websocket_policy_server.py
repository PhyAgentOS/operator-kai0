import asyncio
import http
import logging
import os
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)
_TASK_KEYS = ("prompt_id", "task_index", "subtask")


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
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
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        last_task_value = _parse_default_task_value(
            os.environ.get("DEFAULT_PROMPT_ID") or os.environ.get("DEFAULT_TASK_INDEX")
        )
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                obs, last_task_value = _apply_sticky_task_value(obs, last_task_value)

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
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
    # Continue with the normal request handling.
    return None


def _parse_default_task_value(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError:
        logger.warning("Ignoring invalid DEFAULT_TASK_INDEX=%r", value)
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
