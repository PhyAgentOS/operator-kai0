"""Kai0RtcAdapter：RTC（real-time chunking）异步推理适配器。

RTC 语义（与真机部署口径一致）：

- **常驻连接 + 后台推理线程**：专属 asyncio event loop 线程持有常驻 WS 连接，
  按 rtc_inference_hz 周期请求；观测发布/请求经 Condition + generation 在主线程
  （dora tick）与推理线程间握手。
- **RTC 引导前缀**：请求 payload 在普通观测之上增加三字段（服务端 Policy.infer
  透传给 Pi0RTC 引导去噪）：
  - ``prev_action_chunk``：本地动作流队列的未来轨迹快照（pad 到 chunk_size，
    Forge 单位换算回服务端毫米）；无队列时以当前 state 重复占位（state_hold）；
  - ``inference_delay``：自适应——首请求用 rtc_initial_delay_steps，之后取 RTT
    中位数 × rtc_control_rate 并钳位到 [0, chunk_size]；
  - ``execute_horizon``：引导前缀长度（≤ chunk_size）。
- **动作流缓冲**（StreamActionBuffer）：新块按已流逝控制步数裁剪（上限
  rtc_latency_k）后与队列剩余轨迹线性混合（关节淡化、夹爪取新值）；消费侧
  队列空则 hold 上一步动作（rtc_stale_timeout_s 内），首个动作块等待
  rtc_startup_timeout_s（覆盖服务端 JIT 编译）。
- **故障语义**：连续 3 次推理失败 → fault（generate_action 抛错）；stop 后
  generate_action 直接抛错；reset 清空缓冲并触发一次新观测请求。

单位契约（contract.py）：缓冲内为 Forge 单位（夹爪米）；组包 prev_action_chunk
时夹爪 米→毫米，拆包服务端动作块时 毫米→米。
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from kai0_inference.inference import contract
from kai0_inference.inference.policies.base import Kai0PolicyAdapter
from kai0_inference.inference.policies.kai0 import payloads
from kai0_inference.inference.policies.kai0.stream_buffer import StreamActionBuffer
from kai0_inference.inference.transport.base import AsyncPersistentTransport
from kai0_inference.inference.transport.ws_async import WsAsyncTransport

logger = logging.getLogger(__name__)

# 参数默认值为真机定型值：
# rtc_control_rate=20 / chunk_size=50 / execute_horizon=18 / latency_k=8 /
# min_smooth_steps=8 / inference_hz=2 / initial_delay_steps=4 / delay_window=8 /
# startup_timeout_s=120 / stale_timeout_s=1.0。
# 注意：rtc_control_rate 语义为控制回读频率（Hz），须与节点 tick 频率一致
# （仿真 50Hz tick 时配置覆盖为 50，见 config/inference/kai0_rtc.yaml）。


class Kai0RtcAdapter(Kai0PolicyAdapter):
    """RTC 异步推理适配器：后台线程常驻连接，主线程消费动作流缓冲。"""

    def __init__(
        self,
        *,
        server_url: str | None = None,
        transport_factory: Callable[[], AsyncPersistentTransport] | None = None,
        instruction: str = "",
        timeout_s: float = 60.0,
        rtc_control_rate: float = 20.0,
        rtc_chunk_size: int = 50,
        rtc_execute_horizon: int = 18,
        rtc_latency_k: int = 8,
        rtc_min_smooth_steps: int = 8,
        rtc_inference_hz: float = 2.0,
        rtc_stale_timeout_s: float = 1.0,
        rtc_startup_timeout_s: float = 120.0,
        rtc_delay_window: int = 8,
        rtc_initial_delay_steps: int = 4,
    ) -> None:
        if transport_factory is None:
            if not server_url or not str(server_url).strip():
                raise ValueError("server_url is required when transport_factory is not given")
            server_url = str(server_url).strip()
            timeout_s = float(timeout_s)
            transport_factory = lambda: WsAsyncTransport(  # noqa: E731
                server_url, timeout_s=timeout_s
            )
        if not callable(transport_factory):
            raise ValueError("transport_factory must be callable")

        self._transport_factory = transport_factory
        self._instruction = str(instruction)
        self._timeout_s = float(timeout_s)
        if not self._timeout_s > 0:
            raise ValueError("timeout_s must be positive")

        self._rtc_control_rate = float(rtc_control_rate)
        self._rtc_chunk_size = int(rtc_chunk_size)
        self._rtc_execute_horizon = int(rtc_execute_horizon)
        self._rtc_latency_k = int(rtc_latency_k)
        self._rtc_min_smooth_steps = int(rtc_min_smooth_steps)
        self._rtc_inference_hz = float(rtc_inference_hz)
        self._rtc_stale_timeout_s = float(rtc_stale_timeout_s)
        self._rtc_startup_timeout_s = float(rtc_startup_timeout_s)
        self._rtc_delay_window = int(rtc_delay_window)
        self._rtc_initial_delay_steps = int(rtc_initial_delay_steps)

        if self._rtc_control_rate <= 0 or self._rtc_inference_hz <= 0:
            raise ValueError("RTC control and inference rates must be positive.")
        if self._rtc_chunk_size <= 0 or self._rtc_execute_horizon <= 0:
            raise ValueError("RTC chunk size and execute horizon must be positive.")
        if self._rtc_execute_horizon > self._rtc_chunk_size:
            raise ValueError("rtc_execute_horizon cannot exceed rtc_chunk_size.")
        if self._rtc_latency_k < 0 or self._rtc_min_smooth_steps <= 0:
            raise ValueError("RTC latency_k must be non-negative and smoothing steps positive.")
        if self._rtc_stale_timeout_s <= 0 or self._rtc_startup_timeout_s <= 0:
            raise ValueError("RTC stale/startup timeouts must be positive.")
        if self._rtc_delay_window < 1:
            raise ValueError("rtc_delay_window must be at least 1.")
        if not 0 <= self._rtc_initial_delay_steps <= self._rtc_chunk_size:
            raise ValueError("rtc_initial_delay_steps must be between 0 and rtc_chunk_size.")

        # 动作流缓冲（Forge 单位）与线程间握手状态。
        self._stream_buffer = StreamActionBuffer(state_dim=contract.ACTION_DIM)
        self._observation_condition = threading.Condition()
        self._latest_payload: dict[str, Any] | None = None
        self._latest_action_plan: np.ndarray | None = None
        self._observation_generation = 0
        self._last_inferred_observation_generation = 0
        self._latest_observation_control_step = 0
        self._control_step_counter = 0
        self._async_observation_requested = True
        self._async_stop_event = threading.Event()
        self._async_fault: BaseException | None = None
        self._async_consecutive_failures = 0
        self._rtc_delay_steps = self._rtc_initial_delay_steps
        self._rtc_rtt_seconds: deque[float] = deque(maxlen=self._rtc_delay_window)
        self._request_count = 0
        self._async_last_action: np.ndarray | None = None
        self._async_buffer_empty_since: float | None = None
        self._last_successful_inference_time: float | None = None
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_main_task: asyncio.Task[Any] | None = None
        self._inference_thread: threading.Thread | None = None

        logger.info(
            "[RTC] rtc_async=enabled control_rate=%.4gHz inference_rate=%.4gHz "
            "chunk_size=%d execute_horizon=%d latency_k=%d min_smooth_steps=%d",
            self._rtc_control_rate,
            self._rtc_inference_hz,
            self._rtc_chunk_size,
            self._rtc_execute_horizon,
            self._rtc_latency_k,
            self._rtc_min_smooth_steps,
        )
        self._inference_thread = threading.Thread(
            target=self._inference_thread_main,
            name="Kai0RtcInferenceThread",
            daemon=True,
        )
        self._inference_thread.start()

    # —— Kai0PolicyAdapter 协议 ——

    @property
    def required_image_keys(self) -> frozenset[str]:
        """服务端需要的三路相机观测键（契约 contract.CAMERA_ALIASES）。"""
        return frozenset(f"observation.images.{alias}" for alias in contract.CAMERA_ALIASES)

    @property
    def instruction(self) -> str:
        return self._instruction

    @instruction.setter
    def instruction(self, value: str) -> None:
        # 生命周期 set_instruction（forge_policy control.py）经 setattr 写入。
        self._instruction = str(value)

    def reset(self) -> None:
        """清空动作流并触发一次新观测请求（对齐真机客户端 notify_control_resumed 语义）。

        注意：若推理线程已 fault 退出，本方法不会重启线程。
        """
        self._stream_buffer.clear()
        self._async_last_action = None
        self._async_buffer_empty_since = None
        self._last_successful_inference_time = time.monotonic()
        self._request_async_observation()

    def is_observation_needed(self) -> bool:
        """RTC 语义：仅当推理线程请求新观测时为 True。"""
        with self._observation_condition:
            return self._async_observation_requested

    def generate_action(
        self,
        observation: dict[str, Any],
        alias_for_cameras: list[str] | None = None,
    ) -> np.ndarray:
        """发布观测（非空时）并从动作流缓冲弹出一步动作（Forge 单位）。

        队列空则 hold 上一步动作（rtc_stale_timeout_s 内）；首个动作块等待
        rtc_startup_timeout_s。stop 后或推理故障时抛 RuntimeError。
        """
        if observation:
            self._publish_async_observation(observation)
        if self._async_stop_event.is_set():
            raise RuntimeError("RTC adapter is stopped")
        if self._async_fault is not None:
            raise RuntimeError("RTC async inference is unavailable") from self._async_fault

        is_startup = self._async_last_action is None
        if (
            not is_startup
            and self._last_successful_inference_time is not None
            and time.monotonic() - self._last_successful_inference_time
            >= self._rtc_stale_timeout_s
        ):
            raise RuntimeError(
                "RTC inference results are stale for "
                f"{time.monotonic() - self._last_successful_inference_time:.2f}s."
            )

        wait_timeout_s = self._rtc_startup_timeout_s if is_startup else 0.0
        action, _is_new_generation = self._stream_buffer.pop_next_action(
            wait_timeout_s=wait_timeout_s,
        )
        if action is not None:
            self._async_last_action = action.copy()
            self._async_buffer_empty_since = None
            with self._observation_condition:
                self._control_step_counter += 1
            return action

        if self._async_fault is not None:
            raise RuntimeError("RTC async inference is unavailable") from self._async_fault

        now = time.monotonic()
        if self._async_buffer_empty_since is None:
            self._async_buffer_empty_since = now
            logger.info("[RTC] buffer underrun; holding the last action.")
        stale_s = now - self._async_buffer_empty_since
        if stale_s >= self._rtc_stale_timeout_s:
            raise RuntimeError(f"RTC action buffer has been empty for {stale_s:.2f}s.")
        if self._async_last_action is None:
            raise TimeoutError(
                "RTC first action chunk was not ready within "
                f"{self._rtc_startup_timeout_s:.1f}s."
            )
        with self._observation_condition:
            self._control_step_counter += 1
        return self._async_last_action.copy()

    def stop(self) -> None:
        """停止后台推理线程并关闭常驻连接（幂等）。"""
        if self._async_stop_event.is_set():
            return
        self._async_stop_event.set()
        with self._observation_condition:
            self._observation_condition.notify_all()
        self._stream_buffer.close()

        loop = self._async_loop
        main_task = self._async_main_task
        if loop is not None and loop.is_running() and main_task is not None:
            loop.call_soon_threadsafe(main_task.cancel)

        thread = self._inference_thread
        if thread is not None and thread.is_alive() and threading.current_thread() is not thread:
            thread.join(timeout=2.0)
            if thread.is_alive():
                logger.warning("[RTC] inference thread did not stop in time.")

    # —— 观测发布/请求握手（主线程 ↔ 推理线程） ——

    def _publish_async_observation(self, observation: dict[str, Any]) -> None:
        """主线程：组包基础 payload 并快照动作流轨迹，交给推理线程。"""
        base_payload = payloads.build_payload(observation, self._instruction)
        buffered_plan = self._stream_buffer.snapshot_future(self._rtc_chunk_size)
        with self._observation_condition:
            self._latest_payload = base_payload
            self._latest_action_plan = buffered_plan
            self._observation_generation += 1
            self._latest_observation_control_step = self._control_step_counter
            self._async_observation_requested = False
            self._observation_condition.notify_all()

    def _request_async_observation(self) -> None:
        """推理线程：请求主线程发布一帧新观测。"""
        with self._observation_condition:
            self._async_observation_requested = True

    def _wait_for_async_observation(
        self,
        previous_generation: int,
    ) -> tuple[dict[str, Any], int, int, np.ndarray | None] | None:
        with self._observation_condition:
            while (
                self._observation_generation <= previous_generation
                and not self._async_stop_event.is_set()
            ):
                self._observation_condition.wait(timeout=0.1)

            if self._async_stop_event.is_set() or self._latest_payload is None:
                return None

            # 浅拷贝即可：payload 数组仅被 packb 读取，不会被修改。
            snapshot_payload = dict(self._latest_payload)
            action_plan = (
                None
                if self._latest_action_plan is None
                else self._latest_action_plan.copy()
            )
            self._async_observation_requested = False
            return (
                snapshot_payload,
                self._observation_generation,
                self._latest_observation_control_step,
                action_plan,
            )

    # —— 后台推理线程 ——

    def _inference_thread_main(self) -> None:
        try:
            asyncio.run(self._async_inference_loop())
        except BaseException as exc:
            if not self._async_stop_event.is_set():
                # 正常 stop 由主线程 cancel 主任务触发，不视为故障。
                self._async_fault = exc
                self._stream_buffer.close()
                logger.error("[RTC] inference thread stopped: %s", exc)

    async def _async_inference_loop(self) -> None:
        loop = asyncio.get_running_loop()
        self._async_loop = loop
        self._async_main_task = asyncio.current_task()
        request_period_s = 1.0 / self._rtc_inference_hz

        while not self._async_stop_event.is_set():
            transport: AsyncPersistentTransport | None = None
            try:
                transport = self._transport_factory()
                await transport.connect()

                while not self._async_stop_event.is_set():
                    await self._run_inference_cycle(transport, request_period_s)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self._async_stop_event.is_set():
                    break
                self._async_consecutive_failures += 1
                logger.warning(
                    "[RTC] inference error (%d/3): %s",
                    self._async_consecutive_failures,
                    exc,
                )
                if self._async_consecutive_failures >= 3:
                    self._async_fault = RuntimeError(
                        "RTC inference failed three consecutive times."
                    )
                    self._stream_buffer.close()
                    break
                await asyncio.sleep(0.25)
            finally:
                if transport is not None:
                    try:
                        await transport.close()
                    except Exception:
                        pass

        self._async_loop = None
        self._async_main_task = None

    async def _run_inference_cycle(
        self,
        transport: AsyncPersistentTransport,
        request_period_s: float,
    ) -> None:
        cycle_start = time.perf_counter()
        self._request_async_observation()
        observation_result = await asyncio.to_thread(
            self._wait_for_async_observation,
            self._last_inferred_observation_generation,
        )
        if observation_result is None:
            return
        (
            base_payload,
            generation,
            observation_control_step,
            action_plan,
        ) = observation_result
        self._last_inferred_observation_generation = generation

        payload = dict(base_payload)
        state = np.asarray(payload["state"], dtype=np.float32).reshape(-1)
        if state.shape != (contract.ACTION_DIM,) or not np.all(np.isfinite(state)):
            raise ValueError(
                f"RTC state must be finite [{contract.ACTION_DIM}], got {state.shape}."
            )
        if action_plan is None:
            # 尚无执行计划：以当前 state 重复占位（state_hold），服务端以此为引导前缀。
            prev_chunk_for_request = np.repeat(
                state[None, :], self._rtc_chunk_size, axis=0
            )
            prev_source = "state_hold"
        else:
            # 缓冲内为 Forge 单位；服务端契约 prev_action_chunk 为服务端物理空间。
            prev_chunk_for_request = contract.grippers_to_server_units(
                action_plan
            ).astype(np.float32)
            prev_source = "buffer"
        if (
            prev_chunk_for_request.shape
            != (self._rtc_chunk_size, contract.ACTION_DIM)
            or not np.all(np.isfinite(prev_chunk_for_request))
        ):
            raise ValueError(
                "RTC previous execution plan must be finite "
                f"[{self._rtc_chunk_size}, {contract.ACTION_DIM}], got "
                f"{prev_chunk_for_request.shape}."
            )
        payload["prev_action_chunk"] = prev_chunk_for_request
        payload["inference_delay"] = int(self._rtc_delay_steps)
        payload["execute_horizon"] = int(self._rtc_execute_horizon)

        request_start = time.perf_counter()
        result = await transport.request_actions(payload)
        round_trip_s = time.perf_counter() - request_start

        actions = payloads.parse_actions(result)
        if actions.shape[0] != self._rtc_chunk_size:
            raise ValueError(
                f"RTC server returned horizon {actions.shape[0]}, "
                f"expected {self._rtc_chunk_size}."
            )
        # 服务端夹爪毫米 → Forge 米（契约见 contract.py）。
        execution_chunk = contract.grippers_from_server_units(actions)

        with self._observation_condition:
            elapsed_control_steps = max(
                0,
                self._control_step_counter - observation_control_step,
            )
        trim_steps = self._stream_buffer.integrate_new_chunk(
            execution_chunk,
            max_trim_steps=self._rtc_latency_k,
            min_blend_steps=self._rtc_min_smooth_steps,
            elapsed_control_steps=elapsed_control_steps,
        )

        # 首个请求含服务端 JIT 编译，RTT 不参与统计（与真机客户端一致）。
        warmup_rtt_excluded = self._request_count == 0
        if not warmup_rtt_excluded:
            self._rtc_rtt_seconds.append(round_trip_s)
            median_rtt_s = float(np.median(np.asarray(self._rtc_rtt_seconds)))
            self._rtc_delay_steps = max(
                0,
                min(
                    self._rtc_chunk_size,
                    int(np.ceil(median_rtt_s * self._rtc_control_rate)),
                ),
            )
        self._request_count += 1
        self._async_consecutive_failures = 0
        self._async_fault = None
        self._last_successful_inference_time = time.monotonic()

        server_timing = result.get("server_timing", {})
        infer_ms = server_timing.get("infer_ms")
        logger.info(
            "[RTC] request=%d round_trip_ms=%.1f server_infer_ms=%s inference_delay=%d "
            "rtt_sample=%s prev_source=%s elapsed_steps=%d trim_steps=%d buffer_remaining=%d",
            self._request_count,
            round_trip_s * 1000,
            f"{float(infer_ms):.1f}" if isinstance(infer_ms, (int, float)) else "n/a",
            self._rtc_delay_steps,
            "warmup_excluded" if warmup_rtt_excluded else "accepted",
            prev_source,
            elapsed_control_steps,
            trim_steps,
            self._stream_buffer.remaining(),
        )

        elapsed_s = time.perf_counter() - cycle_start
        await asyncio.sleep(max(0.0, request_period_s - elapsed_s))
