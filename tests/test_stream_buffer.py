"""StreamActionBuffer 单测：trim / blend（关节淡化+夹爪取新值）/ snapshot pad / 生命周期。

本文件只测缓冲器纯逻辑（确定性），与推理线程的集成见 test_kai0_rtc_adapter.py。
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from kai0_inference.inference.policies.kai0.stream_buffer import StreamActionBuffer

JOINT = [i for i in range(14) if i % 7 != 6]
GRIPPER = [i for i in range(14) if i % 7 == 6]
# 线性淡化权重：linspace(1, 0, 5)
BLEND5_WEIGHTS = [1.0, 0.75, 0.5, 0.25, 0.0]


def _chunk(step_start: int, n: int, *, joint_base: float, gripper: float) -> np.ndarray:
    """关节值 = joint_base + 步号×0.1（可区分各行），夹爪 = 常量（区分关节/夹爪行为）。"""
    rows = np.zeros((n, 14), dtype=np.float64)
    for t in range(n):
        rows[t, JOINT] = joint_base + (step_start + t) * 0.1
        rows[t, GRIPPER] = gripper
    return rows


def _pop_all(buffer: StreamActionBuffer, limit: int = 1000) -> list[np.ndarray]:
    actions = []
    for _ in range(limit):
        action, _ = buffer.pop_next_action()
        if action is None:
            break
        actions.append(action)
    return actions


# —— integrate 基础语义 ——


def test_first_integrate_appends_whole_chunk():
    buffer = StreamActionBuffer()
    chunk = _chunk(0, 50, joint_base=0.0, gripper=0.05)
    trim = buffer.integrate_new_chunk(chunk, max_trim_steps=8, min_blend_steps=5)
    assert trim == 0
    popped = _pop_all(buffer)
    assert len(popped) == 50
    for i, action in enumerate(popped):
        np.testing.assert_allclose(action, chunk[i])


def test_pop_reports_new_generation_only_for_first_action():
    buffer = StreamActionBuffer()
    buffer.integrate_new_chunk(_chunk(0, 3, joint_base=0.0, gripper=0.05), max_trim_steps=8, min_blend_steps=5)
    action, is_new = buffer.pop_next_action()
    assert action is not None and is_new
    action, is_new = buffer.pop_next_action()
    assert action is not None and not is_new
    buffer.integrate_new_chunk(_chunk(3, 3, joint_base=0.0, gripper=0.05), max_trim_steps=8, min_blend_steps=5)
    action, is_new = buffer.pop_next_action()
    assert action is not None and is_new


# —— trim：已流逝步数裁剪（上限 max_trim_steps） ——


def test_trim_by_elapsed_steps_and_blend_overlap():
    buffer = StreamActionBuffer()
    chunk_a = _chunk(0, 50, joint_base=0.0, gripper=0.05)
    buffer.integrate_new_chunk(chunk_a, max_trim_steps=8, min_blend_steps=5)
    popped = _pop_all(buffer, limit=3)  # 已流逝 3 步

    chunk_b = _chunk(0, 50, joint_base=100.0, gripper=0.06)
    trim = buffer.integrate_new_chunk(
        chunk_b, max_trim_steps=8, min_blend_steps=5, elapsed_control_steps=3
    )
    assert trim == 3
    # 旧队列 = A[3:50]，新块 = B[3:50]，重叠 5 步线性淡化（夹爪取新值）
    remaining = _pop_all(buffer)
    assert len(remaining) == 50 - 3
    for i, weight in enumerate(BLEND5_WEIGHTS):
        expected = weight * chunk_a[3 + i] + (1.0 - weight) * chunk_b[3 + i]
        np.testing.assert_allclose(remaining[i][JOINT], expected[JOINT])
        np.testing.assert_allclose(remaining[i][GRIPPER], chunk_b[3 + i][GRIPPER])
    # 重叠段之后为 B 原样
    np.testing.assert_allclose(remaining[5], chunk_b[8])


def test_trim_capped_by_max_trim_steps():
    buffer = StreamActionBuffer()
    buffer.integrate_new_chunk(_chunk(0, 50, joint_base=0.0, gripper=0.05), max_trim_steps=8, min_blend_steps=5)
    _pop_all(buffer, limit=3)
    trim = buffer.integrate_new_chunk(
        _chunk(0, 50, joint_base=100.0, gripper=0.06),
        max_trim_steps=8,
        min_blend_steps=5,
        elapsed_control_steps=20,  # 实际流逝 20 步，裁剪上限 8
    )
    assert trim == 8


def test_trim_defaults_to_steps_since_integrate():
    buffer = StreamActionBuffer()
    buffer.integrate_new_chunk(_chunk(0, 50, joint_base=0.0, gripper=0.05), max_trim_steps=8, min_blend_steps=5)
    _pop_all(buffer, limit=2)  # 未显式传 elapsed：以消费步数为准
    trim = buffer.integrate_new_chunk(
        _chunk(0, 50, joint_base=100.0, gripper=0.06), max_trim_steps=8, min_blend_steps=5
    )
    assert trim == 2


def test_grippers_always_take_newest_chunk_values():
    buffer = StreamActionBuffer()
    chunk_a = _chunk(0, 50, joint_base=0.0, gripper=0.05)
    chunk_b = _chunk(0, 50, joint_base=100.0, gripper=0.09)
    buffer.integrate_new_chunk(chunk_a, max_trim_steps=8, min_blend_steps=5)
    _pop_all(buffer, limit=3)
    buffer.integrate_new_chunk(chunk_b, max_trim_steps=8, min_blend_steps=5, elapsed_control_steps=3)
    remaining = _pop_all(buffer)
    # 重叠段夹爪直接取新块值（不被平均延迟），关节按权重淡化
    for i in range(5):
        assert remaining[i][GRIPPER[0]] == pytest.approx(0.09)
        assert remaining[i][JOINT[0]] == pytest.approx(
            BLEND5_WEIGHTS[i] * chunk_a[3 + i][JOINT[0]] + (1 - BLEND5_WEIGHTS[i]) * chunk_b[3 + i][JOINT[0]]
        )


def test_empty_queue_uses_last_action_as_blend_base():
    buffer = StreamActionBuffer()
    chunk_a = _chunk(0, 10, joint_base=0.0, gripper=0.05)
    buffer.integrate_new_chunk(chunk_a, max_trim_steps=8, min_blend_steps=5)
    _pop_all(buffer)  # 队列耗尽，last_action = A[9]

    chunk_b = _chunk(0, 10, joint_base=100.0, gripper=0.07)
    # 显式 elapsed=0（否则默认取消费步数，会触发裁剪）
    buffer.integrate_new_chunk(
        chunk_b, max_trim_steps=8, min_blend_steps=5, elapsed_control_steps=0
    )
    remaining = _pop_all(buffer)
    assert len(remaining) == 10
    # 混合起点为上一步动作（权重 1.0），保证轨迹连续；夹爪取新值
    np.testing.assert_allclose(remaining[0][JOINT], chunk_a[9][JOINT])
    np.testing.assert_allclose(remaining[0][GRIPPER], chunk_b[0][GRIPPER])
    np.testing.assert_allclose(remaining[4], chunk_b[4])


# —— snapshot_future：未来轨迹快照 + pad ——


def test_snapshot_future_pads_to_horizon():
    buffer = StreamActionBuffer()
    chunk = _chunk(0, 3, joint_base=0.0, gripper=0.05)
    buffer.integrate_new_chunk(chunk, max_trim_steps=8, min_blend_steps=5)
    snapshot = buffer.snapshot_future(50)
    assert snapshot is not None and snapshot.shape == (50, 14)
    np.testing.assert_allclose(snapshot[:3], chunk)
    for i in range(3, 50):  # 不足部分以末行补齐
        np.testing.assert_allclose(snapshot[i], chunk[2])


def test_snapshot_future_empty_queue_uses_last_action():
    buffer = StreamActionBuffer()
    buffer.integrate_new_chunk(_chunk(0, 3, joint_base=0.0, gripper=0.05), max_trim_steps=8, min_blend_steps=5)
    _pop_all(buffer)
    snapshot = buffer.snapshot_future(5)
    assert snapshot is not None and snapshot.shape == (5, 14)
    for i in range(5):
        np.testing.assert_allclose(snapshot[i], _chunk(0, 3, joint_base=0.0, gripper=0.05)[2])


def test_snapshot_future_fresh_buffer_is_none():
    buffer = StreamActionBuffer()
    assert buffer.snapshot_future(50) is None


# —— 消费侧阻塞 / 生命周期 ——


def test_pop_wait_timeout_returns_none_on_empty():
    buffer = StreamActionBuffer()
    start = time.monotonic()
    action, is_new = buffer.pop_next_action(wait_timeout_s=0.05)
    assert action is None and not is_new
    assert time.monotonic() - start < 1.0


def test_clear_resets_queue_and_last_action():
    buffer = StreamActionBuffer()
    buffer.integrate_new_chunk(_chunk(0, 3, joint_base=0.0, gripper=0.05), max_trim_steps=8, min_blend_steps=5)
    _pop_all(buffer, limit=1)
    buffer.clear()
    assert buffer.remaining() == 0
    assert buffer.snapshot_future(50) is None
    assert buffer.pop_next_action()[0] is None


def test_close_blocks_integration_and_unblocks_pops():
    buffer = StreamActionBuffer()
    buffer.close()
    assert buffer.integrate_new_chunk(
        _chunk(0, 3, joint_base=0.0, gripper=0.05), max_trim_steps=8, min_blend_steps=5
    ) == 0
    assert buffer.pop_next_action()[0] is None


def test_dim_mismatch_rejected():
    buffer = StreamActionBuffer()
    with pytest.raises(ValueError, match="dim mismatch"):
        buffer.integrate_new_chunk(
            np.zeros((3, 7)), max_trim_steps=8, min_blend_steps=5
        )
