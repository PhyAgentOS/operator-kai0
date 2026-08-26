#!/usr/bin/env python3
"""Run one inference step locally (no Dora) for smoke testing."""

from __future__ import annotations

import json
import sys

import numpy as np

from kai0_inference.inference.config import load_config
from kai0_inference.inference.policies.registry import create_policy_adapter


def _build_observation(
    config,
    camera_aliases: list[str] | None = None,
    *,
    state_dim: int | None,
    height: int,
    width: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    state_dim = state_dim or len(config.state_joint_order)
    observation: dict[str, np.ndarray] = {
        "observation.state": rng.normal(size=state_dim).astype(np.float32),
    }
    aliases = config.alias_for_cameras if camera_aliases is None else camera_aliases
    for alias in aliases:
        observation[f"observation.images.{alias}"] = rng.integers(
            0,
            255,
            size=(height, width, 3),
            dtype=np.uint8,
        )
    return observation


def run_infer_once_smoke(
    config,
    *,
    state_dim: int | None = None,
    height: int = 224,
    width: int = 224,
    seed: int = 0,
) -> dict:
    """单步冒烟核心：返回 payload 字典（action_dim 等）；session 就绪门复用。"""
    policy = create_policy_adapter(config.runtime_policy_config())
    policy.reset()
    image_inputs = config.image_inputs_for(policy.required_image_keys)
    camera_aliases = list(image_inputs.values())
    try:
        action = policy.generate_action(
            _build_observation(
                config,
                camera_aliases,
                state_dim=state_dim,
                height=height,
                width=width,
                seed=seed,
            ),
            camera_aliases,
        )
        if action is None:
            raise RuntimeError("policy returned no action")
        action = config.contract.validate_action(action)
        return {
            "policy_type": config.policy.get("type"),
            "action": action.tolist(),
            "action_dim": int(action.shape[0]),
        }
    finally:
        stop = getattr(policy, "stop", None)
        if callable(stop):
            stop()


def run_infer_once(args) -> int:
    """执行单步 smoke；args 需含 config，以及可选 state_dim/height/width/seed。"""
    config = load_config(config_path=args.config)
    payload = run_infer_once_smoke(
        config,
        state_dim=args.state_dim,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )
    print(json.dumps(payload, indent=2))
    return 0


def main() -> int:
    """兼容旧入口：转发到统一 CLI 的 infer-once 子命令。"""
    from kai0_inference.cli import main_infer_once

    return main_infer_once()


if __name__ == "__main__":
    sys.exit(main())
