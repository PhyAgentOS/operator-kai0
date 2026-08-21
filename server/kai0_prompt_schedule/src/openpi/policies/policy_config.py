import logging
import json
import os
import pathlib
from typing import Any, Literal

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
import openpi.shared.normalize as _normalize
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms

PromptMode = Literal["auto", "fixed", "task"]


def _load_local_task_prompts(repo_id: str) -> dict[int, str] | None:
    tasks_path = pathlib.Path(repo_id) / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        return None

    prompts: dict[int, str] = {}
    with tasks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompts[int(row["task_index"])] = str(row["task"])
    logging.info("Loaded local task prompts from %s", tasks_path)
    return prompts


def _load_task_prompts(data_config: _config.DataConfig, *, require_prompt_from_task: bool = True) -> dict[int, str] | None:
    if data_config.repo_id in (None, "fake"):
        return None
    if require_prompt_from_task and not data_config.prompt_from_task:
        return None

    if (task_prompts := _load_local_task_prompts(str(data_config.repo_id))) is not None:
        return task_prompts

    try:
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

        return lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id).tasks
    except Exception as exc:  # noqa: BLE001 - inference can still use explicit prompt/default_prompt.
        logging.warning("Could not load LeRobot task prompts from %s: %s", data_config.repo_id, exc)
        return None


def _insert_prompt_debug_transform(
    model_transforms: transforms.Group,
) -> list[transforms.DataTransformFn]:
    inputs: list[transforms.DataTransformFn] = []
    for transform in model_transforms.inputs:
        if isinstance(transform, transforms.TokenizePrompt | transforms.TokenizeFASTInputs):
            inputs.append(transforms.DebugPrintPrompt())
        inputs.append(transform)
    return inputs


def _prompt_selection_transforms(
    train_config: _config.TrainConfig,
    data_config: _config.DataConfig,
    *,
    default_prompt: str | None,
    prompt_mode: PromptMode,
) -> list[transforms.DataTransformFn]:
    if prompt_mode == "task":
        task_prompts = _load_task_prompts(data_config, require_prompt_from_task=False)
        if task_prompts is None:
            raise ValueError(
                "prompt_mode='task' requires a real LeRobot dataset with meta/tasks.jsonl so task_index/subtask can be mapped."
            )
        return [
            transforms.InjectPromptFromTaskIndex(task_prompts),
            transforms.InjectDefaultPrompt(default_prompt),
        ]

    configured_default_prompt = default_prompt or getattr(train_config.data, "default_prompt", None)
    if prompt_mode == "fixed":
        if configured_default_prompt is None:
            raise ValueError("prompt_mode='fixed' requires a configured default_prompt.")
        return [transforms.ForceDefaultPrompt(configured_default_prompt)]

    task_prompts = _load_task_prompts(data_config)
    if task_prompts is not None:
        return [
            transforms.InjectPromptFromTaskIndex(task_prompts),
            transforms.InjectDefaultPrompt(default_prompt),
        ]

    if configured_default_prompt is not None:
        return [transforms.ForceDefaultPrompt(configured_default_prompt)]

    return [transforms.InjectDefaultPrompt(default_prompt)]


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
    prompt_mode: PromptMode = "auto",
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        try:
            norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
        except FileNotFoundError:
            local_norm_stats_dir = checkpoint_dir.parent
            if not (local_norm_stats_dir / "norm_stats.json").exists():
                raise
            norm_stats = _normalize.load(local_norm_stats_dir)
            logging.info("Loaded norm stats from %s", local_norm_stats_dir)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    return _policy.Policy(
        model,
        transforms=[
            transforms.DebugPrintClientRequest(),
            *repack_transforms.inputs,
            *_prompt_selection_transforms(
                train_config,
                data_config,
                default_prompt=default_prompt,
                prompt_mode=prompt_mode,
            ),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *_insert_prompt_debug_transform(data_config.model_transforms),
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
