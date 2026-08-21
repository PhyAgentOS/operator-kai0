# χ₀

<div id="top" align="center">

[![Blog Page](https://img.shields.io/badge/Blog_Page-green)](https://mmlab.hk/research/kai0)
[![arXiv](https://img.shields.io/badge/arXiv-2602.09021-b31b1b)](https://arxiv.org/abs/2602.09021)
[![Kai0 Data](https://img.shields.io/badge/huggingface-Kai0_Data-orange?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/OpenDriveLab-org/Kai0)
[![Kai0 Model](https://img.shields.io/badge/huggingface-Kai0_Model-orange?logo=huggingface&logoColor=white)](https://huggingface.co/OpenDriveLab-org/Kai0)
[![ModelScope Data](https://img.shields.io/badge/ModelScope-Kai0_Data-purple)](https://www.modelscope.cn/datasets/OpenDriveLab/Kai0)
[![ModelScope Model](https://img.shields.io/badge/ModelScope-Kai0_Model-purple)](https://www.modelscope.cn/models/OpenDriveLab/Kai0)

<!-- [![Repo](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/OpenDriveLab/kai0) -->


</div>

χ₀ (**kai0**) is a resource-efficient framework for achieving production-level robustness in robotic manipulation by taming distributional inconsistencies.
<!-- This repository is built on top of [openpi](https://github.com/Physical-Intelligence/openpi), the open-source models and packages for robotics published by the [Physical Intelligence team](https://www.physicalintelligence.company/). -->

χ₀ addresses the systematic distributional shift among the human demonstration distribution ($P_\text{train}$), the inductive bias learned by the policy ($Q_\text{model}$), and the test-time execution distribution ($P_\text{test}$) through three technical modules:

- **[Model Arithmetic](#model-arithmetic)**: A weight-space merging strategy that combines models trained on different data subsets, efficiently capturing diverse knowledge without architectural complexity. **[Released]**
- **[Stage Advantage](#stage-advantage)**: A stage-aware advantage estimator that provides stable, dense progress signals for policy training. **[Released]**
- **[Train-Deploy Alignment](#train-deploy-alignment)**: Bridges the distribution gap via spatio-temporal augmentation, heuristic DAgger corrections, and temporal chunk-wise smoothing. **[Released]**

χ₀ enables two sets of dual-arm robots to collaboratively orchestrate long-horizon garment manipulation — flattening, folding, and hanging — surpassing the state-of-the-art $\pi_{0.5}$ baseline by approximately 250% in success rate, with `only 20 hours of data and 8 A100 GPUs`.

<!-- [[Paper]](https://github.com/OpenDriveLab/kai0) [[Blog]](https://mmlab.hk/research/kai0) -->

https://github.com/user-attachments/assets/3f5f0c48-ff3f-4b9b-985b-59ad0b2ea97c

## End-to-End Workflow (Based on `train.markdown`)

This section is a practical, reproducible pipeline for local training and deployment with this repo. It summarizes the commands used in `train.markdown` and explains when to use each step.

### 0. Environment and basic checks

```bash
cd <repo_root>
source .venv/bin/activate
```

Recommended quick checks before training:

- Verify dataset path exists (for example `<path_to_dataset>/fold_box_subtask_8_12`).
- Verify target config exists in `src/openpi/training/config.py` (for example `fold_box_normal`).

### 1. Merge raw LeRobot datasets (optional, when creating a new merged dataset)

Use this when your data is split across multiple recording folders and you want one merged training dataset:

```bash
uv run python scripts/merge_lerobot.py \
  --src_paths <path_to_recordings>/fold_box_subtask_8_12 \
  <path_to_recordings>/fold_box_subtask_8_12 \
  <path_to_recordings>/fold_box_subtask_8_12 \
  <path_to_recordings>/fold_box_subtask_8_12 \
  --tgt_path <path_to_dataset>/fold_box_subtask_8_12
```

### 2. Compute normalization stats (required before training)

```bash
uv run scripts/compute_norm_states_fast.py --config-name fold_box_normal
```

This generates/updates normalization stats used by the training data pipeline.

### 3. Start training (main training entry)

Example from `train.markdown`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py fold_box_normal \
  --exp_name=fold_box_normal \
  --batch-size=256 \
  --fsdp-devices=8 \
  --num_train_steps=10000 \
  --save-interval=5000 \
  --keep-period=5000
```


### 4. Train with validation loss loop (optional)

Use `scripts/train_val_loss.py` when you want periodic validation metrics in the same run:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
uv run python scripts/train_val_loss.py fold_box_normal \
  --exp-name=fold_box_normal \
  --num-train-steps=20000 \
  --save-interval=5000 \
  --keep-period=5000 \
  --num-workers=8 \
  --batch-size=256 \
  --fsdp-devices=4
```

### 5. Launch policy server for inference/deployment

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
PORT=8080 \
POLICY_CONFIG=fold_box_normal \
POLICY_DIR=checkpoints/fold_box_normal/fold_box_normal/39999 \
bash policy_server.sh
```

Notes:

- `POLICY_DIR` should point to a checkpoint step directory containing `params/`.
- `policy_server.sh` already handles `<output>` vs `<output>/0` style directories.

### 6. Sync offline W&B runs to cloud (optional)

If you trained in offline mode and want to upload later:

```bash
export WANDB_API_KEY="<your_wandb_api_key>"
wandb sync <path_to_wandb_runs>/offline-run-<run_id>
```

Do not commit API keys to git. Keep secrets in environment variables only.

### 7. Most important config knobs for behavior

In `src/openpi/training/config.py` (for each training entry):

- `repo_id`: dataset path.
- `weight_loader`: checkpoint initialization path.
- `use_delta_joint_actions`:
  - `False`: train with absolute actions.
  - `True`: convert to delta action space for selected joint dims.
- For pi05 state ablation:
  - `data.mask_state=True` to zero continuous state input.
  - `model.discrete_state_input=False` to disable state-in-prompt.

For full setup instructions (IPC environment, CAN, ROS/ROS2, platform-specific details), see [`train_deploy_alignment/README.md`](train_deploy_alignment/README.md).

## License and Citation

All assets and code in this repository are under the Apache 2.0 license unless specified otherwise. The data and checkpoint are under CC BY-NC-SA 4.0. Other modules (including PaliGemma) inherit their own distribution licenses. If you find χ₀ useful in your research, please consider citing:

```bibtex
@article{sima2026kai0,
  title={$\chi_{0}$: Resource-Aware Robust Manipulation via Taming Distributional Inconsistencies},
  author={Yu, Checheng and Sima, Chonghao and Jiang, Gangcheng and Zhang, Hai and Mai, Haoguang and Li, Hongyang and Wang, Huijie and Chen, Jin and Wu, Kaiyang and Chen, Li and Zhao, Lirui and Shi, Modi and Luo, Ping and Bu, Qingwen and Peng, Shijia and Li, Tianyu and Yuan, Yibo},
  journal={arXiv preprint arXiv:2602.09021},
  year={2026}
}
```