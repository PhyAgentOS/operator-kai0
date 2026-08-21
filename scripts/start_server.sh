#!/usr/bin/env bash
# kai0 推理服务端启动脚本（自包含：服务端代码位于本仓库 server/kai0_prompt_schedule/，
# 来源与同步说明见 server/SOURCE.md）。
# 与 server/ 快照配套：venv 可用 SERVER_VENV 覆盖、--ckpt 相对路径按仓库根解析；
# 内置动作空间安全检查。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_DIR="${ROOT_DIR}/server/kai0_prompt_schedule"
DEFAULT_CKPT=""
CKPT_ARG=""
PORT="${PORT:-8000}"
CONFIG_BASE="fold_box_normal_subtask_8_12"
POLICY_CONFIG="${POLICY_CONFIG:-}"
LORA_BASE="fold_box_normal_subtask_8_12_lora_packaging"
USE_RTC=0
USE_DELTA=0
USE_LORA=0

usage() {
  echo "用法: $0 [--ckpt <step目录>] [--port <端口>] [--rtc] [--delta] [--lora] [--policy-config <名称>]"
  echo "step 目录必须包含 params/，其父目录必须包含 norm_stats.json。"
  echo "--rtc   使用 RTC 变体，供客户端 policy.rtc_async=true；与非 RTC 版共用同一 checkpoint。"
  echo "--delta 用于以相对关节角（use_delta_joint_actions=True）训练的 checkpoint。"
  echo "--lora  用于 LoRA 微调的 checkpoint（params 里含 lora_a/lora_b），基名改为 ${LORA_BASE}。"
  echo "开关可组合，对应配置名 ${CONFIG_BASE}[_delta][_rtc] 或 ${LORA_BASE}[_rtc]。"
  echo "--policy-config 显式指定配置名，会覆盖上面几个开关。"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)
      [[ $# -ge 2 ]] || { echo "[ERROR] --ckpt 缺少路径参数" >&2; exit 2; }
      CKPT_ARG="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || { echo "[ERROR] --port 缺少端口参数" >&2; exit 2; }
      PORT="$2"
      shift 2
      ;;
    --rtc)
      USE_RTC=1
      shift
      ;;
    --delta)
      USE_DELTA=1
      shift
      ;;
    --lora)
      USE_LORA=1
      shift
      ;;
    --policy-config)
      [[ $# -ge 2 ]] || { echo "[ERROR] --policy-config 缺少名称参数" >&2; exit 2; }
      POLICY_CONFIG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] 未知参数: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${POLICY_CONFIG}" ]]; then
  if [[ "${USE_LORA}" -eq 1 ]]; then
    # 目前只有绝对关节角的 LoRA 配置，没有 delta 版；不拦住会拼出不存在的配置名。
    if [[ "${USE_DELTA}" -eq 1 ]]; then
      echo "[ERROR] --lora 与 --delta 不能同时使用：尚未提供 delta 版的 LoRA 配置。" >&2
      echo "[ERROR] 如确有 delta 训练的 LoRA checkpoint，请先在 config.py 中新增对应配置，再用 --policy-config 指定。" >&2
      exit 2
    fi
    POLICY_CONFIG="${LORA_BASE}"
  else
    POLICY_CONFIG="${CONFIG_BASE}"
    [[ "${USE_DELTA}" -eq 1 ]] && POLICY_CONFIG="${POLICY_CONFIG}_delta"
  fi
  [[ "${USE_RTC}" -eq 1 ]] && POLICY_CONFIG="${POLICY_CONFIG}_rtc"
fi

# --ckpt 相对路径按仓库根解析（任意工作目录下运行结果一致）。
if [[ -n "${CKPT_ARG}" && "${CKPT_ARG}" != /* ]]; then
  CKPT_ARG="${ROOT_DIR}/${CKPT_ARG}"
fi

POLICY_DIR="${CKPT_ARG:-${POLICY_DIR:-${DEFAULT_CKPT}}}"
if [[ -z "${POLICY_DIR}" ]]; then
  echo "[ERROR] 请用 --ckpt 指定权重 step 目录（直接包含 params/；放置约定见 ckpt/README.md）。" >&2
  exit 1
fi
if [[ ! -d "${POLICY_DIR}/params" ]]; then
  echo "[ERROR] checkpoint 无效: ${POLICY_DIR}" >&2
  echo "[ERROR] 期望存在: ${POLICY_DIR}/params/" >&2
  exit 1
fi

POLICY_DIR="$(cd "${POLICY_DIR}" && pwd)"
# LIBERO 官方 checkpoint 布局（pi05_libero 等）：<dir>/params/ +
# <dir>/assets/<repo_id>/norm_stats.json（create_trained_policy 原生支持，
# asset_id 由配置的 DataConfig 解析）。双臂本地约定（<step>/params/ +
# 父目录 norm_stats.json）继续走下方原校验。
LIBERO_LAYOUT=0
if [[ "${POLICY_CONFIG}" == *libero* ]]; then
  LIBERO_LAYOUT=1
  if ! find "${POLICY_DIR}/assets" -name "norm_stats.json" -print -quit 2>/dev/null | grep -q .; then
    echo "[ERROR] LIBERO checkpoint 缺少归一化统计：" >&2
    echo "[ERROR] 期望存在: ${POLICY_DIR}/assets/<repo_id>/norm_stats.json" >&2
    exit 1
  fi
  echo "[INFO] LIBERO checkpoint 布局校验通过（assets/<repo_id>/norm_stats.json）"
elif [[ ! -f "$(dirname "${POLICY_DIR}")/norm_stats.json" ]]; then
  echo "[ERROR] 缺少归一化统计: $(dirname "${POLICY_DIR}")/norm_stats.json" >&2
  exit 1
fi

# 运行环境：server/setup.sh 一键安装的 venv（可用 SERVER_VENV 指向其它等价环境）。
SERVER_PYTHON="${SERVER_VENV:-${SERVER_DIR}/.venv}/bin/python"
if [[ ! -x "${SERVER_PYTHON}" ]]; then
  echo "[ERROR] 服务端环境不存在: ${SERVER_PYTHON}" >&2
  echo "[ERROR] 请先执行 bash server/setup.sh（有网环境一键安装）。" >&2
  exit 1
fi

# 配错 delta/绝对会让客户端把接近 0 的增量当作绝对目标下发，双臂直冲零位。
# norm_stats 的动作均值可以区分两者：delta 训练下臂关节均值塌到 0。
# 该检查只针对 14 维双臂 fold_box 配置；LIBERO 配置（7 维 delta 域）跳过。
if [[ "${LIBERO_LAYOUT}" -eq 0 ]]; then
NORM_STATS_PATH="$(dirname "${POLICY_DIR}")/norm_stats.json"
if ! "${SERVER_PYTHON}" - "${NORM_STATS_PATH}" "${POLICY_CONFIG}" <<'PY'
import json, sys

norm_stats_path, policy_config = sys.argv[1], sys.argv[2]
stats = json.load(open(norm_stats_path))
stats = stats.get("norm_stats", stats)
means = stats["actions"]["mean"]
joint_means = [abs(means[i]) for i in range(14) if i % 7 != 6]

ckpt_is_delta = max(joint_means) < 0.05
config_is_delta = "_delta" in policy_config
if ckpt_is_delta == config_is_delta:
    sys.exit(0)

space = "相对关节角 (delta)" if ckpt_is_delta else "绝对关节角"
expect = "带 _delta 的配置（--delta）" if ckpt_is_delta else "不带 _delta 的配置（去掉 --delta）"
print(f"[ERROR] checkpoint 与配置的动作空间不一致。", file=sys.stderr)
print(f"[ERROR] norm_stats 显示该 checkpoint 是{space}，臂关节动作均值最大 {max(joint_means):.4f}。", file=sys.stderr)
print(f"[ERROR] 当前配置 {policy_config} 需要改用{expect}。", file=sys.stderr)
print(f"[ERROR] 强行启动会让客户端收到错误量纲的动作，机械臂可能高速冲向零位。", file=sys.stderr)
sys.exit(1)
PY
then
  exit 1
fi
fi   # LIBERO_LAYOUT == 0（双臂 delta/绝对安全检查结束）

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${ROOT_DIR}/cache/openpi}"
# JAX 编译缓存落盘：服务端重启后复用已编译的推理路径，冷启动首个请求从数分钟降到数秒。
# 缓存目录随 cache/ 一并 gitignore；如遇缓存兼容问题，删除该目录即可回退重编译。
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${ROOT_DIR}/cache/jax_compilation}"

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] PORT=${PORT}"
if [[ "${LIBERO_LAYOUT}" -eq 1 ]]; then
  PROMPT_MODE="auto"   # LIBERO：指令随请求 payload（prompt_from_task），fixed 会因无默认 prompt 拒绝启动
else
  PROMPT_MODE="fixed"
fi
echo "[INFO] PROMPT_MODE=${PROMPT_MODE}"
echo "[INFO] POLICY_CONFIG=${POLICY_CONFIG}"
echo "[INFO] POLICY_DIR=${POLICY_DIR}"
if [[ "${POLICY_CONFIG}" == *_rtc ]]; then
  echo "[INFO] RTC 模式：客户端需设置 policy.rtc_async=true。"
fi
if [[ "${POLICY_CONFIG}" == *_lora* ]]; then
  echo "[INFO] LoRA 模式：PaliGemma rank 16 / action expert rank 32，checkpoint 必须含 lora_a、lora_b。"
fi
if [[ "${POLICY_CONFIG}" == *_delta* ]]; then
  echo "[INFO] 相对关节角模式：服务端会在输出端还原为绝对角，客户端无需改动。"
fi

cd "${SERVER_DIR}"
exec "${SERVER_PYTHON}" scripts/serve_policy.py \
  --port "${PORT}" \
  --prompt-mode "${PROMPT_MODE}" \
  policy:checkpoint \
  --policy.config="${POLICY_CONFIG}" \
  --policy.dir="${POLICY_DIR}"
