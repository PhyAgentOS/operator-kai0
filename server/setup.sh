#!/usr/bin/env bash
# kai0 推理服务端环境一键安装（有网环境）。
#
# 在 server/kai0_prompt_schedule/.venv 创建 Python 3.12 环境，按实测验证的
# 「三步装」顺序安装依赖（该顺序绕开 uv workspace/lock 的解析问题，说明见下）：
#   1. lerobot（pyproject [tool.uv.sources] 锁定的 github commit，先装避免二次解析）
#   2. openpi-client（本地 workspace 包，editable）
#   3. openpi 根包（editable；其余 PyPI 依赖在此步统一解析）
# pyproject [tool.uv] override-dependencies（ml-dtypes==0.4.1 / tensorstore==0.1.74，
# 与 jax 0.5.3 实测兼容组合）在 uv pip 模式下不生效，本脚本显式以 --override 复刻。
#
# 网络：默认走官方 PyPI / github；受限环境可用 UV_INDEX_URL 指向镜像
# （如 https://pypi.tuna.tsinghua.edu.cn/simple），github 需可达。
# venv 可用 SERVER_VENV 环境变量覆盖（默认 server/kai0_prompt_schedule/.venv）。
set -euo pipefail

SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/kai0_prompt_schedule" && pwd)"
VENV="${SERVER_VENV:-${SERVER_DIR}/.venv}"
LEROBOT_REV="0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"   # pyproject [tool.uv.sources] 锁定值

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] 未找到 uv。请先安装：https://docs.astral.sh/uv/（curl -LsSf https://astral.sh/uv/install.sh | sh）" >&2
  exit 1
fi
echo "[INFO] uv: $(command -v uv)"

if [[ -n "${UV_INDEX_URL:-}" ]]; then
  export UV_INDEX_URL
  echo "[INFO] UV_INDEX_URL=${UV_INDEX_URL}（默认官方 PyPI；受限环境可指向镜像）"
else
  echo "[INFO] 使用官方 PyPI（UV_INDEX_URL 未设置）"
fi

if [[ -x "${VENV}/bin/python" ]]; then
  echo "[INFO] 环境已存在: ${VENV}（如需重装请先删除该目录）"
else
  echo "[INFO] 创建 venv（Python 3.12，缺失时回退 3.11）..."
  uv venv -p 3.12 "${VENV}" || uv venv -p 3.11 "${VENV}"
fi

PY="${VENV}/bin/python"
echo "[INFO] python: $("${PY}" --version 2>&1)"

cd "${SERVER_DIR}"

echo "[INFO] 1/3 安装 lerobot（github @ ${LEROBOT_REV}，约 264MB，需几分钟）..."
uv pip install --python "${PY}" "lerobot @ git+https://github.com/huggingface/lerobot@${LEROBOT_REV}"

echo "[INFO] 2/3 安装 openpi-client（本地包，editable）..."
uv pip install --python "${PY}" -e packages/openpi-client

echo "[INFO] 3/3 安装 openpi（本地包，editable，含 jax/torch 等重依赖）..."
OVERRIDES="$(mktemp)"
trap 'rm -f "${OVERRIDES}"' EXIT
# 复刻 pyproject [tool.uv] override-dependencies（与 jax 0.5.3 实测兼容的组合）
printf 'ml-dtypes==0.4.1\ntensorstore==0.1.74\n' > "${OVERRIDES}"
uv pip install --python "${PY}" -e . --override "${OVERRIDES}"

# openpi 的 fsq_tokenizer 直接 import chex，但 pyproject 未声明依赖；
# --no-deps 避免解析器把 jax/jaxlib 升级出锁文件组合（版本与实测验证一致）。
echo "[INFO] 3.5/3 安装 chex（openpi fsq_tokenizer 需要；--no-deps 保持 jax 版本组合）..."
uv pip install --python "${PY}" --no-deps "chex==0.1.92" "jmp" "toolz"

# openpi 的 gemma_pytorch 在 import 链顶层依赖 pytest（源仓库同样装在运行 venv）。
echo "[INFO] 3.6/3 安装 pytest（openpi gemma_pytorch 顶层 import 依赖）..."
uv pip install --python "${PY}" pytest

echo "[INFO] 验证安装..."
"${PY}" -c "import jax, openpi, openpi_client, chex, pytest; print('jax', jax.__version__, '| openpi / openpi_client / chex / pytest OK')"

echo "[INFO] 完成。启动服务端（仓库根目录）："
echo "  bash scripts/start_server.sh --ckpt ckpt/<任务名>/<step> [--rtc] [--delta] [--lora]"
echo "  首次启动会从 gs://big_vision 自动下载 tokenizer 到 OPENPI_DATA_HOME（默认 <仓库>/cache/openpi）。"
