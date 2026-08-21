#!/usr/bin/env bash
# kai0 真机启动脚本：真实 kai0 推理服务端 + 真机 dora dataflow（双臂 + 三相机）。
# 前置：真实服务端已在 8000 端口以 RTC 运行——
#   bash scripts/start_server.sh --ckpt <step> --rtc
# 用法：bash scripts/start_real.sh [-h] [--mock]
#   --mock  软件在环模式（无机械臂/相机）：mock 末端 + 同一 dataflow 拓扑，
#           无安全确认；用于无硬件验收（需 usb_camera 之外的节点依赖已解析）。
# 安全：真机模式启动前要求输入 RUN 确认（非交互必须显式传 --yes）。
# 可选环境变量（Forge 框架工作区非默认位置时）：
#   FORGE_ROOT  Forge 框架工作区根目录（默认自示例目录相对上溯 5 级，与 dataflow 内
#               task_robot/piper/usb_camera/image_viewer 的相对路径约定一致）
#   DORA_BIN    dora 可执行文件（默认 ${FORGE_ROOT}/forge_runtime/.venv/bin/dora）
#   UV_BIN      uv 可执行文件（默认 PATH 中的 uv；无网环境可另设 UV_INDEX_URL 镜像）
#   SERVER_HOST 推理服务端地址（默认 127.0.0.1；客户端与服务端分机部署时设为服务端 IP，
#               须与 kai0_rtc.yaml 的 server_url 一致）
#   SERVER_PORT 推理服务端端口（默认 8000；改端口须同步改 kai0_rtc.yaml server_url）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8000}"
ASSUME_YES=0
MOCK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes)
      ASSUME_YES=1
      shift
      ;;
    --mock)
      MOCK=1
      shift
      ;;
    -h|--help)
      echo "用法: $0 [--yes] [--mock]"
      echo "  启动真机双臂（kai0 策略 + 真机 Piper + 三路相机）。"
      echo "  前置: bash scripts/start_server.sh --ckpt <step> --rtc 已在 ${SERVER_PORT} 端口运行。"
      echo "  --yes   跳过 RUN 安全确认（非交互场景）"
      echo "  --mock  软件在环模式（无硬件）：mock 末端 + 同一拓扑，跳过安全确认"
      exit 0
      ;;
    *)
      echo "[ERROR] 未知参数: $1" >&2
      exit 2
      ;;
  esac
done

REAL_DIR="${ROOT_DIR}/examples/dora_real_kai0_dual_arm"
if [[ "${MOCK}" -eq 1 ]]; then
  DATAFLOW="dataflow_mock.yaml"
  LOG="dora_run_mock.log"
else
  DATAFLOW="dataflow.yaml"
  LOG="dora_run.log"
fi

FORGE_ROOT="${FORGE_ROOT:-$(cd "${REAL_DIR}/../../../../.." && pwd)}"
DORA="${DORA_BIN:-${FORGE_ROOT}/forge_runtime/.venv/bin/dora}"
UV="${UV_BIN:-$(command -v uv || true)}"
USB_CAMERA="${USB_CAMERA_BIN:-${FORGE_ROOT}/devices/camera/usb_camera/target/release/usb_camera}"

if [[ ! -x "${DORA}" ]]; then
  echo "== dora 不存在: ${DORA}（可用 DORA_BIN 指定 forge_runtime 的 dora）=="
  exit 1
fi
if [[ -z "${UV}" ]]; then
  echo "== 未找到 uv：请先安装（https://docs.astral.sh/uv/），或用 UV_BIN 指定 =="
  exit 1
fi
if [[ "${MOCK}" -ne 1 && ! -x "${USB_CAMERA}" ]]; then
  echo "== usb_camera 二进制不存在: ${USB_CAMERA} =="
  echo "== 请先构建：cd ${FORGE_ROOT}/devices/camera/usb_camera && cargo build --release --bins =="
  exit 1
fi
export PATH="$(dirname "${UV}"):$PATH"

# 安全确认（真机部署定型口径）；mock 模式无硬件，跳过
if [[ "${MOCK}" -eq 1 ]]; then
  echo "[INFO] mock 模式：软件在环末端（无机械臂/相机），仅验证数据流与配置。"
elif [[ "${ASSUME_YES}" -ne 1 ]]; then
  echo "[SAFETY] 即将启动真实双臂控制。"
  echo "[SAFETY] 请确认工作区无人、急停可用、CAN/相机映射正确。"
  read -r -p "输入 RUN 继续: " confirmation
  if [[ "${confirmation}" != "RUN" ]]; then
    echo "[INFO] 已取消。"
    exit 0
  fi
fi

cd "${REAL_DIR}"

# dora run 结束后清理残留节点进程：timeout/Ctrl+C 只终止 dora daemon，
# daemon spawn 的节点进程可能残留为孤儿。按示例目录节点 + 框架工作区节点
# （task_robot / image_viewer）的路径模式匹配清理，无匹配时是 no-op；
# EXIT + INT + TERM 覆盖正常退出 / Ctrl+C / timeout 杀进程全部路径。
cleanup_nodes() {
  local pids
  pids="$(pgrep -f "${REAL_DIR}/nodes/" 2>/dev/null || true) $(pgrep -f 'packages/nodes/task_robot/main[.]py' 2>/dev/null || true) $(pgrep -f 'target/release/image_viewer' 2>/dev/null || true)"
  pids="$(echo ${pids} | tr ' ' '\n' | sed '/^$/d' | sort -u | tr '\n' ' ')"
  if [ -n "${pids// /}" ]; then
    echo "== 清理残留节点进程: ${pids} =="
    kill ${pids} 2>/dev/null || true
    sleep 2
    pids="$(pgrep -f "${REAL_DIR}/nodes/" 2>/dev/null || true) $(pgrep -f 'packages/nodes/task_robot/main[.]py' 2>/dev/null || true) $(pgrep -f 'target/release/image_viewer' 2>/dev/null || true)"
    pids="$(echo ${pids} | tr ' ' '\n' | sed '/^$/d' | sort -u | tr '\n' ' ')"
    if [ -n "${pids// /}" ]; then
      echo "== 强杀残留: ${pids} =="
      kill -9 ${pids} 2>/dev/null || true
    fi
  fi
}
trap cleanup_nodes EXIT INT TERM

echo "== 检查真实推理服务端（ws://${SERVER_HOST}:${SERVER_PORT}，与 dataflow 的 server_url 一致）=="
if ! curl -s "http://${SERVER_HOST}:${SERVER_PORT}/healthz" >/dev/null 2>&1; then
  echo "== 服务端未就绪：请先以 --rtc 启动（bash scripts/start_server.sh --ckpt <step> --rtc）=="
  exit 1
fi

echo "== dora build（${DATAFLOW}）=="
"$DORA" build "${DATAFLOW}" --uv

echo "== dora run（Ctrl+C 结束，日志写入 ${LOG}）=="
"$DORA" run "${DATAFLOW}" --uv 2>&1 | tee "${LOG}" || {
  code=$?
  echo "== dora run 退出（exit $code）；真机请确认已先按急停 =="
  exit "$code"
}
