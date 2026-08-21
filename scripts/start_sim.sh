#!/usr/bin/env bash
# 双臂仿真启动脚本（合并 run.sh / run_rtc.sh，RTC 为推荐默认路径）：真实 kai0 推理服务端
# + dora dataflow。前置：真实服务端已在 8000 端口运行——
#   RTC（默认）：bash scripts/start_server.sh --ckpt <step> --rtc
#   同步（--sync）：bash scripts/start_server.sh --ckpt <step>（不加 --rtc）
# 用法：bash scripts/start_sim.sh [秒数] [--rtc|--sync]，默认 RTC、默认 30 秒
# 可选环境变量（Forge 框架工作区非默认位置时）：
#   FORGE_ROOT  Forge 框架工作区根目录（默认自示例目录相对上溯 5 级，与 dataflow 内
#               robot_router/mujoco_sim/image_viewer 的相对路径约定一致）
#   DORA_BIN    dora 可执行文件（默认 ${FORGE_ROOT}/forge_runtime/.venv/bin/dora）
#   UV_BIN      uv 可执行文件（默认 PATH 中的 uv；无网环境可另设 UV_INDEX_URL 镜像）
#   SERVER_HOST 推理服务端地址（默认 127.0.0.1；客户端与服务端分机部署时设为服务端 IP，
#               须与 kai0_rtc.yaml 的 server_url 一致）
#   SERVER_PORT 推理服务端端口（默认 8000；改端口须同步改 kai0_rtc.yaml server_url）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_DIR="${ROOT_DIR}/examples/dora_sim_kai0_dual_arm"

MODE="rtc"   # 默认 RTC（推荐）；--sync 切同步
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8000}"
SECONDS_TO_RUN=30

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rtc)   MODE="rtc"; shift ;;
    --sync)  MODE="sync"; shift ;;
    -h|--help)
      echo "用法: $0 [秒数] [--rtc|--sync]"
      echo "  --rtc    RTC 异步推理（默认，推荐；服务端须以 --rtc 启动）"
      echo "  --sync   同步模式（调试/回退；服务端不加 --rtc）"
      exit 0
      ;;
    *)
      SECONDS_TO_RUN="$1"
      shift
      ;;
  esac
done

if [[ "${MODE}" == "rtc" ]]; then
  DATAFLOW="dataflow_rtc.yaml"
  LOG="dora_run_rtc.log"
else
  DATAFLOW="dataflow.yaml"
  LOG="dora_run.log"
fi

cd "${SIM_DIR}"

# dora run 结束后清理残留节点进程：timeout 只终止 dora daemon，daemon spawn 的
# 节点进程可能残留为孤儿（实测存在）。按示例目录节点与 robot_router 的路径模式
# 匹配清理，无匹配时是 no-op；trap EXIT 覆盖正常超时/异常退出/Ctrl+C 全部路径。
cleanup_nodes() {
  local pids
  pids="$(pgrep -f "${SIM_DIR}/nodes/" 2>/dev/null || true) $(pgrep -f 'forge_robot_router/main[.]py' 2>/dev/null || true)"
  pids="$(echo ${pids} | tr ' ' '\n' | sed '/^$/d' | sort -u | tr '\n' ' ')"
  if [ -n "${pids// /}" ]; then
    echo "== 清理残留节点进程: ${pids} =="
    kill ${pids} 2>/dev/null || true
    sleep 2
    pids="$(pgrep -f "${SIM_DIR}/nodes/" 2>/dev/null || true) $(pgrep -f 'forge_robot_router/main[.]py' 2>/dev/null || true)"
    pids="$(echo ${pids} | tr ' ' '\n' | sed '/^$/d' | sort -u | tr '\n' ' ')"
    if [ -n "${pids// /}" ]; then
      echo "== 强杀残留: ${pids} =="
      kill -9 ${pids} 2>/dev/null || true
    fi
  fi
}
trap cleanup_nodes EXIT

FORGE_ROOT="${FORGE_ROOT:-$(cd "${SIM_DIR}/../../../../.." && pwd)}"
DORA="${DORA_BIN:-${FORGE_ROOT}/forge_runtime/.venv/bin/dora}"
UV="${UV_BIN:-$(command -v uv || true)}"

if [[ ! -x "${DORA}" ]]; then
  echo "== dora 不存在: ${DORA}（可用 DORA_BIN 指定 forge_runtime 的 dora）=="
  exit 1
fi
if [[ -z "${UV}" ]]; then
  echo "== 未找到 uv：请先安装（https://docs.astral.sh/uv/），或用 UV_BIN 指定 =="
  exit 1
fi
export PATH="$(dirname "${UV}"):$PATH"

echo "== 检查真实推理服务端（ws://${SERVER_HOST}:${SERVER_PORT}，与 ${DATAFLOW} 的 server_url 一致）=="
if ! curl -s "http://${SERVER_HOST}:${SERVER_PORT}/healthz" >/dev/null 2>&1; then
  if [[ "${MODE}" == "rtc" ]]; then
    echo "== 服务端未就绪：请先以 --rtc 启动（bash scripts/start_server.sh --ckpt <step> --rtc）=="
  else
    echo "== 服务端未就绪：请先启动（bash scripts/start_server.sh --ckpt <step>，不加 --rtc）=="
  fi
  exit 1
fi

echo "== dora build（${DATAFLOW}）=="
"$DORA" build "${DATAFLOW}" --uv

echo "== dora run（${SECONDS_TO_RUN}s 后自动停止，日志写入 ${LOG}）=="
timeout "${SECONDS_TO_RUN}" "$DORA" run "${DATAFLOW}" --uv 2>&1 | tee "${LOG}" || {
  code=$?
  if [ "$code" -eq 124 ]; then
    echo "== 运行时长达到 ${SECONDS_TO_RUN}s，正常超时结束 =="
  else
    echo "== dora run 异常退出（exit $code）=="
    exit "$code"
  fi
}
