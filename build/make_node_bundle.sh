#!/usr/bin/env bash
# kai0_runner：policy.runner 节点单文件 bundle 组装（forge skill_runtime 安装契约）。
#
# 契约（PhyAgentOS skill_runtime/installer.py _extract_executable）：
#   tar.gz 内恰好一个普通文件、位于归档根、文件名 == lock.entrypoint（kai0_policy）。
#   → 服务端代码 / start_server.sh / 推理配置必须内嵌进 PyInstaller 单文件
#     （见 build/kai0_policy.spec 的 datas），tar 包本身不允许放任何伴生文件。
#
# 产出：build_binaries/bundles/kai0_policy-1.0.3-linux-x86_64.tar.gz
# 并打印二进制与归档 sha256（归档值即 skill lock.sha256；二进制值供 receipt 核对）。
#
# 环境门：需要 pyinstaller（含 PyInstaller 可用的 Python 环境）——
# 不可用时退出码 2 并给出提示，绝不静默产出残缺 bundle。
set -euo pipefail

VERSION="1.0.3"
ENTRYPOINT="kai0_policy"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$REPO_ROOT/build_binaries"
OUT="$BUILD/bundles"
ARTIFACT_ID="${ENTRYPOINT}-${VERSION}-linux-x86_64"

if ! command -v pyinstaller >/dev/null 2>&1; then
  echo "[ENV-GATE] pyinstaller 不可用。请先安装（如：uv pip install pyinstaller，"
  echo "           或激活含 PyInstaller 的 build venv）后重跑本脚本。" >&2
  exit 2
fi

mkdir -p "$OUT"
cd "$REPO_ROOT"
pyinstaller build/kai0_policy.spec --noconfirm \
  --distpath "$BUILD/dist" --workpath "$BUILD/pyinstaller-work"

# 单文件 staging：归档根恰好一个普通文件 kai0_policy（installer 断言同名）
STAGE="$BUILD/staging/${ARTIFACT_ID}"
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp "$BUILD/dist/${ENTRYPOINT}" "$STAGE/${ENTRYPOINT}"
chmod +x "$STAGE/${ENTRYPOINT}"

ARCHIVE="$OUT/${ARTIFACT_ID}.tar.gz"
tar -czf "$ARCHIVE" -C "$STAGE" "$ENTRYPOINT"
rm -rf "$STAGE"

# 单入口断言（installer _extract_executable 同口径：恰好一行且等于 entrypoint）
MEMBERS="$(tar -tzf "$ARCHIVE")"
if [ "$MEMBERS" != "$ENTRYPOINT" ]; then
  echo "[ERROR] bundle 根文件必须恰好为 ${ENTRYPOINT}，实际：${MEMBERS}" >&2
  exit 1
fi

BIN_SHA="$(sha256sum "$BUILD/dist/${ENTRYPOINT}" | cut -d' ' -f1)"
ARCHIVE_SHA="$(sha256sum "$ARCHIVE" | cut -d' ' -f1)"
echo "已产出 $ARCHIVE（$(du -h "$ARCHIVE" | cut -f1)）"
echo "  binary_sha256:  $BIN_SHA"
echo "  archive_sha256: $ARCHIVE_SHA  （skill lock.sha256）"
