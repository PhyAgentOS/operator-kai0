#!/usr/bin/env bash
# kai0_runner：policy.runner 节点单文件 bundle 组装（forge skill_runtime 安装契约）。
#
# 契约（PhyAgentOS skill_runtime/installer.py _extract_executable）：
#   tar.gz 内恰好一个普通文件、位于归档根、文件名 == lock.entrypoint（kai0_policy）。
#   → 服务端代码 / start_server.sh / 推理配置必须内嵌进 PyInstaller 单文件
#     （见 build/kai0_policy.spec 的 datas），tar 包本身不允许放任何伴生文件。
#
# 产出：build_binaries/bundles/kai0_policy-1.0.4-linux-x86_64.tar.gz
# 并打印二进制与归档 sha256（归档值即 skill lock.sha256；二进制值供 receipt 核对）。
#
# 环境门：需要 pyinstaller（含 PyInstaller 可用的 Python 环境）——
# 不可用时退出码 2 并给出提示，绝不静默产出残缺 bundle。
set -euo pipefail

VERSION="1.0.4"
ENTRYPOINT="kai0_policy"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$REPO_ROOT/build_binaries"
OUT="$BUILD/bundles"
ARTIFACT_ID="${ENTRYPOINT}-${VERSION}-linux-x86_64"

# 构建解释器钉死仓库 .venv：site-packages 的 kai0_inference 是 copy 安装，
# PATH 上的过期 build venv（.venv_build，0.1.0 旧码）曾把旧代码打进 1.0.3 bundle。
# 禁止回落 command -v：宁可失败重跑，不可静默用错环境。
PYINSTALLER="$REPO_ROOT/.venv/bin/pyinstaller"
if [ ! -x "$PYINSTALLER" ]; then
  echo "[ENV-GATE] 未找到 $PYINSTALLER 。请先执行：uv sync --extra dev" >&2
  exit 2
fi

mkdir -p "$OUT"
cd "$REPO_ROOT"
"$PYINSTALLER" build/kai0_policy.spec --noconfirm \
  --distpath "$BUILD/dist" --workpath "$BUILD/pyinstaller-work"

# 产出断言：二进制自报版本必须等于 VERSION（1.0.3 事故的闸门）
REPORTED="$("$BUILD/dist/$ENTRYPOINT" --version 2>&1 || true)"
if ! grep -qF "$VERSION" <<<"$REPORTED"; then
  echo "[ERROR] 二进制版本断言失败：'$REPORTED'（期望包含 kai0 $VERSION）。" >&2
  exit 1
fi

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
