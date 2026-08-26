# -*- mode: python ; coding: utf-8 -*-
"""kai0_policy 节点单文件打包（forge skill_runtime 单可执行 bundle 契约）。

单文件契约（PhyAgentOS skill_runtime/installer.py _extract_executable）：
tar.gz 内恰好一个普通文件（本二进制）——因此服务端代码 / start_server.sh /
推理配置全部内嵌进 onefile，运行期经 sys._MEIPASS 解析：
  - _MEIPASS/scripts/start_server.sh（ROOT_DIR=_MEIPASS → server/kai0_prompt_schedule）
  - _MEIPASS/config/inference/kai0_libero.yaml（session_endpoint 就绪门配置回退）
服务端运行依赖外部 SERVER_VENV（start_server.sh 用 SERVER_VENV 覆盖默认 .venv 路径）。
"""
import os

from PyInstaller.utils.hooks import collect_dynamic_libs

# SPECPATH 是 spec 所在目录（PyInstaller build_main.py: CONF['specpath']），
# 不是文件路径——别再对它 dirname（那会多向上一层）。
SPEC_DIR = os.path.abspath(SPECPATH)
ROOT = os.path.abspath(os.path.join(SPEC_DIR, ".."))

binaries = []
binaries += collect_dynamic_libs("dora")

datas = [
    # 服务端代码树（start_server.sh ROOT_DIR 解析 _MEIPASS 后取 server/kai0_prompt_schedule）
    (os.path.join(ROOT, "server", "kai0_prompt_schedule"), "server/kai0_prompt_schedule"),
    # 服务启动脚本（_default_start_script 优先取 _MEIPASS/scripts/start_server.sh）
    (os.path.join(ROOT, "scripts", "start_server.sh"), "scripts"),
    # 推理配置（_resolve_inference_config 的 _MEIPASS 回退路径）
    (os.path.join(ROOT, "config", "inference", "kai0_libero.yaml"), "config/inference"),
]

a = Analysis(
    [os.path.join(SPEC_DIR, "kai0_policy_entry.py")],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="kai0_policy",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
