#!/usr/bin/env python3
"""本地包装：以包上下文启动 mujoco_sim 节点。

mujoco_sim 为 src 布局且使用相对导入（from . import __version__），
不能作为独立脚本运行；本包装将其 src 目录加入 sys.path 后调用其 cli 入口。
节点以打包二进制形态运行后，本包装仅为源码模式下的过渡方案。

mujoco_sim 源码位置按 Forge 框架工作区推断（本文件位于框架工作区的
<root>/policies/embodied/kai0_runner/examples/... 内，上溯 6 级即框架根目录，
与 dataflow 内 robot_router 等节点的相对路径约定一致）；非默认布局可用
MUJOCO_SIM_SRC 环境变量显式指定。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_FRAMEWORK_ROOT = Path(__file__).resolve().parents[6]
_SRC = Path(os.environ.get("MUJOCO_SIM_SRC", _FRAMEWORK_ROOT / "simulators/mujoco_sim/src"))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mujoco_sim.main import cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(cli())
