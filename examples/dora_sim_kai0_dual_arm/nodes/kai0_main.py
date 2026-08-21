#!/usr/bin/env python3
"""本地包装：以包上下文启动 kai0 推理节点（源码模式过渡方案）。

dora 的节点 path 必须是可直接运行的 python 文件；kai0_runner 以 pip 包安装
（本目录 pyproject 中 editable path 依赖），本包装直接调用其统一 CLI 入口，
argv 由 dataflow 的 args 透传（infer --config kai0_rtc.yaml）。
节点以 PyInstaller 二进制形态运行后，本包装退役。
"""

from __future__ import annotations

import sys

from kai0_inference.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
