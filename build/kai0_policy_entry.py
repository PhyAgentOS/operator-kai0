# PyInstaller 入口：统一 CLI（infer / infer-once 子命令分发；勿用 main_infer——
# 那是 kai0-infer console script 语义，会把首参强制改为 infer）。
import sys

from kai0_inference.cli import main

sys.exit(main())
