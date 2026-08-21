"""统一入口：根据首个参数选择 infer / infer-once。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kai0_inference import __version__


_COMMANDS = ("infer", "infer-once")

_PROG_ALIASES = {
    "kai0-infer-once": "infer-once",
    "kai0_infer_once": "infer-once",
    "kai0-infer": "infer",
    "kai0_infer": "infer",
}


class _ChineseHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """保留多行说明；命令简述使用中文。"""


def _prog_name() -> str:
    return Path(sys.argv[0]).name


def _alias_command() -> str | None:
    name = _prog_name()
    if name in _PROG_ALIASES:
        return _PROG_ALIASES[name]
    # PyInstaller onedir 可执行名常见为 kai0_infer
    stem = Path(name).stem
    return _PROG_ALIASES.get(stem)


def _localize_parser(parser: argparse.ArgumentParser) -> None:
    """将 argparse 默认英文分组标题与 -h 说明改为中文。"""
    parser._optionals.title = "可选参数"
    if parser._positionals.title in (None, "positional arguments"):
        parser._positionals.title = "位置参数"
    # 覆盖默认英文 help 动作说明
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            action.help = "显示帮助信息并退出"


def _add_subparser(subparsers: argparse._SubParsersAction, *args, **kwargs) -> argparse.ArgumentParser:
    kwargs.setdefault("formatter_class", _ChineseHelpFormatter)
    kwargs.setdefault("add_help", True)
    sp = subparsers.add_parser(*args, **kwargs)
    _localize_parser(sp)
    return sp


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kai0",
        description=(
            "kai0 策略节点统一工具。\n"
            "通过第一个参数选择功能：infer（Dora 在线推理）、"
            "infer-once（本地单步 smoke）。"
        ),
        epilog=(
            "示例:\n"
            "  kai0 infer --config config/inference/kai0.yaml\n"
            "  kai0 infer-once --config config/inference/kai0.yaml\n"
            "\n"
            "兼容说明:\n"
            "  若通过 kai0-infer / kai0-infer-once 调用，可省略子命令"
            "（按可执行文件名自动选择功能）。\n"
            "  二进制名含 kai0_infer 且未写子命令时，默认按 infer 处理。"
        ),
        formatter_class=_ChineseHelpFormatter,
    )
    _localize_parser(parser)
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="显示版本信息并退出",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        title="功能",
        description="必须指定下列之一作为第一个参数",
        metavar="{infer,infer-once}",
        required=True,
    )

    # —— infer ——
    p_infer = _add_subparser(
        subparsers,
        "infer",
        help="以 Dora 策略节点运行在线推理",
        description=(
            "推理：作为 Dora 节点订阅 proprio_state 与多路图像，"
            "经策略适配器（DummyAdapter / 同步 WS 客户端）输出 JointCommand。"
        ),
    )
    p_infer.add_argument(
        "--config",
        type=str,
        default=None,
        help="推理 YAML 配置路径（也可用环境变量 KAI0_INFERENCE_CONFIG）",
    )
    p_infer.set_defaults(_handler="infer")

    # —— infer-once ——
    p_once = _add_subparser(
        subparsers,
        "infer-once",
        help="本地单步推理 smoke（不依赖 Dora）",
        description=(
            "单步 smoke：按配置加载策略适配器，用随机观测跑一步 generate_action，"
            "打印 action JSON，用于验证环境与链路。"
        ),
    )
    p_once.add_argument(
        "--config",
        type=str,
        required=True,
        help="推理 YAML 配置文件路径",
    )
    p_once.add_argument(
        "--state-dim",
        type=int,
        default=None,
        help="覆盖 observation.state 长度（默认：关节数）",
    )
    p_once.add_argument(
        "--height",
        type=int,
        default=224,
        help="假图像高度（像素，默认 224，与服务端契约一致）",
    )
    p_once.add_argument(
        "--width",
        type=int,
        default=224,
        help="假图像宽度（像素，默认 224，与服务端契约一致）",
    )
    p_once.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子（默认 0）",
    )
    p_once.set_defaults(_handler="infer-once")

    return parser


def _normalize_argv(argv: list[str] | None) -> list[str]:
    """保证 argv[0] 为子命令；兼容旧入口名自动注入。

    不注入的情况：
    - 已显式写出 infer / infer-once
    - 仅要顶层元信息（无参数、-h / --help 或 --version）
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in _COMMANDS:
        return args
    # 顶层帮助/版本：不要因可执行名 kai0_infer 被误注入子命令。
    if not args or args[0] in ("-h", "--help", "--version"):
        return args
    alias = _alias_command()
    if alias is not None:
        return [alias, *args]
    return args


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(_normalize_argv(argv))
    handler = getattr(args, "_handler", None)

    if handler == "infer":
        from kai0_inference.inference.main import run_infer

        return run_infer(args)
    if handler == "infer-once":
        from kai0_inference.inference.run_once import run_infer_once

        return run_infer_once(args)

    parser.error("未知功能，请指定 infer / infer-once")
    return 2


def main_infer(argv: list[str] | None = None) -> int:
    """兼容 console_script：kai0-infer。"""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] != "infer":
        args = ["infer", *args]
    return main(args)


def main_infer_once(argv: list[str] | None = None) -> int:
    """兼容 console_script：kai0-infer-once。"""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] != "infer-once":
        args = ["infer-once", *args]
    return main(args)


if __name__ == "__main__":
    raise SystemExit(main())
