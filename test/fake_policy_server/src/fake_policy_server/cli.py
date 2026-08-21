"""命令行入口：python -m fake_policy_server --port 8000 --mode sine"""

from __future__ import annotations

import argparse
import logging

from fake_policy_server.actions import SineActionSource, ZeroActionSource
from fake_policy_server.policy import FakePolicy
from fake_policy_server.server import FakePolicyServer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")
    parser.add_argument("--mode", choices=("zero", "sine"), default="zero",
                        help="动作模式：zero=全零；sine=逐关节正弦（用于目视验证顺序映射）")
    parser.add_argument("--action-dim", type=int, default=14, help="动作维度（默认 14）")
    parser.add_argument("--chunk-size", type=int, default=50, help="动作块长度（默认 50）")
    parser.add_argument("--default-task-index", type=int, default=None,
                        help="初始 sticky prompt_id，模拟 DEFAULT_PROMPT_ID 环境变量")
    parser.add_argument("--infer-delay-ms", type=float, default=0.0,
                        help="每次推理的模拟耗时（毫秒，用于时序相关测试）")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    source = SineActionSource() if args.mode == "sine" else ZeroActionSource()
    policy = FakePolicy(
        source,
        action_dim=args.action_dim,
        chunk_size=args.chunk_size,
        infer_delay_ms=args.infer_delay_ms,
    )
    server = FakePolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata={"server": "fake-policy-server", "mode": args.mode},
        default_task_index=args.default_task_index,
    )
    server.serve_forever()
    return 0
