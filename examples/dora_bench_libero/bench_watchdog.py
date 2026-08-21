#!/usr/bin/env python3
"""LIBERO 批次外部 watchdog（第 4 步增强 C 的替代口径，benchmark 节点零改动）。

职责：
- 为每次批次生成 run_id（UUID4），归档结果到 run-<run_id>-<stamp>.json（设计文档 3.10 静态模式口径）
- 包装 `dora run <dataflow>`，流式监控日志进度
- wall-clock 两档超时：episode（距最后进度事件）/ batch（总时长）——超时**杀进程组**、
  写 partial 标注 sidecar，支持直接重跑
- 日志记录最后进度时间与当前 episode 供诊断

用法：
  python3 bench_watchdog.py --dora-bin ~/miniconda3/envs/paos-skill-demo/bin \
      --dataflow <dataflow.yaml> --result-dir <结果目录> \
      [--episode-timeout-s 300] [--batch-timeout-s 7200] [--log <运行日志>]

退出码：0=批次正常完成；2=watchdog 超时杀批（已写 partial 标注，可直接重跑）；1=其他失败
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import yaml

# benchmark 节点日志的进度锚点（main.py 的 _emit_status/_log 格式）
RE_EPISODE_START = re.compile(r"episode (\d+)/(\d+) 开始 t(\d+)_i(\d+)_r(\d+) :: (.*)")
RE_EPISODE_END = re.compile(r"^([\w_]+_t\d+_i\d+_r\d+) (success|failed|timed_out) \((\d+) 步\)")
RE_BATCH_END = re.compile(r"批次 (succeeded|failed|cancelled)：")
# policy 侧真实活动信号（精确锚点）。不得用宽松关键字匹配——实测踩坑：policy 异常
# traceback 刷屏（行内含 "action_payload"/"generate_action" 等）会让宽匹配误判为
# "有进展"，持续刷新最后进度时间，episode 超时永不触发、挂死检测失效。
RE_POLICY_PROGRESS = re.compile(
    r"\[stub_policy.*\](command|summary)"
    r"|request=\d+ round_trip_ms="
    r"|action_stats\["
    r"|episode_first_obs"
)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Watchdog:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_id = str(uuid.uuid4())
        self.stamp = time.strftime("%Y%m%d-%H%M%S")
        self.current_episode: str | None = None
        self.last_progress_at: float = time.time()
        self.last_progress_line: str | None = None
        self.episodes_seen: list[str] = []
        self.batch_status: str | None = None
        self.child: subprocess.Popen | None = None
        self._stop_reason: str | None = None

    # ---- 进度监控 ----

    def _on_line(self, line: str) -> None:
        line = line.rstrip()
        m = RE_EPISODE_START.search(line)
        if m:
            self.current_episode = f"t{m.group(3)}_i{m.group(4)}_r{m.group(5)}"
            if self.current_episode not in self.episodes_seen:
                self.episodes_seen.append(self.current_episode)
            self._progress(line, "episode_start")
            return
        m = RE_EPISODE_END.match(line)
        if m:
            self._progress(line, "episode_end")
            return
        m = RE_BATCH_END.search(line)
        if m:
            self.batch_status = m.group(1)
            self._progress(line, "batch_end")
            return
        if RE_POLICY_PROGRESS.search(line):
            self._progress(line, "policy_activity")

    def _progress(self, line: str, kind: str) -> None:
        self.last_progress_at = time.time()
        self.last_progress_line = line
        print(f"[watchdog] progress({kind}): {line[:120]}", flush=True)

    # ---- 日志跟踪线程 ----

    def _follow_log(self, log_path: str) -> None:
        with open(log_path, "rb") as f:
            while True:
                where = f.tell()
                line = f.readline()
                if not line:
                    f.seek(where)
                    time.sleep(0.2)
                    continue
                try:
                    self._on_line(line.decode("utf-8", errors="replace"))
                except Exception:  # noqa: BLE001 - 监控不能因单行解析崩溃
                    pass

    # ---- dataflow 节点进程识别（共享 daemon 场景下的兜底清理）----

    def _node_patterns(self) -> list[str]:
        """从 dataflow yaml 的 args 里提取节点脚本路径，用于 pgrep 兜底清理。"""
        try:
            doc = yaml.safe_load(Path(self.args.dataflow).read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            return []
        patterns: list[str] = []
        for node in doc.get("nodes", []):
            for a in node.get("args", []) or []:
                if a.endswith(".py") and "python" not in a:
                    patterns.append(a)
        return patterns

    def _kill_node_processes(self) -> None:
        patterns = self._node_patterns()
        if not patterns:
            return
        import shlex
        # 用固定标记文件避免误杀：只杀「本批次启动后」的节点进程？
        # 简化口径：按脚本路径匹配（本任务节点脚本路径唯一）。
        for pat in patterns:
            code = subprocess.run(
                ["pgrep", "-f", pat], capture_output=True, text=True
            )
            for pid in code.stdout.split():
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    print(f"[watchdog] 兜底清理节点进程 {pid}（{pat}）", flush=True)
                except (ProcessLookupError, PermissionError):
                    pass

    def _stop_batch(self, reason: str, graceful: bool = True) -> None:
        """分级停止：SIGINT（graceful，触发 dora 停机让节点收 STOP）→ SIGKILL → 节点进程兜底。"""
        self._stop_reason = reason
        if self.child is None or self.child.poll() is not None:
            return
        print(f"[watchdog] 停止批次（{reason}）…", flush=True)
        if graceful:
            try:
                os.killpg(os.getpgid(self.child.pid), signal.SIGINT)
                try:
                    self.child.wait(timeout=15)
                    print("[watchdog] SIGINT 优雅停止成功")
                    return
                except subprocess.TimeoutExpired:
                    print("[watchdog] SIGINT 后 15s 未退出，升级 SIGKILL")
            except (ProcessLookupError, PermissionError):
                pass
        try:
            os.killpg(os.getpgid(self.child.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        print("[watchdog] 已杀进程组（SIGKILL）")
        # 共享 dora daemon 场景：节点进程不属于 dora run 的会话，pgrep 兜底
        self._kill_node_processes()

    # ---- 批次执行 ----

    def run(self) -> int:
        dataflow = Path(self.args.dataflow).resolve()
        result_dir = Path(self.args.result_dir).resolve()
        result_dir.mkdir(parents=True, exist_ok=True)
        log_path = Path(self.args.log).resolve() if self.args.log else (
            result_dir / f"run-{self.run_id}-{self.stamp}.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[watchdog] run_id={self.run_id} stamp={self.stamp}")
        print(f"[watchdog] dataflow={dataflow}")
        print(f"[watchdog] log={log_path}")

        log_file = open(log_path, "wb")
        env = dict(os.environ)
        dora_bin = str(Path(self.args.dora_bin).resolve())
        env["PATH"] = dora_bin + os.pathsep + env.get("PATH", "")
        cmd = [os.path.join(dora_bin, "dora"), "run", str(dataflow)]
        print(f"[watchdog] cmd: {' '.join(cmd)}")

        self.child = subprocess.Popen(
            cmd,
            cwd=str(dataflow.parent),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        follower = threading.Thread(target=self._follow_log, args=(str(log_path),), daemon=True)
        follower.start()

        batch_start = time.time()
        batch_end_at: float | None = None
        verdict: int = 1
        try:
            while True:
                rc = self.child.poll()
                if rc is not None:
                    time.sleep(1.5)  # 等 follower 读完尾部
                    if rc == 0 and self.batch_status in ("succeeded", "failed", "cancelled"):
                        verdict = 0
                        print(f"[watchdog] 批次正常结束（{self.batch_status}，exit {rc}）")
                    else:
                        verdict = 1
                        print(f"[watchdog] dora run 异常退出（exit {rc}）")
                    break
                now = time.time()
                if self.batch_status is not None and batch_end_at is None:
                    batch_end_at = now
                if batch_end_at is not None and now - batch_end_at > self.args.stop_grace_s:
                    # 批次已结束但 flow 未退出（如 policy 节点不随批次退出）→ 优雅停止
                    print(f"[watchdog] 批次已结束但 flow {self.args.stop_grace_s:.0f}s 未退出，"
                          f"触发优雅停止")
                    self._stop_batch("batch_end_flow_still_running")
                    # 停止后等 dora run 退出
                    try:
                        self.child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
                    verdict = 0
                    break
                if now - self.last_progress_at > self.args.episode_timeout_s:
                    print(f"[watchdog] episode 超时：距最后进度 "
                          f"{now - self.last_progress_at:.0f}s > {self.args.episode_timeout_s}s，"
                          f"当前 episode={self.current_episode}")
                    print(f"[watchdog] 最后进度：{self.last_progress_line}")
                    self._stop_batch(reason=f"episode_timeout（当前 {self.current_episode}）")
                    verdict = 2
                    break
                if now - batch_start > self.args.batch_timeout_s:
                    self._stop_batch(reason="batch_timeout")
                    verdict = 2
                    break
                time.sleep(2.0)
        finally:
            log_file.close()
            if self.child.poll() is None:
                self._stop_batch("run_cleanup")
            self._kill_node_processes()
            self._write_sidecar(result_dir, log_path, verdict)

        return verdict

    # ---- 结果归档与标注 ----

    def _archive_result(self, result_dir: Path) -> str | None:
        """把本批次新落盘的结果 JSON 归档为 run-<uuid>-<stamp>.json（防覆盖命名）。"""
        candidates = sorted(
            result_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
        )
        for p in reversed(candidates):
            if p.name.startswith("run-") or p.name.endswith(".partial.json") or p.name.endswith(".meta.json"):
                continue
            mtime = p.stat().st_mtime
            if time.time() - mtime < 300:  # 本批次窗口内
                target = result_dir / f"run-{self.run_id}-{self.stamp}.json"
                import shutil
                shutil.copy2(p, target)
                print(f"[watchdog] 结果归档：{p.name} → {target.name}")
                return str(target)
        return None

    def _write_sidecar(self, result_dir: Path, log_path: Path, verdict: int) -> None:
        status = {0: "completed", 1: "failed", 2: "partial"}[verdict]
        archived: str | None = None
        if verdict == 0:
            archived = self._archive_result(result_dir)
        sidecar = {
            "run_id": self.run_id,
            "stamp": self.stamp,
            "status": status,
            "verdict": "clean" if verdict == 0 else ("timeout" if verdict == 2 else "error"),
            "stop_reason": self._stop_reason,
            "result_file": archived,
            "log_file": str(log_path),
            "dataflow": str(Path(self.args.dataflow).resolve()),
            "current_episode": self.current_episode,
            "episodes_seen": self.episodes_seen,
            "last_progress_at": self.last_progress_line,
            "rerun_hint": "重跑本命令即可重新发起批次；结果文件按 run_id 命名不会互相覆盖",
        }
        sidecar_path = result_dir / f"run-{self.run_id}-{self.stamp}.meta.json"
        sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[watchdog] sidecar：{sidecar_path.name}（status={status}）")


def main() -> int:
    ap = argparse.ArgumentParser(description="LIBERO 批次 watchdog")
    ap.add_argument("--dora-bin", required=True, help="dora CLI 目录（含 dora 可执行）")
    ap.add_argument("--dataflow", required=True, help="dataflow yaml 路径")
    ap.add_argument("--result-dir", required=True, help="结果目录（sidecar 与归档都落这里）")
    ap.add_argument("--log", default=None, help="运行日志路径（默认结果目录下自动命名）")
    ap.add_argument("--episode-timeout-s", type=float, default=300.0, help="距最后进度事件超时（秒）")
    ap.add_argument("--batch-timeout-s", type=float, default=7200.0, help="批次总时长上限（秒）")
    ap.add_argument("--stop-grace-s", type=float, default=30.0, help="批次结束信号后等待 flow 自退的宽限（秒）")
    args = ap.parse_args()

    wd = Watchdog(args)
    return wd.run()


if __name__ == "__main__":
    sys.exit(main())
