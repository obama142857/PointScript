import argparse
import datetime
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

from automation.default_config import get_default_batch_config
from automation.runner import BatchAutomationRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PointScript batch automation from command line."
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Root directory containing sample folders.",
    )
    parser.add_argument(
        "--downsample-length",
        type=float,
        default=None,
        help="Pcot downsample length. If omitted, uses UI default (0.001).",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = get_default_batch_config()
    if args.downsample_length is not None:
        config["pcot"]["downsample_length"] = float(args.downsample_length)
    return config


def prepare_log_file(root_dir: str) -> str:
    base = Path(root_dir) / "automation_logs"
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = base / f"batch_automation_cli_{ts}.log"
    latest = base / "latest.log"
    log_path.write_text("", encoding="utf-8")
    latest.write_text(str(log_path) + "\n", encoding="utf-8")
    return str(log_path)


def append_log_file(log_file: str, msg: str) -> None:
    line = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line)


def main() -> int:
    args = parse_args()
    root_dir = str(Path(args.root_dir).expanduser().resolve())
    config = build_config(args)
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:
        print(
            "[error] PySide6 is required for FactPoints automation. Please use the env where UI can run.",
            flush=True,
        )
        return 3

    app = QApplication.instance() or QApplication(sys.argv)
    _ = app

    if not Path(root_dir).is_dir():
        print(f"[error] root directory not found: {root_dir}", flush=True)
        return 2

    log_file = prepare_log_file(root_dir)
    progress_state = {"last": None}

    def log(msg: str) -> None:
        print(msg, flush=True)
        append_log_file(log_file, msg)

    def progress(current: int, total: int) -> None:
        key = (current, total)
        if progress_state["last"] == key:
            return
        progress_state["last"] = key
        if total <= 0:
            msg = "[progress] 0/0 (0%)"
            print(msg, flush=True)
            append_log_file(log_file, msg)
            return
        pct = int((current / total) * 100)
        msg = f"[progress] {current}/{total} ({pct}%)"
        print(msg, flush=True)
        append_log_file(log_file, msg)

    log(f"[log] file: {log_file}")
    log(f"[run] root_dir={root_dir}")
    log(f"[config] pcot.downsample_length={config['pcot']['downsample_length']}")

    try:
        runner = BatchAutomationRunner(
            config=config,
            log=log,
            progress=progress,
            should_stop=lambda: False,
        )
        summary = runner.run(root_dir)
    except Exception:
        log("[fatal] batch run failed")
        log(traceback.format_exc())
        return 1

    log(f"[done] total={summary['total']} success={summary['success']} failed={summary['failed']}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
