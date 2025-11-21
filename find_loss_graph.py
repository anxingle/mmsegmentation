#!/usr/bin/env python3
"""Parse training logs and visualize loss changes."""

"""
  python find_loss_graph.py \
    --log an-Configs/train.log \
    --out an-Configs/loss_curve.png \
    --csv an-Configs/loss_values.csv \
    --smooth 5
"""

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence


LOG_PATTERN = re.compile(
    (
        r"(?P<ts>\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}).*?"
        r"Epoch\(train\)\s*\[(?P<epoch>\d+)\]\[(?P<iter>\d+)/(?P<total>\d+)\]"
        r".*?loss:\s*(?P<loss>[0-9]*\.?[0-9]+)"
    )
)


@dataclass
class LossRecord:
    epoch: int
    iter_in_epoch: int
    total_iters: int
    global_step: int
    loss: float
    timestamp: datetime


def parse_log(log_path: Path) -> List[LossRecord]:
    records: List[LossRecord] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            match = LOG_PATTERN.search(line)
            if not match:
                continue

            epoch = int(match.group("epoch"))
            iter_in_epoch = int(match.group("iter"))
            total_iters = int(match.group("total"))
            loss = float(match.group("loss"))
            ts = datetime.strptime(match.group("ts"), "%Y/%m/%d %H:%M:%S")
            global_step = (epoch - 1) * total_iters + iter_in_epoch

            records.append(
                LossRecord(
                    epoch=epoch,
                    iter_in_epoch=iter_in_epoch,
                    total_iters=total_iters,
                    global_step=global_step,
                    loss=loss,
                    timestamp=ts,
                )
            )

    return records


def moving_average(values: Sequence[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)

    window = min(window, len(values))
    averaged: List[float] = []
    current_sum = 0.0
    for idx, value in enumerate(values):
        current_sum += value
        if idx >= window:
            current_sum -= values[idx - window]
        averaged.append(current_sum / min(window, idx + 1))
    return averaged


def save_csv(records: Sequence[LossRecord], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "epoch",
                "iter_in_epoch",
                "total_iters",
                "global_step",
                "loss",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    r.timestamp.isoformat(sep=" "),
                    r.epoch,
                    r.iter_in_epoch,
                    r.total_iters,
                    r.global_step,
                    f"{r.loss:.6f}",
                ]
            )


def plot_curve(
    records: Sequence[LossRecord],
    output_path: Path,
    ma_window: int,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is not installed; skipping plot. "
            "Install it with `pip install matplotlib` to enable plotting.",
            file=sys.stderr,
        )
        return False

    steps = [r.global_step for r in records]
    losses = [r.loss for r in records]
    smoothed = moving_average(losses, ma_window)

    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, label="loss", linewidth=1.0)
    if ma_window > 1:
        plt.plot(
            steps,
            smoothed,
            label=f"loss (ma{ma_window})",
            linewidth=1.5,
        )
    plt.xlabel("Global iteration")
    plt.ylabel("Loss")
    plt.title("Training loss over time")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    return True


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent
    default_log = repo_root / "an-Configs" / "train.log"
    default_png = default_log.with_name("loss_curve.png")

    parser = argparse.ArgumentParser(
        description="Parse mmengine training logs and plot loss changes."
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=default_log,
        help=f"Path to training log file (default: {default_log})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_png,
        help=f"Output path for the loss curve image (default: {default_png})",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to save parsed loss values as CSV.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Moving average window size for smoothing (1 = no smoothing).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.log.exists():
        parser.error(f"Log file not found: {args.log}")

    records = parse_log(args.log)
    if not records:
        parser.error(f"No training loss entries found in: {args.log}")

    if args.csv:
        save_csv(records, args.csv)
        print(f"Saved raw loss data to {args.csv}")

    plotted = plot_curve(records, args.out, args.smooth)
    if plotted:
        print(
            f"Loss curve saved to {args.out} "
            f"(points: {len(records)}, smooth window: {args.smooth})"
        )
    else:
        print(
            f"Parsed {len(records)} loss points. "
            f"Re-run after installing matplotlib to generate a plot."
        )


if __name__ == "__main__":
    main()
