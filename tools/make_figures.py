"""Figures for docs/report.md from the JSON files in docs/results/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("docs/results")
IMG = Path("docs/img")
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#e1e0d9"


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def counting(fps: float) -> None:
    rows = json.loads((RES / "counting.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150, sharey=False)
    names = [r["video"] + ("*" if r["unseen_hive"] else "") for r in rows]
    for ax, key in zip(axes, ("in", "out"), strict=True):
        gt = [r[f"gt_{key}"] for r in rows]
        pred = [next(x for x in r["runs"] if x["fps"] == fps)[f"pred_{key}"] for r in rows]
        x = range(len(rows))
        ax.bar([i - 0.2 for i in x], gt, width=0.38, color=ORANGE, label="manual tracks", edgecolor="white")
        ax.bar([i + 0.2 for i in x], pred, width=0.38, color=BLUE, label="Smart Hive", edgecolor="white")
        ax.set_xticks(list(x), names, rotation=20, ha="right")
        ax.set_title(f"Bees {key.upper()} per video", fontsize=10, color=INK, loc="left")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, f"* hive and camera never seen in training. Analysed at {fps:g} fps.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(IMG / "counting.png")
    plt.close(fig)


def health() -> None:
    h = json.loads((RES / "health.json").read_text())["healthy"]
    fig, ax = plt.subplots(figsize=(5.2, 2.6), dpi=150)
    labels = ["own baseline\n(Smart Hive)", "fixed thresholds"]
    vals = [h["check_days_per_season_180d"], h["fixed_threshold_alarm_days_per_season_180d"]]
    ax.barh(labels, vals, color=[BLUE, ORANGE], height=0.5)
    for i, v in enumerate(vals):
        ax.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=9, color=INK)
    ax.set_xlabel("false alarm days per 180-day season (healthy colonies)", fontsize=9, color=INK)
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID)
    fig.tight_layout()
    fig.savefig(IMG / "health_false_alarms.png")
    plt.close(fig)


def main() -> int:
    IMG.mkdir(parents=True, exist_ok=True)
    if (RES / "counting.json").exists():
        rows = json.loads((RES / "counting.json").read_text())
        counting(rows[0]["runs"][0]["fps"])
    if (RES / "health.json").exists():
        health()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
