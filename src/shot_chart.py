from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from session import ShotRecord


def _draw_half_court(ax) -> None:
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 47)
    ax.set_aspect("equal")

    court = patches.Rectangle((0, 0), 50, 47, linewidth=2, edgecolor="black", facecolor="#f6f1e7")
    paint = patches.Rectangle((17, 0), 16, 19, linewidth=2, edgecolor="black", facecolor="none")
    hoop = patches.Circle((25, 5.25), 0.75, linewidth=2, edgecolor="orange", facecolor="none")
    free_throw = patches.Circle((25, 19), 6, linewidth=2, edgecolor="black", facecolor="none")
    three_arc = patches.Arc((25, 5.25), 47.5, 47.5, theta1=22, theta2=158, linewidth=2)

    ax.add_patch(court)
    ax.add_patch(paint)
    ax.add_patch(hoop)
    ax.add_patch(free_throw)
    ax.add_patch(three_arc)
    ax.plot([3, 3], [0, 14], color="black", linewidth=2)
    ax.plot([47, 47], [0, 14], color="black", linewidth=2)

    ax.set_title("Workout Shot Chart")
    ax.axis("off")


def save_shot_chart(records: Iterable[ShotRecord], out_path: Path) -> None:
    makes_x, makes_y, misses_x, misses_y = [], [], [], []

    for r in records:
        if r.court_x_ft is None or r.court_y_ft is None:
            continue
        if r.result == "make":
            makes_x.append(r.court_x_ft)
            makes_y.append(r.court_y_ft)
        else:
            misses_x.append(r.court_x_ft)
            misses_y.append(r.court_y_ft)

    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_half_court(ax)

    ax.scatter(makes_x, makes_y, c="#2e7d32", marker="o", s=70, label="Make", edgecolors="white", linewidths=0.8)
    ax.scatter(misses_x, misses_y, c="#c62828", marker="x", s=70, label="Miss")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
