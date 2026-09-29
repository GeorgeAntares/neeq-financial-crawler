"""Print-style theme for analysis figures.

Language is a tearsheet / 一页纸: warm paper, one ink, one firm blue,
one alert red. Sentence titles. No rainbow, gauges, or 3-D.
"""
from __future__ import annotations

import os

BG = "#FFFEF9"
SURFACE = "#F3EFE6"
INK = "#1C1917"
MUTED = "#78716C"
RULE = "#E4DDD2"
FIRM = "#1F4E79"
PEER = "#A8A29E"
UP = "#3F6F4A"
DOWN = "#B42318"
FLAG = "#C2410C"
NA = "#D6D0C6"
ACCENT_2 = "#7A5C2E"
ACCENT_3 = "#4F6F8F"

GROUP_COLORS = {
    "制造": FIRM,
    "软件信息": UP,
    "其他": PEER,
}

FONT_STACK = ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans"]


def use_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.sans-serif": FONT_STACK,
            "font.family": "sans-serif",
            "axes.unicode_minus": False,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.facecolor": BG,
            "axes.edgecolor": RULE,
            "axes.labelcolor": MUTED,
            "axes.titlesize": 10,
            "axes.titleweight": "regular",
            "axes.titlecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": INK,
            "axes.grid": False,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "figure.dpi": 140,
        }
    )
    return matplotlib, plt


def restyle(ax):
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(RULE)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(length=0, colors=MUTED, labelsize=8)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)


def hide_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(BG)


def set_panel(ax, title=None, kicker=None, ylabel=None):
    restyle(ax)
    if kicker:
        ax.text(
            0.0,
            1.14,
            kicker,
            transform=ax.transAxes,
            fontsize=7.5,
            color=MUTED,
            ha="left",
            va="bottom",
        )
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=10, fontweight="regular")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)


def draw_empty(ax, message="缺科目", kicker=None, title=None):
    hide_axes(ax)
    if kicker:
        ax.text(0.0, 1.14, kicker, transform=ax.transAxes, fontsize=7.5, color=MUTED, ha="left")
    if title:
        ax.text(0.0, 1.02, title, transform=ax.transAxes, fontsize=10, color=INK, ha="left")
    ax.text(0.5, 0.48, message, ha="center", va="center", color=MUTED, fontsize=10, transform=ax.transAxes)


def group_color(name, fallback=PEER):
    return GROUP_COLORS.get(name, fallback)


def save_fig(fig, path, dpi=160):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    fig.savefig(path, dpi=dpi, facecolor=BG, edgecolor="none", bbox_inches="tight", pad_inches=0.28)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path
