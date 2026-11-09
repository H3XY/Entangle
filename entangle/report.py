"""Results output: JSON, Markdown tables and charts in results/."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Fixed colour per model family, the same in every chart (colour follows the entity).
STYLE = {
    "quantum": dict(color="#2a78d6", marker="o"),          # the main quantum model (2-copy QNN / QCNN)
    "quantum_alt": dict(color="#eb6834", marker="s"),      # the weaker quantum model (1-copy QNN / generic ansatz)
    "classical_state": dict(color="#1baf7a", marker="^"),  # classical, full state vector
    "classical_snap": dict(color="#6250d6", marker="D"),   # classical, measurement snapshots
}
SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def path(name):
    os.makedirs(RESULTS, exist_ok=True)
    return os.path.join(RESULTS, name)


def save_json(name, data):
    with open(path(name), "w") as f:
        json.dump(data, f, indent=2)


def save_text(name, text):
    with open(path(name), "w") as f:
        f.write(text)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt(x):
    if x is None:
        return "–"
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def axes_grid(ncols, title=None, width=5.2, height=3.8):
    fig, axes = plt.subplots(1, ncols, figsize=(width * ncols, height), facecolor=SURFACE,
                             squeeze=False)
    axes = axes[0]
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    if title:
        fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    return fig, axes


def line(ax, x, y, label, family, err=None, dashed=False):
    st = STYLE[family]
    ax.plot(x, y, label=label, color=st["color"], marker=st["marker"], markersize=6,
            linewidth=2, linestyle="--" if dashed else "-",
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    if err is not None:
        lo = [a - b for a, b in zip(y, err)]
        hi = [a + b for a, b in zip(y, err)]
        ax.fill_between(x, lo, hi, color=st["color"], alpha=0.12, linewidth=0)


def finish(fig, axes, name, legend_ax=0):
    for ax in axes:
        ax.xaxis.label.set_color(INK2)
        ax.yaxis.label.set_color(INK2)
        ax.title.set_color(INK)
    h, lab = axes[legend_ax].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(4, len(lab)), frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path(name), dpi=150, facecolor=SURFACE)
    plt.close(fig)
