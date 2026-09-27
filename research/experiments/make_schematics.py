"""Conceptual schematic for the paper: fig_mechanism.pdf (router, admission rule, four outcomes).
Monochrome, so it cannot clash with the outcome colors of the data figures.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Serif font to match the LaTeX body text (paper style, not slide style).
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
})

_DIR = os.path.dirname(os.path.abspath(__file__))

INK, MUTED = "#1A1A1A", "#777777"


def _box(ax, xy, w, h, text, edge=INK, fs=10, bold=False, tc=INK, ls="-", lw=1.1):
    """Flat box: white fill, thin border, lightly rounded corners. Color lives in the
    border and (optionally) the text, never a filled background."""
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                linewidth=lw, edgecolor=edge, facecolor="#FFFFFF",
                                linestyle=ls, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", zorder=4, wrap=True)


def _arrow(ax, p0, p1, color=INK, lw=1.3, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13,
                                 linewidth=lw, color=color, zorder=2,
                                 shrinkA=2, shrinkB=2))


def _save(fig, name):
    """Save as vector PDF (used by the paper) and PNG (for quick preview)."""
    figdir = os.path.join(_DIR, "..", "figures")
    pdf = os.path.join(figdir, name + ".pdf")
    fig.savefig(pdf, facecolor="white", bbox_inches="tight")
    fig.savefig(os.path.join(figdir, name + ".png"), dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return pdf


def fig_mechanism():
    """Pipeline in the paper's notation. Monochrome, so it cannot clash with the outcome
    colors used by the data figures."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.axis("off")
    mid = 1.8
    _box(ax, (0.1, mid - 0.4), 1.5, 0.8, "item\n$x=(t,\\,o)$", edge=MUTED, fs=10)
    _box(ax, (2.3, mid - 0.5), 1.9, 1.0, "router\ntype $\\tau(t)$", fs=10)
    _box(ax, (5.0, mid - 0.5), 2.0, 1.0, "admission rule\n$A(\\tau(t),\\,\\lambda(o))$", fs=10)
    _arrow(ax, (1.6, mid), (2.3, mid))
    _arrow(ax, (4.2, mid), (5.0, mid))
    ax.text(3.25, mid + 0.68, "sees $t$ only", fontsize=8.5, color=MUTED, style="italic", ha="center")
    ax.text(6.0, mid + 0.68, "sees type and channel", fontsize=8.5, color=MUTED, style="italic", ha="center")
    outs = [("belief", "personal claim, trusted channel"),
            ("candidate evidence", "world fact; untrusted personal (variant)"),
            ("episodic", "event"),
            ("rejected", "untrusted personal; unsupported")]
    ys = [3.1, 2.25, 1.35, 0.5]
    for (label, why), y in zip(outs, ys):
        _box(ax, (7.6, y - 0.36), 2.35, 0.72, "", lw=1.4 if label == "belief" else 1.0)
        ax.text(8.775, y + 0.1, label, ha="center", va="center", fontsize=9.5, color=INK, zorder=4)
        ax.text(8.775, y - 0.17, why, ha="center", va="center", fontsize=6.8, color=MUTED,
                style="italic", zorder=4)
        _arrow(ax, (7.0, mid), (7.6, y), color=MUTED, lw=1.0)
    fig.tight_layout()
    return _save(fig, "fig_mechanism")


if __name__ == "__main__":
    print("wrote", fig_mechanism())
