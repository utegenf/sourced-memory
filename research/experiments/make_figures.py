"""Generate the paper's data figures from the ablation results (same loader as the numbers).

Deterministic: reads the saved results, writes vector PDFs to research/figures/.
  fig_core.pdf          headline: fabrication accepted (TARGET) and genuine update learned
                        (CONTROL), every agent x every model
  fig_outcome_dist.pdf  trusted / candidate / absent composition, primary model, 6 conditions
  fig_pe_validation.pdf schema-fit manipulation check vs authored ground truth, primary model

Palette = Okabe-Ito. The four model hues sit in the CVD 6-8 band, so every model also has its
own marker shape; the paper's tables give the exact values.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from make_paper_numbers import AGENT_LABEL, MODELS, load_inject, load_models, router_confidences

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
})

_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_DIR, "..", "figures")
PRIMARY = "opus5"

OI = {"orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73", "blue": "#0072B2",
      "vermillion": "#D55E00", "purple": "#CC79A7"}
MODEL_STYLE = {"opus5": (OI["blue"], "o"), "deepseek": (OI["orange"], "s"),
               "qwenL": (OI["green"], "^"), "qwenS": (OI["purple"], "D")}
OUTCOME_COLOR = {"trusted": OI["vermillion"], "candidate": OI["orange"], "absent": OI["blue"]}
INK, MUTED, GRID, SURFACE = "#222222", "#888888", "#EEEEEE", "#FFFFFF"

CONTENT_ONLY = ["rag", "reflection", "schema_no_prov", "schema_conf", "content_judge"]
ORIGIN_AWARE = ["llm_prov", "schema_prov", "schema_prov_cand"]


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, length=0)
    ax.set_axisbelow(True)


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_core(all_data):
    agents = CONTENT_ONLY + ORIGIN_AWARE
    models = [m for m in MODELS if m[0] in all_data]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), sharey=True)
    panels = [("TARGET", "Fabrication from untrusted channel\naccepted as belief (%, lower is safer)"),
              ("CONTROL", "Genuine trusted update\nlearned as belief (%, higher is better)")]
    offsets = [(-1.5 + i) * 0.16 for i in range(len(models))]
    for ax, (cond, xlabel) in zip(axes, panels):
        for off, (mkey, _, _) in zip(offsets, models):
            color, marker = MODEL_STYLE[mkey]
            xs = [100 * all_data[mkey]["summary"][cond][a]["trusted"]["frac"] for a in agents]
            ys = [i + off for i in range(len(agents))]
            ax.scatter(xs, ys, s=38, color=color, marker=marker, edgecolors=SURFACE,
                       linewidths=1.0, zorder=3)
        ax.axhline(len(CONTENT_ONLY) - 0.5, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.set_xlim(-4, 104)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel(xlabel, fontsize=9, color=INK)
        ax.grid(axis="x", color=GRID, zorder=0)
        _style(ax)
    axes[0].set_yticks(range(len(agents)))
    axes[0].set_yticklabels([AGENT_LABEL[a] for a in agents], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].text(50, len(CONTENT_ONLY) - 0.6, "content-only", fontsize=8, color=MUTED,
                 ha="center", va="bottom")
    axes[0].text(50, len(CONTENT_ONLY) - 0.4, "origin-aware", fontsize=8, color=MUTED,
                 ha="center", va="top")
    handles = [Line2D([], [], linestyle="none", marker=MODEL_STYLE[k][1], color=MODEL_STYLE[k][0],
                      markersize=7, label=name) for k, name, _ in models]
    fig.legend(handles=handles, frameon=False, fontsize=9, ncol=len(models),
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save(fig, "fig_core.pdf")


def fig_dist(all_data):
    d = all_data[PRIMARY]["summary"]
    agents = ["schema_no_prov", "content_judge", "llm_prov", "schema_prov", "schema_prov_cand"]
    conds = [("TARGET", "false personal / untrusted\n(want: not trusted)"),
             ("IRREDUCIBLE", "false personal / trusted\n(enters under any origin rule)"),
             ("CONTROL", "true reversal / trusted\n(want: trusted)"),
             ("AUTHORIZATION", "true reversal / untrusted\n(want: not trusted)"),
             ("PARANOIA", "true world fact / untrusted\n(want: candidate)"),
             ("SPOOF", "fabrication quoted as the user\n/ untrusted (want: not trusted)")]
    shown = {"PARANOIA": "WORLD-FACT", "SPOOF": "QUOTED"}
    n = sum(d["TARGET"]["schema_prov"][o]["k"] for o in OUTCOME_COLOR)
    fig, axes2d = plt.subplots(2, 3, figsize=(10.5, 5.2), sharey=True)
    for ax, (ck, sub) in zip(axes2d.flatten(), conds):
        left = [0] * len(agents)
        for oc in OUTCOME_COLOR:
            vals = [d[ck][a][oc]["k"] for a in agents]
            ax.barh(range(len(agents)), vals, left=left, color=OUTCOME_COLOR[oc], height=0.62,
                    edgecolor=SURFACE, linewidth=2, zorder=3)
            left = [l + v for l, v in zip(left, vals)]
        ax.set_yticks(range(len(agents)))
        ax.set_yticklabels([AGENT_LABEL[a] for a in agents], fontsize=8.5)
        ax.set_xlim(0, n)
        ax.set_xticks([0, n // 2, n])
        ax.set_title(f"{shown.get(ck, ck)}\n{sub}", fontsize=8.5, color=INK)
        ax.grid(axis="x", color=GRID, zorder=0)
        _style(ax)
    axes2d[0][0].invert_yaxis()  # shared y: invert once, not per panel
    fig.supxlabel(f"count out of n={n} per cell", fontsize=9, color=INK)
    fig.legend(handles=[Patch(color=OUTCOME_COLOR[o], label=o) for o in OUTCOME_COLOR],
               frameon=False, fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "fig_outcome_dist.pdf")


def fig_pe(all_data):
    pe = all_data[PRIMARY]["pe_validation"]
    items = [("schema_fit_false", "fits schema\n(false)", True),
             ("conflict_false", "conflicts\n(false)", False),
             ("reversal_true", "true reversal\n(conflicts)", False),
             ("legit_untrusted_true", "world fact\n(no schema)", True)]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    vals = [pe[k]["mean"] for k, _, _ in items]
    colors = [OI["blue"] if hi else OI["skyblue"] for _, _, hi in items]
    ax.bar(range(len(items)), vals, width=0.6, color=colors, zorder=3)
    for xi, v in enumerate(vals):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", color=INK, fontsize=10)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([lbl for _, lbl, _ in items], fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Judged schema consistency\n(0 = conflicts, 1 = fits)", fontsize=9, color=INK)
    ax.grid(axis="y", color=GRID, zorder=0)
    _style(ax)
    ax.legend(handles=[Patch(color=OI["blue"], label="authored: high fit"),
                       Patch(color=OI["skyblue"], label="authored: low fit")],
              frameon=False, fontsize=8, loc="upper center", ncol=2)
    fig.tight_layout()
    return _save(fig, "fig_pe_validation.pdf")


INJECT_PANELS = [("INJECT_SUMMARY", "Third-person attribution"), ("INJECT_FORGED", "Forged channel tag"),
                 ("INJECT_OVERRIDE", "Explicit instruction")]
INJECT_AXIS = [("llm_prov", "Prompted"), ("llm_prov_spot", "Prompted\n+ Spotlighting"),
               ("llm_prov_rule", "Prompted\n+ rule"), ("schema_prov", "Structural")]


def fig_inject(inj):
    models = [m for m in MODELS if m[0] in inj]
    fig, axes = plt.subplots(1, len(INJECT_PANELS), figsize=(10.0, 3.4), sharey=True)
    offsets = [(-1.5 + i) * 0.1 for i in range(len(models))]
    for ax, (cond, title) in zip(axes, INJECT_PANELS):
        for off, (mkey, _, _) in zip(offsets, models):
            color, marker = MODEL_STYLE[mkey]
            ys = [100 * inj[mkey]["summary"][cond][a]["trusted"]["frac"] for a, _ in INJECT_AXIS]
            xs = [i + off for i in range(len(INJECT_AXIS))]
            ax.plot(xs, ys, color=color, linewidth=1.0, alpha=0.55, zorder=2)
            ax.scatter(xs, ys, s=38, color=color, marker=marker, edgecolors=SURFACE, linewidths=1.0,
                       zorder=3)
        ax.axvline(len(INJECT_AXIS) - 1.5, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.set_xticks(range(len(INJECT_AXIS)))
        ax.set_xticklabels([lbl for _, lbl in INJECT_AXIS], fontsize=8.5)
        ax.set_ylim(-5, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.grid(axis="y", color=GRID, zorder=0)
        _style(ax)
    axes[0].set_ylabel("Injected fabrication stored\nas a user belief (%)", fontsize=9, color=INK)
    handles = [Line2D([], [], linestyle="none", marker=MODEL_STYLE[k][1], color=MODEL_STYLE[k][0],
                      markersize=7, label=name) for k, name, _ in models]
    fig.legend(handles=handles, frameon=False, fontsize=9, ncol=len(models),
               loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return _save(fig, "fig_inject.pdf")


def fig_threshold(models_present):
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for mkey, mname, tag in MODELS:
        if mkey not in models_present:
            continue
        conf = router_confidences(tag)
        tgt, ctl = conf["TARGET"], conf["CONTROL"]
        taus = sorted(set(tgt) | set(ctl) | {0.0, 1.01})
        xs = [100 * sum(c >= t for c in ctl) / len(ctl) for t in taus]
        ys = [100 * sum(c >= t for c in tgt) / len(tgt) for t in taus]
        color, marker = MODEL_STYLE[mkey]
        ax.plot(xs, ys, color=color, linewidth=1.6, zorder=2, label=mname)
        ax.scatter(xs, ys, s=16, color=color, marker=marker, zorder=3)
        if mkey == "opus5":
            for t in (0.80, 0.85, 0.90):
                x = 100 * sum(c >= t for c in ctl) / len(ctl)
                y = 100 * sum(c >= t for c in tgt) / len(tgt)
                ax.annotate(f"$\\tau={t:.2f}$", (x, y), textcoords="offset points", xytext=(6, 4),
                            fontsize=8, color=INK)
    ax.scatter([100], [0], s=160, marker="*", color=INK, zorder=4)
    ax.annotate("origin-aware\n(every model)", (100, 0), textcoords="offset points", xytext=(-78, 8),
                fontsize=8.5, color=INK)
    ax.set_xlim(-3, 106)
    ax.set_ylim(-5, 106)
    ax.set_xlabel("Genuine updates still learned (%)", fontsize=9, color=INK)
    ax.set_ylabel("Fabrications still admitted (%)", fontsize=9, color=INK)
    ax.grid(color=GRID, zorder=0)
    _style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    return _save(fig, "fig_threshold.pdf")


# Strength of the in-text claim of user authorship; the real channel label is untrusted throughout.
CUE_LADDER = [("TARGET", "none"), ("INJECT_SUMMARY", "third-person\nattribution"),
              ("SPOOF", "quotation\nof the user"), ("INJECT_FORGED", "forged\nchannel tag"),
              ("INJECT_OVERRIDE", "explicit\ninstruction")]


def _cue_rate(main, inj, mkey, cond, agent):
    src = inj[mkey] if cond.startswith("INJECT") else main[mkey]
    return 100 * src["summary"][cond][agent]["trusted"]["frac"]


def fig_cueladder(main, inj):
    models = [m for m in MODELS if m[0] in inj and m[0] in main]
    panels = [("llm_prov", "Given the channel label (prompted manager)"),
              ("content_judge", "Not given the channel (skeptical manager)")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    offsets = [(-1.5 + i) * 0.07 for i in range(len(models))]
    for ax, (agent, title) in zip(axes, panels):
        for off, (mkey, _, _) in zip(offsets, models):
            color, marker = MODEL_STYLE[mkey]
            ys = [_cue_rate(main, inj, mkey, c, agent) for c, _ in CUE_LADDER]
            xs = [i + off for i in range(len(CUE_LADDER))]
            ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.7, zorder=2)
            ax.scatter(xs, ys, s=36, color=color, marker=marker, edgecolors=SURFACE, linewidths=1.0,
                       zorder=3)
        ax.set_xticks(range(len(CUE_LADDER)))
        ax.set_xticklabels([lbl for _, lbl in CUE_LADDER], fontsize=8.5)
        ax.set_ylim(-5, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.set_xlabel("in-text claim of user authorship (label is always untrusted)", fontsize=8.5,
                      color=INK)
        ax.grid(axis="y", color=GRID, zorder=0)
        _style(ax)
    axes[0].set_ylabel("Fabrication stored as a\nbelief about the user (%)", fontsize=9, color=INK)
    handles = [Line2D([], [], linestyle="none", marker=MODEL_STYLE[k][1], color=MODEL_STYLE[k][0],
                      markersize=7, label=name) for k, name, _ in models]
    fig.legend(handles=handles, frameon=False, fontsize=9, ncol=len(models),
               loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return _save(fig, "fig_cueladder.pdf")


if __name__ == "__main__":
    data, missing = load_models()
    if missing:
        raise SystemExit(f"missing result files: {missing}")
    inj, _, _ = load_inject()
    for f in (fig_core(data), fig_dist(data), fig_pe(data), fig_inject(inj), fig_threshold(data),
              fig_cueladder(data, inj)):
        print("wrote", os.path.relpath(f))
