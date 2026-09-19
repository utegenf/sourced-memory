"""Generate publication figures from the source-aware ablation (results/prov_ablation_sonnet45.json).

Deterministic: reads the saved result, writes PNGs. Re-run any time the ablation is re-run.
Palette = Okabe-Ito (colorblind-safe categorical). Marks thin, direct labels, legend present,
recessive axes. Three figures:
  fig_core_target.png    - headline: TARGET acceptance rate by agent + Wilson CI (the result)
  fig_outcome_dist.png   - full trusted/candidate/absent composition, 4 conditions x 5 agents
  fig_pe_validation.png  - PE controlled-variable check vs authored ground truth
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Match the paper's serif body font (Times / Computer Modern look) so figures don't read as slides.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
})

_DIR = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(_DIR, "..", "results", "prov_ablation_sonnet45.json")))["result"]
S = D["summary"]
PE = D["pe_validation"]

# --- Okabe-Ito (CVD-safe) ---------------------------------------------------
OI = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
      "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
      "vermillion": "#D55E00", "purple": "#CC79A7"}
# Outcome = 3 states (not good/bad status — semantics flip by condition), fixed order.
OUTCOME_COLOR = {"trusted": OI["vermillion"], "candidate": OI["orange"], "absent": OI["blue"]}
INK, MUTED = "#222222", "#888888"

AGENTS = ["rag", "reflection", "schema_no_prov", "schema_conf", "schema_prov"]
AGENT_LABEL = {"rag": "RAG", "reflection": "Reflection", "schema_no_prov": "schema\nno-prov",
               "schema_conf": "schema\nconf", "schema_prov": "schema\nprov"}

# Condition keys (long) -> short titles
CONDS = {
    "plausible-false personal / UNTRUSTED (TARGET: reject)": "TARGET\nfalse-personal / untrusted\n(want: reject)",
    "plausible-false personal / trusted (irreducible: enters)": "IRREDUCIBLE\nfalse-personal / trusted\n(enters)",
    "TRUE reversal (personal) / trusted (CONTROL: enters=learning)": "CONTROL\ntrue reversal / trusted\n(want: enters)",
    "TRUE world-fact / UNTRUSTED (PARANOIA: candidate, not belief)": "PARANOIA\ntrue world-fact / untrusted\n(want: candidate)",
}
TARGET = "plausible-false personal / UNTRUSTED (TARGET: reject)"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, length=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# FIG 1 — headline: TARGET trusted-belief acceptance rate by agent, Wilson CI
# ---------------------------------------------------------------------------
def fig_core():
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    fracs, los, his = [], [], []
    for ag in AGENTS:
        t = S[TARGET][ag]["trusted"]
        fracs.append(t["frac"] * 100)
        lo, hi = t["ci"]
        los.append(max(0.0, (t["frac"] - lo) * 100))
        his.append(max(0.0, (hi - t["frac"]) * 100))
    x = range(len(AGENTS))
    # source-blind content methods in vermillion (fail), source-aware in blue (resolves)
    colors = [OI["vermillion"]] * 4 + [OI["blue"]]
    bars = ax.bar(x, fracs, width=0.62, color=colors, zorder=3)
    ax.errorbar(x, fracs, yerr=[los, his], fmt="none", ecolor=INK, elinewidth=1.4,
                capsize=4, zorder=4)
    # value labels ABOVE the upper CI whisker (never inside the bar / over the cap).
    # For a 0% bar the label sits at y=0 by default which reads as a rendering artifact;
    # lift it to a visible height so the 0% contrast with the ~100% bars is unmissable.
    for xi, f, hi in zip(x, fracs, his):
        y_label = max(f + hi + 3, 8)
        ax.text(xi, y_label, f"{f:.0f}%", ha="center", va="bottom",
                color=INK, fontsize=10, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([AGENT_LABEL[a] for a in AGENTS], fontsize=9)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Plausible lie accepted as\ntrusted belief (%, lower=safer)", fontsize=9, color=INK)
    ax.set_title("Content-only memory assimilates the plausible lie; origin resolves it",
                 fontsize=11, color=INK, pad=26)
    ax.grid(axis="y", color="#EEEEEE", zorder=0)
    _style(ax)
    # legend ABOVE the axes (outside the plot area) so it never collides with the ~100% bars
    ax.legend(handles=[Patch(color=OI["vermillion"], label="content-only (source-blind)"),
                       Patch(color=OI["blue"], label="source-aware")],
              frameon=False, fontsize=8, loc="lower center", ncol=2,
              bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    out = os.path.join(_DIR, "..", "figures", "fig_core_target.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# FIG 2 — full composition: 4 condition panels, 5 stacked bars each
# ---------------------------------------------------------------------------
def fig_dist():
    # schema agents only (the causal comparison); RAG/Reflection reference points live in fig_core.
    AG = ["schema_no_prov", "schema_conf", "schema_prov"]
    # derive N from the data (total outcomes per cell) so the axis is never hardcoded
    N = sum(S[TARGET]["schema_no_prov"][o]["k"] for o in ("trusted", "candidate", "absent"))
    half = N // 2
    fig, axes2d = plt.subplots(2, 2, figsize=(9.5, 5.4), sharey=True)
    axes = axes2d.flatten()
    order = ["absent", "candidate", "trusted"]  # bottom->top
    for ax, (ck, title) in zip(axes, CONDS.items()):
        y = range(len(AG))
        left = [0] * len(AG)
        for oc in order:
            vals = [S[ck][ag][oc]["k"] for ag in AG]
            ax.barh(y, vals, left=left, color=OUTCOME_COLOR[oc], height=0.6, zorder=3)
            left = [l + v for l, v in zip(left, vals)]
        ax.set_yticks(list(y))
        ax.set_yticklabels([AGENT_LABEL[a].replace("\n", " ") for a in AG], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, N)
        ax.set_xticks([0, half, N])
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.grid(axis="x", color="#EEEEEE", zorder=0)
        _style(ax)
    axes[0].set_ylabel("")
    fig.supxlabel(f"count out of n={N}", fontsize=9, color=INK, y=0.02)
    handles = [Patch(color=OUTCOME_COLOR[o], label=o) for o in ["trusted", "candidate", "absent"]]
    fig.legend(handles=handles, frameon=False, fontsize=9, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("")
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    out = os.path.join(_DIR, "..", "figures", "fig_outcome_dist.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# FIG 3 — PE validation (controlled variable vs authored ground truth)
# ---------------------------------------------------------------------------
def fig_pe():
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    items = [("schema_fit_false", "fits schema\n(false)", "HIGH"),
             ("conflict_false", "conflicts\n(false)", "LOW"),
             ("reversal_true", "true reversal\n(conflicts)", "LOW"),
             ("legit_untrusted_true", "world fact\n(no schema)", "HIGH")]
    x = range(len(items))
    vals = [PE[k] for k, _, _ in items]
    # color by authored ground-truth direction (sequential-ish: HIGH=dark, LOW=light of one hue)
    colors = [OI["blue"] if gt == "HIGH" else OI["skyblue"] for _, _, gt in items]
    ax.bar(x, vals, width=0.6, color=colors, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([lbl for _, lbl, _ in items], fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Judged schema consistency\n(0=conflicts, 1=fits)", fontsize=9, color=INK)
    ax.set_title("Prediction error behaved as a controlled variable", fontsize=11, color=INK, pad=10)
    ax.grid(axis="y", color="#EEEEEE", zorder=0)
    _style(ax)
    ax.legend(handles=[Patch(color=OI["blue"], label="ground truth: HIGH fit"),
                       Patch(color=OI["skyblue"], label="ground truth: LOW fit")],
              frameon=False, fontsize=8, loc="upper center", ncol=2)
    fig.tight_layout()
    out = os.path.join(_DIR, "..", "figures", "fig_pe_validation.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for f in (fig_core(), fig_dist(), fig_pe()):
        print("wrote", f)