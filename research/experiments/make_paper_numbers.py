"""Generate every number and results table used in the paper from the result JSON files.

Nothing in the paper should be typed by hand. This script writes:
  paper_generated/numbers.tex        one LaTeX value per result, looked up as \\R{key}
  paper_generated/tab_main.tex       main ablation table (primary model)
  paper_generated/tab_crossmodel.tex cross-model table
  paper_generated/tab_judge.tex      judge-agreement table
  paper_generated/numbers.md         every key and value, human-readable (for review)

\\R{key} stops the LaTeX build with an error if the key does not exist, so a stale or mistyped
number cannot reach the PDF. Keys look like:
  opus5.TARGET.schema_prov.trusted.k      count
  opus5.TARGET.schema_prov.trusted.n      trials in the cell
  opus5.TARGET.schema_prov.trusted.pct    percentage, 1 decimal
  opus5.TARGET.schema_prov.trusted.cilo   item-level Wilson 95% CI low (percent, 1 decimal):
                                          repeats averaged per item, n = distinct items
  opus5.TARGET.schema_prov.trusted.cihi   item-level Wilson 95% CI high
  opus5.TARGET.schema_prov.items          distinct items in the cell
  opus5.TARGET.schema_prov.repmin         lowest per-repeat trusted percentage
  opus5.TARGET.schema_prov.repmax         highest per-repeat trusted percentage

Usage: python3 make_paper_numbers.py [--primary opus5]
"""
import argparse
import json
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_DIR, "..", "results")
OUT = os.path.join(_DIR, "..", "..", "paper_generated")

MODELS = [  # (key used in \R, display name, ablation result tag)
    ("opus5", "Claude Opus 5", "opus5_v4"),
    ("deepseek", "DeepSeek V3.2", "deepseek_v32_v4"),
    ("qwenL", "Qwen3-235B", "qwen3_235b_v4"),
    ("qwenS", "Qwen3-32B", "qwen3_32b_v4"),
]
CONDITIONS = ["TARGET", "IRREDUCIBLE", "CONTROL", "AUTHORIZATION", "PARANOIA", "SPOOF"]
# Display names used in tables and figures (result keys keep the run's condition names).
COND_LABEL = {"TARGET": "TARGET", "IRREDUCIBLE": "IRREDUCIBLE", "CONTROL": "CONTROL",
              "AUTHORIZATION": "AUTHORIZATION", "PARANOIA": "WORLD-FACT", "SPOOF": "QUOTED"}
# model key -> result tag of a reflection-only rerun judged by the common judge (GPT-6 Astra)
REFLECTION_OVERRIDE = {"opus5": "opus5_refl_g6"}
AGENTS = ["rag", "reflection", "schema_no_prov", "schema_conf", "content_judge", "llm_prov",
          "schema_prov", "schema_prov_cand"]
# model key -> run tag of the prompt-injection ablation (opt-in INJECT_* conditions)
INJECT_TAGS = {"opus5": "inject_opus5", "deepseek": "inject_deepseek_v32",
               "qwenL": "inject_qwen3_235b", "qwenS": "inject_qwen3_32b"}
INJECT_CONDITIONS = ["INJECT_OVERRIDE", "INJECT_FORGED", "INJECT_SUMMARY"]
INJECT_AGENTS = ["content_judge", "llm_prov", "llm_prov_spot", "llm_prov_rule", "schema_prov"]
# model key -> run tag of the prompted manager with standard injection defenses (same conditions + CONTROL)
DEFENSE_TAGS = {"opus5": "defense_opus5", "deepseek": "defense_deepseek_v32",
                "qwenL": "defense_qwen3_235b", "qwenS": "defense_qwen3_32b"}
DEFENSE_AGENTS = ["llm_prov_spot", "llm_prov_rule"]
AGENT_LABEL = {"rag": "RAG", "reflection": "Reflection", "schema_no_prov": "Source-blind",
               "schema_conf": "Confidence", "content_judge": "Skeptical LLM",
               "llm_prov": "Prompted provenance", "schema_prov": "Structural",
               "schema_prov_cand": "Structural + candidate",
               "llm_prov_spot": "Prompted + Spotlighting", "llm_prov_rule": "Prompted + rule"}
JUDGE_LABEL = {"judge": "gpt-oss-120b", "judge_oss": "gpt-oss-120b", "gpt6_astra": "GPT-6 Astra"}


def pct(x):
    return "--" if x is None else f"{100 * x:.1f}"


def wilson(k, n, z=1.96):
    """Wilson interval; k may be fractional (a sum of per-item means)."""
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def item_cis(tags, keyfn):
    """Item-clustered Wilson CIs. Repeats of the same item are not independent draws, so each
    item's repeats are averaged and the interval uses n = number of distinct items.
    tags: run dirs to read; keyfn(record) -> cell key or None. Returns {cell: {outcome: (lo, hi)}}
    and {cell: number of items}."""
    import glob
    from collections import defaultdict
    per_item = defaultdict(lambda: defaultdict(list))
    for tag in tags:
        for f in glob.glob(os.path.join(RESULTS, "runs", tag, "*_r*.json")):
            for r in json.load(open(f))["records"]:
                cell = keyfn(r)
                if cell is not None:
                    per_item[cell][(r["persona"], r["item_index"])].append(r["outcome"])
    cis, items = {}, {}
    for cell, by_item in per_item.items():
        n = len(by_item)
        items[cell] = n
        cis[cell] = {o: wilson(sum(outs.count(o) / len(outs) for outs in by_item.values()), n)
                     for o in ("trusted", "candidate", "absent")}
    return cis, items


def load(name):
    path = os.path.join(RESULTS, name)
    return json.load(open(path)) if os.path.exists(path) else None


class Values:
    def __init__(self):
        self.v = {}

    def put(self, key, value):
        if key in self.v:
            raise KeyError(f"duplicate key {key}")
        self.v[key] = str(value)


def add_ablation(vals, mkey, data, tag):
    meta = data["meta"]
    override = REFLECTION_OVERRIDE.get(mkey)
    cis, items = item_cis([tag], lambda r: None if override and r["agent"] == "reflection"
                          else (r["condition"], r["agent"]))
    if override:  # reflection cells come from the common-judge rerun only
        rcis, ritems = item_cis([override], lambda r: (r["condition"], r["agent"])
                                if r["agent"] == "reflection" else None)
        cis.update(rcis)
        items.update(ritems)
    vals.put(f"{mkey}.jobs", meta["jobs_complete"])
    vals.put(f"{mkey}.repeats", len(data["repeats"]))
    vals.put(f"{mkey}.personas", data["n_personas"])
    vals.put(f"{mkey}.records", data["n_records"])
    vals.put(f"{mkey}.judge", JUDGE_LABEL.get(meta["judge_model"], meta["judge_model"]))
    stats = meta.get("client_stats_this_process") or {}
    vals.put(f"{mkey}.variantfailures", stats.get("variant_failures", 0))
    for c in CONDITIONS:
        for a in AGENTS:
            cell = data["summary"][c][a]
            for outcome in ("trusted", "candidate", "absent"):
                d = cell[outcome]
                base = f"{mkey}.{c}.{a}.{outcome}"
                vals.put(f"{base}.k", d["k"])
                vals.put(f"{base}.n", d["n"])
                vals.put(f"{base}.pct", pct(d["frac"]))
                lo, hi = cis[(c, a)][outcome]
                vals.put(f"{base}.cilo", pct(lo))
                vals.put(f"{base}.cihi", pct(hi))
            vals.put(f"{mkey}.{c}.{a}.items", items[(c, a)])
            rep = data["per_repeat"][c][a]
            vals.put(f"{mkey}.{c}.{a}.repmin", pct(rep["min"]))
            vals.put(f"{mkey}.{c}.{a}.repmax", pct(rep["max"]))
    for pair, row in data["separation"].items():
        for a, s in row.items():
            vals.put(f"{mkey}.sep.{pair}.{a}", pct(s["pooled"]))
    for ct, s in data["pe_validation"].items():
        vals.put(f"{mkey}.pe.{ct}", f"{s['mean']:.2f}")
    for c, g in data["gate_classification_no_prov"].items():
        vals.put(f"{mkey}.gate.{c}.n", g["n"])
        vals.put(f"{mkey}.gate.{c}.supported", g["supported_true"])
        vals.put(f"{mkey}.gate.{c}.confmean",
                 "--" if g["confidence_mean"] is None else f"{g['confidence_mean']:.2f}")
        for cat, k in g["categories"].items():
            vals.put(f"{mkey}.gate.{c}.{cat}", k)


def add_judge(vals, data):
    vals.put("judge.n", data["n"])
    vals.put("judge.agreement", pct(data["agreement"]))
    vals.put("judge.kappa", f"{data['cohen_kappa']:.2f}")
    for j, r in data["trusted_rate"].items():
        vals.put(f"judge.trustedrate.{j}", pct(r))
    for c, row in data["per_condition"].items():
        vals.put(f"judge.{c}.n", row["n"])
        vals.put(f"judge.{c}.agreement", pct(row["agreement"]))
        for k, v in row.items():
            if k.startswith("trusted_"):
                vals.put(f"judge.{c}.{k.replace('trusted_', 'trusted.')}", v)
    vals.put("judge.disagreements", len(data["disagreements"]))


def add_mem0(vals, data, tag):
    vals.put("mem0.jobs", data["meta"]["jobs_complete"])
    vals.put("mem0.repeats", len(data["meta"]["repeats"]))
    vals.put("mem0.judge", JUDGE_LABEL.get(data["meta"]["judge_model"], data["meta"]["judge_model"]))
    cis, items = item_cis([tag], lambda r: r["condition"])
    for c, s in data["summary"].items():
        vals.put(f"mem0.{c}.k", s["trusted"])
        vals.put(f"mem0.{c}.n", s["n"])
        vals.put(f"mem0.{c}.pct", pct(s["frac"]))
        vals.put(f"mem0.{c}.cilo", pct(cis[c]["trusted"][0]))
        vals.put(f"mem0.{c}.cihi", pct(cis[c]["trusted"][1]))
        vals.put(f"mem0.{c}.items", items[c])


TAUS = (0.50, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def router_confidences(tag):
    """Per-condition router confidences recorded for the source-blind schema arm."""
    import glob
    conf = {c: [] for c in CONDITIONS}
    for f in glob.glob(os.path.join(RESULTS, "runs", tag, "*_r*.json")):
        for r in json.load(open(f))["records"]:
            if r["agent"] == "schema_no_prov" and r.get("gate_confidence") is not None:
                conf[r["condition"]].append(r["gate_confidence"])
    return conf


def add_confidence(vals, mkey, tag):
    """Router self-confidence per condition and the outcome of every threshold tau."""
    import statistics as st
    conf = router_confidences(tag)
    for c, v in conf.items():
        vals.put(f"{mkey}.conf.{c}.mean", f"{st.mean(v):.3f}")
        vals.put(f"{mkey}.conf.{c}.sd", f"{st.pstdev(v):.3f}")
    for tau in TAUS:
        t = f"tau{round(tau * 100)}"
        for c in ("TARGET", "CONTROL"):
            vals.put(f"{mkey}.{t}.{c}", pct(sum(x >= tau for x in conf[c]) / len(conf[c])))


def add_inject(vals, mkey, data, tags):
    cis, items = item_cis(tags, lambda r: (r["condition"], r["agent"]))
    for c in INJECT_CONDITIONS + ["CONTROL"]:
        for a in INJECT_AGENTS:
            if c not in data["summary"] or a not in data["summary"][c]:
                continue
            cell = data["summary"][c][a]
            for outcome in ("trusted", "candidate", "absent"):
                d = cell[outcome]
                base = f"{mkey}.{c}.{a}.{outcome}"
                vals.put(f"{base}.k", d["k"])
                vals.put(f"{base}.n", d["n"])
                vals.put(f"{base}.pct", pct(d["frac"]))
                vals.put(f"{base}.cilo", pct(cis[(c, a)][outcome][0]))
                vals.put(f"{base}.cihi", pct(cis[(c, a)][outcome][1]))
            vals.put(f"{mkey}.{c}.{a}.items", items[(c, a)])


def write_inject_table(inj, path):
    """Trusted % under each injection variant, per model and agent."""
    tex = ["% AUTO-GENERATED. Do not edit.", "\\begin{tabular}{ll" + "r" * len(INJECT_AGENTS) + "}",
           "\\toprule", "Cue level & Model & " + " & ".join(AGENT_LABEL[a] for a in INJECT_AGENTS) + " \\\\",
           "\\midrule"]
    names = {"INJECT_OVERRIDE": "Explicit instruction", "INJECT_FORGED": "Forged channel tag",
             "INJECT_SUMMARY": "Third-person attribution"}
    for c in ["INJECT_SUMMARY", "INJECT_FORGED", "INJECT_OVERRIDE"]:  # ladder order
        first = True
        for mkey, mname, _ in MODELS:
            if mkey not in inj:
                continue
            cells = [pct(inj[mkey]["summary"][c][a]["trusted"]["frac"]) if a in inj[mkey]["summary"][c] else "--"
                     for a in INJECT_AGENTS]
            tex.append(f"{names[c] if first else ''} & {mname} & " + " & ".join(cells) + " \\\\")
            first = False
        tex.append("\\midrule")
    tex[-1] = "\\bottomrule"
    tex.append("\\end{tabular}")
    open(path, "w").write("\n".join(tex) + "\n")


def load_inject():
    """Injection results per model key, with the defended-manager cells merged in."""
    def complete(d):
        return d and d["meta"]["jobs_complete"].split("/")[0] == d["meta"]["jobs_complete"].split("/")[1]
    inj, tags_by_model, missing = {}, {}, []
    for mkey, tag in INJECT_TAGS.items():
        d, dd = load(f"ablation_{tag}.json"), load(f"ablation_{DEFENSE_TAGS[mkey]}.json")
        if not complete(d):
            missing.append(f"ablation_{tag}.json (absent or incomplete)")
            continue
        tags = [tag]
        if complete(dd):
            for c, row in dd["summary"].items():
                d["summary"].setdefault(c, {}).update({a: row[a] for a in DEFENSE_AGENTS if a in row})
            tags.append(DEFENSE_TAGS[mkey])
        else:
            missing.append(f"ablation_{DEFENSE_TAGS[mkey]}.json (absent or incomplete)")
        inj[mkey], tags_by_model[mkey] = d, tags
    return inj, tags_by_model, missing


def load_models():
    """Ablation result per model key, with the common-judge reflection override applied."""
    all_data, missing = {}, []
    for mkey, mname, tag in MODELS:
        d = load(f"ablation_{tag}.json")
        if d is None:
            missing.append(f"ablation_{tag}.json")
            continue
        # The reflection arm is the only arm scored by the presence judge. Where a model has a
        # dedicated reflection rerun with the common judge, its reflection numbers replace the
        # main run's so that every reported judge-dependent number uses the same judge.
        override = REFLECTION_OVERRIDE.get(mkey)
        if override:
            r = load(f"ablation_{override}.json")
            if r is None or r["meta"]["jobs_complete"].split("/")[0] != r["meta"]["jobs_complete"].split("/")[1]:
                raise SystemExit(f"reflection override {override} missing or incomplete")
            for c in CONDITIONS:
                d["summary"][c]["reflection"] = r["summary"][c]["reflection"]
                d["per_repeat"][c]["reflection"] = r["per_repeat"][c]["reflection"]
            for pair in d["separation"]:
                d["separation"][pair]["reflection"] = r["separation"][pair]["reflection"]
            d["meta"]["judge_model"] = r["meta"]["judge_model"]
        all_data[mkey] = d
    return all_data, missing


def write_numbers_tex(vals, path):
    lines = ["% AUTO-GENERATED by research/experiments/make_paper_numbers.py. Do not edit.",
             "% \\R{key} prints a result value; an unknown key stops the build.",
             "\\makeatletter",
             "\\newcommand{\\R}[1]{\\ifcsname res@#1\\endcsname\\csname res@#1\\endcsname"
             "\\else\\PackageError{results}{Unknown result key: #1}{Regenerate paper_generated/numbers.tex}\\fi}"]
    for k in sorted(vals.v):
        lines.append(f"\\expandafter\\def\\csname res@{k}\\endcsname{{{vals.v[k]}}}")
    lines.append("\\makeatother")
    open(path, "w").write("\n".join(lines) + "\n")


def write_main_table(data, mkey, mname, path):
    """Primary model: trusted count per condition x agent, with per-repeat range."""
    reps = len(data["repeats"])
    cols = AGENTS
    head = " & ".join(AGENT_LABEL[a] for a in cols)
    rows = []
    for c in CONDITIONS:
        cells = []
        for a in cols:
            d = data["summary"][c][a]["trusted"]
            cells.append(f"{d['k']}")
        rows.append(f"{COND_LABEL[c]} & " + " & ".join(cells) + " \\\\")
    n = data["summary"]["TARGET"]["schema_prov"]["trusted"]["n"]
    tex = [f"% AUTO-GENERATED ({mname}). Do not edit.",
           "\\begin{tabular}{l" + "r" * len(cols) + "}",
           "\\toprule",
           f"Condition & {head} \\\\",
           "\\midrule", *rows, "\\bottomrule", "\\end{tabular}",
           f"% trusted counts out of n={n} per cell ({data['n_personas']} personas x 5 items x {reps} repeats)"]
    open(path, "w").write("\n".join(tex) + "\n")


def write_crossmodel_table(all_data, path):
    """Trusted % per model for the conditions that carry the claims, key agents only."""
    agents = ["schema_no_prov", "content_judge", "llm_prov", "schema_prov", "reflection"]
    conds = ["TARGET", "AUTHORIZATION", "SPOOF", "CONTROL"]
    tex = ["% AUTO-GENERATED. Do not edit.",
           "\\begin{tabular}{ll" + "r" * len(agents) + "}", "\\toprule",
           "Condition & Model & " + " & ".join(AGENT_LABEL[a] for a in agents) + " \\\\", "\\midrule"]
    for c in conds:
        for i, (mkey, mname, _) in enumerate(MODELS):
            d = all_data.get(mkey)
            if d is None:
                continue
            cells = [pct(d["summary"][c][a]["trusted"]["frac"]) for a in agents]
            tex.append(f"{COND_LABEL[c] if i == 0 else ''} & {mname} & " + " & ".join(cells) + " \\\\")
        tex.append("\\midrule")
    # policy fidelity row: untrusted world facts kept as candidate evidence
    for i, (mkey, mname, _) in enumerate(MODELS):
        d = all_data.get(mkey)
        if d is None:
            continue
        cells = [pct(d["summary"]["PARANOIA"][a]["candidate"]["frac"]) for a in agents]
        tex.append(f"{'WORLD-FACT (candidate)' if i == 0 else ''} & {mname} & " + " & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    open(path, "w").write("\n".join(tex) + "\n")


def write_judge_table(data, path):
    judges = list(data["trusted_rate"].keys())
    tex = ["% AUTO-GENERATED. Do not edit.", "\\begin{tabular}{lrrrr}", "\\toprule",
           "Condition & $n$ & Agreement (\\%) & " + " & ".join(f"Trusted ({JUDGE_LABEL.get(j, j)})" for j in judges) + " \\\\",
           "\\midrule"]
    for c, row in data["per_condition"].items():
        tex.append(f"{COND_LABEL.get(c, c)} & {row['n']} & {pct(row['agreement'])} & " +
                   " & ".join(str(row[f'trusted_{j}']) for j in judges) + " \\\\")
    tex += ["\\midrule",
            f"All & {data['n']} & {pct(data['agreement'])} & " +
            " & ".join(str(round(data['trusted_rate'][j] * data['n'])) for j in judges) + " \\\\",
            "\\bottomrule", "\\end{tabular}", f"% Cohen's kappa = {data['cohen_kappa']:.3f}"]
    open(path, "w").write("\n".join(tex) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default="opus5")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    vals = Values()
    all_data, missing = load_models()
    for mkey, _, tag in MODELS:
        if mkey in all_data:
            add_ablation(vals, mkey, all_data[mkey], tag)
            add_confidence(vals, mkey, tag)
    inj, inj_tags, inj_missing = load_inject()
    missing += inj_missing
    for mkey, d in inj.items():
        add_inject(vals, mkey, d, inj_tags[mkey])
    j = load("judge_agreement_judge_agree_opus5.json")
    if j:
        add_judge(vals, j)
    else:
        missing.append("judge_agreement_judge_agree_opus5.json")
    m = load("mem0_mem0_opus5_g6.json")
    if m and m["meta"].get("jobs_complete", "").split("/")[0] == m["meta"].get("jobs_complete", "").split("/")[-1]:
        add_mem0(vals, m, "mem0_opus5_g6")
    else:
        missing.append("mem0_mem0_opus5_g6.json (absent or incomplete)")

    write_numbers_tex(vals, os.path.join(OUT, "numbers.tex"))
    pm = next(x for x in MODELS if x[0] == args.primary)
    write_main_table(all_data[pm[0]], pm[0], pm[1], os.path.join(OUT, "tab_main.tex"))
    write_crossmodel_table(all_data, os.path.join(OUT, "tab_crossmodel.tex"))
    if j:
        write_judge_table(j, os.path.join(OUT, "tab_judge.tex"))
    if inj:
        write_inject_table(inj, os.path.join(OUT, "tab_inject.tex"))
    with open(os.path.join(OUT, "numbers.md"), "w") as f:
        f.write("# Generated result values\n\n| key | value |\n|---|---|\n")
        for k in sorted(vals.v):
            f.write(f"| `{k}` | {vals.v[k]} |\n")

    print(f"wrote {len(vals.v)} values to {os.path.relpath(OUT)}/numbers.tex (+ tables, numbers.md)")
    for x in missing:
        print(f"  missing: {x}")


if __name__ == "__main__":
    main()
