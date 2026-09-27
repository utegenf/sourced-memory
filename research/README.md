# Research: reproducing the paper

This subtree contains the experiments behind *Content Interprets, Origin Decides: Source-Aware
Admission for Persistent Agent Memory*. It is a reproducibility artifact and is not part of the
installable library; it does not import `src/`.

## Layout

```
research/
├── benchmark/     20 hand-authored personas and their items (author-set ground truth), plus
│                  third-person rewrites of the fabrications used by INJECT_SUMMARY
├── experiments/   experiment code, model access, and the scripts that regenerate every number and figure
├── results/       aggregate result files (per-job files in results/runs/ are kept out of git)
└── figures/       figures used in the paper (generated)
```

## Regenerate every number, table, and figure (no model access needed)

```bash
pip install -r research/requirements.txt
cd research/experiments
python3 make_paper_numbers.py   # ../../paper_generated/: numbers.tex, tables, numbers.md
python3 make_figures.py         # ../figures/fig_core.pdf, fig_outcome_dist.pdf, fig_pe_validation.pdf
python3 make_schematics.py      # ../figures/fig_mechanism.pdf
```

`paper_generated/numbers.md` lists every value cited in the paper by key. Confidence intervals are
item-level Wilson intervals (repeats averaged per item, n = distinct items). The intervals and the
confidence-threshold sweep are computed from the per-job files in `results/runs/`, which are not in
git; they are included in the paper's supplementary material.

## Rerun the experiments

Model calls go through the Amazon Bedrock Converse API (standard AWS credential chain; public model
IDs in `model_client.py`). Account-specific overrides go in `experiments/model_registry_local.py`,
which is git-ignored.

```bash
cd research/experiments
AGENT_MODEL=opus5 python3 run_provenance_ablation.py --tag opus5_v4 --repeats 5   # main ablation
AGENT_MODEL=opus5 python3 run_mem0_spotcheck.py --tag mem0_opus5_g6               # Mem0 check
AGENT_MODEL=opus5 python3 judge_agreement.py --tag judge_agree_opus5              # judge study
AGENT_MODEL=opus5 python3 run_provenance_ablation.py --tag inject_opus5 --repeats 3 --no-pe \
  --agents llm_prov,content_judge,schema_prov --conditions INJECT_OVERRIDE,INJECT_FORGED,INJECT_SUMMARY
```

Runs checkpoint per (persona, repeat) and resume with the same `--tag`. The presence judge
(`JUDGE_MODEL`, default `gpt6_astra`) is used only for the Reflection baseline and the Mem0 check.

## Result files

| File | Contents |
|---|---|
| `ablation_opus5_v4.json` | Main ablation, Claude Opus 5, 5 repeats |
| `ablation_opus5_refl_g6.json` | Opus 5 Reflection arm rerun with the common judge |
| `ablation_deepseek_v32_v4.json` | DeepSeek V3.2, 3 repeats |
| `ablation_qwen3_235b_v4.json` | Qwen3-235B-A22B, 3 repeats |
| `ablation_qwen3_32b_v4.json` | Qwen3-32B, 3 repeats |
| `mem0_mem0_opus5_g6.json` | Mem0 with Opus 5 (TARGET, CONTROL, SPOOF) |
| `judge_agreement_judge_agree_opus5.json` | Agreement between GPT-6 Astra and gpt-oss-120b |
| `ablation_inject_<model>.json` | Prompt-injection conditions (INJECT_OVERRIDE, INJECT_FORGED, INJECT_SUMMARY), 3 agents, 3 repeats |
| `ablation_defense_<model>.json` | Same conditions plus CONTROL, prompted manager with Spotlighting (`llm_prov_spot`) or an explicit rule (`llm_prov_rule`) |
