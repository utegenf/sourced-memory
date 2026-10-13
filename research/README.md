# Research: reproducing the paper

This subtree contains the experiments behind *Content Interprets, Origin Decides: Stability and
Plasticity in Agent Belief Formation*. It is a reproducibility artifact and is not part of the
installable library; it does not import `src/`.

## Layout

```
research/
├── benchmark/     20 hand-authored personas and their items (author-set ground truth), plus
│                  third-person rewrites of the fabrications used by INJECT_SUMMARY;
│                  e2e/ and e2e_targeted/ hold the carrier documents, tasks, and probes
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

End-to-end pipeline (design in [`docs/E1_E2_design.md`](../docs/E1_E2_design.md)): the injected
document arrives through a real tool call, five memory writers decide what to store, and a later
session tests whether the stored claim changes the answer.

```bash
export JUDGE_MODEL=judge_oss
AGENT_MODEL=opus5_global python3 run_e2e.py --tag full_opus5 --repeats 1
AGENT_MODEL=opus5_global python3 run_e2e.py --tag extra_opus5 --repeats 1 \
  --conditions T-STEALTH,LEGIT-NOTES,T-CONFIRM-YES-NORULE
AGENT_MODEL=opus5_global python3 augment_detector_e2e.py --tag full_opus5    # detector + content writer
JUDGE_MODEL=gpt6_astra python3 rejudge_e2e.py --tag full_opus5 --sample-frac 0.1   # judge agreement
```

Qwen3-32B uses `AGENT_MODEL=qwen3_32b`, tags `full_qwen3_32b` and `extra_qwen3_32b`, and 3 repeats.

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
| `e2e_full_<model>.json` | End-to-end pipeline: injection, confirmation, and utility conditions, five writers |
| `e2e_extra_<model>.json` | End-to-end stealthy injection, user's own notes, and confirmation without the ask-first rule |
| `e2e_rejudge_<tag>_gpt6_astra.json` | End-to-end judge agreement (GPT-6 Astra vs gpt-oss-120b) |
| `e2e_pilot_<model>.json` | End-to-end pilot (5 personas, 1 repeat) |
