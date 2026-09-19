# Research: reproducing the paper

This subtree contains the controlled experiments behind
*Content Interprets, Origin Decides: Source-Aware Belief Updating for Lifelong Agent Memory*.
It is a reproducibility artifact, **not** part of the installable library.
The library lives in `../src/sourced_memory/` and can be used independently.

## Layout

```
research/
├── experiments/    experiment scripts + LLM provider glue
├── benchmark/      hand-authored personas + injected items (author-set ground truth)
├── results/        raw JSON outputs from each run
└── figures/        generated figures used in the paper
```

## Install research dependencies

The research scripts pull `anthropic`, `pydantic`, `matplotlib`, and `numpy`.
Install them via the `research` extra from the repo root:

```bash
pip install -e ".[research]"
```

## Run the ablation

```bash
export ANTHROPIC_API_KEY=<your key>
export AGENT_MODEL=claude-sonnet-4-5                # main agent
export JUDGE_MODEL=<a different-family model>       # cross-family presence judge only

python3 research/experiments/run_provenance_ablation.py    # core n=100 ablation
python3 research/experiments/run_mem0_spotcheck.py         # external-validity anchor
python3 research/experiments/measure_gate_confidence.py    # confidence-vs-truth measurement
python3 research/experiments/instrument_reversal.py        # trace the 2/100 reversal drops
python3 research/experiments/make_figures.py               # regenerate paper figures
python3 research/experiments/make_schematics.py            # regenerate concept diagrams
```

All runs use temperature 0 (deterministic decoding); variation across `n=100`
comes from personas and items, not sampling. The core schema-agent metric is
deterministic (`source_id` inspection, no LLM in the loop); a cross-family
judge is only used for paraphrase-robust presence checks on the summarization
baseline.

## Results shipped in this repo

The `results/` directory contains the JSON files behind every number reported
in the paper:

| File | Contents |
|---|---|
| `prov_ablation_sonnet45.json` | Core `n=100` ablation on Claude Sonnet-4.5 |
| `prov_ablation_llama4.json` | Cross-family replication on Llama-4-Maverick |
| `mem0_spotcheck.json` | External-validity check against Mem0 |
| `gate_confidence.json` | Router confidence on fabrications vs genuine updates |
| `reversal_drops.json` | Trace of the 2 dropped reversals (Stage-1 routing errors) |

## Relationship to the library

The research scripts predate the library refactor and remain self-contained:
they carry their own belief store implementation (`SleepAgent`), their own
prompts, and their own consolidation logic tuned for the paper's benchmark.
They are **not** an example of how to use `sourced_memory`; for that see the
top-level `examples/` directory.

Rewriting the research code on top of the library is out of scope for v0 and
would change the experimental protocol.
