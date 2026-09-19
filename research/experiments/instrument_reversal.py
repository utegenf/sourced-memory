"""Instrumented re-run of the CONTROL (true reversal / trusted) condition for schema_prov,
to capture the Stage-1 admission -> Stage-2 consolidation trace of the dropped reversals.

Deterministic (temperature 0), so it reproduces the same dropped reversals as
results/prov_ablation_sonnet45.json. For each dropped reversal it records: the injected episode and
the Stage-1 gate decision, showing the loss is a Stage-1 content-routing error (a durable preference
change misclassified as a transient EVENT), independent of origin.

Writes results/reversal_drops.json.
"""
import copy
import json
import os

import run_experiment_v2 as V
from run_provenance_ablation import PERSONAS, NUM_DAYS, build_cached_schema_sleep

_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- capture every Stage-2 consolidation (MergedStore) call: prompt in, beliefs out ----
_orig = V.complete_structured
_capture = []


def _wrapped(system_prompt, user_prompt, pydantic_model, model_id=V.AGENT_MODEL_ID, seed=None):
    out = _orig(system_prompt, user_prompt, pydantic_model, model_id=model_id, seed=seed)
    if pydantic_model is V.MergedStore:
        _capture.append({"prompt": user_prompt,
                         "beliefs": [f"[{b.category}] {b.belief}" for b in out.beliefs]})
    return out


V.complete_structured = _wrapped


def run_one(persona, claim, schema_store, seed=0):
    """Inject one trusted reversal, consolidate, return (outcome, gate_rows, consolidation_calls)."""
    sleep = V.SleepAgent(use_provenance=True)
    sleep.semantic_store = copy.deepcopy(schema_store)
    inj = {"text": claim, "channel": V.TRUSTED_CHANNEL, "source_id": "injected"}
    sleep.interact(inj)
    _capture.clear()
    sleep.trigger_sleep_cycle(NUM_DAYS + 1, seed=seed)
    entries = [b for b in sleep.semantic_store if b.source_id == "injected"]
    if not entries:
        outcome = "absent"
    elif any(getattr(b, "status", "trusted") == "trusted" and not b.quarantined for b in entries):
        outcome = "trusted"
    elif any(getattr(b, "status", "trusted") == "candidate" for b in entries):
        outcome = "candidate"
    else:
        outcome = "absent"
    # Stage-1 gate decision for the injected episode
    gate_rows = [g for g in sleep.gate_log if g["memory"] == claim]
    # The Stage-2 consolidation call whose input contained the injected reversal
    injected_calls = [c for c in _capture if claim.split()[0] in c["prompt"] or "injected" in c["prompt"]
                      or any(w in c["prompt"] for w in claim.split()[-3:])]
    return outcome, gate_rows, list(_capture)


def main():
    results, drops = [], []
    for persona in PERSONAS:
        schema_store = build_cached_schema_sleep(persona, seed=0)  # build once per persona
        for claim in persona["reversal_true"]:
            outcome, gate_rows, calls = run_one(persona, claim, schema_store)
            rec = {"persona": persona["name"], "schema": persona["schema"], "claim": claim,
                   "outcome": outcome, "gate": gate_rows, "consolidation_calls": calls}
            results.append({k: rec[k] for k in ("persona", "claim", "outcome")})
            if outcome != "trusted":
                drops.append(rec)
            print(f"  {persona['name']:12s} {outcome:8s} :: {claim[:60]}")
    n = len(results)
    n_trusted = sum(1 for r in results if r["outcome"] == "trusted")
    print(f"\nreversal retained (trusted): {n_trusted}/{n}; dropped: {len(drops)}")
    out = {"summary": {"n": n, "trusted": n_trusted, "dropped": len(drops)},
           "all": results, "drops": drops}
    path = os.path.join(_DIR, "..", "results", "reversal_drops.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
