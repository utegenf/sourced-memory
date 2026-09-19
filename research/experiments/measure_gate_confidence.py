"""Measure the gate's classification confidence for TARGET fabrications vs. genuine reversals.

This is the evidence behind the confidence-threshold argument: schema_conf admits a personal
claim iff gate.confidence >= tau. If the confidence assigned to a schema-fitting FABRICATION
overlaps the confidence assigned to a GENUINE reversal, then no tau separates them, and raising
tau cannot fix the blind spot without also rejecting real updates.

For each persona we build the schema store, inject one item, run consolidation, and read the
Stage-1 gate confidence for that injected item from the gate_log. Deterministic (temperature 0).
Writes results/gate_confidence.json.
"""
import json
import os
import statistics as st

import run_experiment_v2 as V
from run_provenance_ablation import PERSONAS, NUM_DAYS, build_cached_schema_sleep

_DIR = os.path.dirname(os.path.abspath(__file__))


def gate_confidence_for(persona, claim, schema_store, seed=0):
    """Inject one claim, consolidate, return the Stage-1 gate confidence for that item."""
    import copy
    sleep = V.SleepAgent(use_provenance=False)  # config irrelevant; we only read the gate log
    sleep.semantic_store = copy.deepcopy(schema_store)
    inj = {"text": claim, "channel": V.TRUSTED_CHANNEL, "source_id": "injected"}
    sleep.interact(inj)
    sleep.trigger_sleep_cycle(NUM_DAYS + 1, seed=seed)
    rows = [g for g in sleep.gate_log if g["memory"] == claim]
    if not rows:
        return None
    return {"confidence": rows[0]["confidence"], "category": rows[0]["category"]}


def main():
    fab, rev = [], []          # gate confidences: fabrications vs. genuine reversals
    fab_rows, rev_rows = [], []
    for persona in PERSONAS:
        schema_store = build_cached_schema_sleep(persona, seed=0)
        for claim in persona["schema_fit_false"]:          # TARGET fabrications
            r = gate_confidence_for(persona, claim, schema_store)
            if r:
                fab.append(r["confidence"])
                fab_rows.append({"persona": persona["name"], "claim": claim, **r})
        for claim in persona["reversal_true"]:             # genuine trusted reversals
            r = gate_confidence_for(persona, claim, schema_store)
            if r:
                rev.append(r["confidence"])
                rev_rows.append({"persona": persona["name"], "claim": claim, **r})
        print(f"  {persona['name']:12s} done")

    def stats(xs):
        xs = sorted(xs)
        return {"n": len(xs), "mean": round(st.mean(xs), 3),
                "min": min(xs), "max": max(xs),
                "sd": round(st.pstdev(xs), 3) if len(xs) > 1 else 0.0}

    fab_s, rev_s = stats(fab), stats(rev)
    # Overlap: does any tau in (0,1] separate the two sets? Separable iff min(one) > max(other).
    separable = (min(fab) > max(rev)) or (min(rev) > max(fab))
    overlap_lo, overlap_hi = max(min(fab), min(rev)), min(max(fab), max(rev))
    out = {
        "fabrication_confidence": fab_s,
        "reversal_confidence": rev_s,
        "separable_by_threshold": separable,
        "overlap_range": [overlap_lo, overlap_hi] if overlap_hi >= overlap_lo else None,
        "fab_rows": fab_rows, "rev_rows": rev_rows,
    }
    print("\n=== GATE CONFIDENCE ===")
    print(f"  fabrication (TARGET):  n={fab_s['n']} mean={fab_s['mean']} range[{fab_s['min']},{fab_s['max']}]")
    print(f"  genuine reversal:      n={rev_s['n']} mean={rev_s['mean']} range[{rev_s['min']},{rev_s['max']}]")
    print(f"  separable by any tau?  {separable}")
    print(f"  overlap range:         {out['overlap_range']}")
    path = os.path.join(_DIR, "..", "results", "gate_confidence.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
