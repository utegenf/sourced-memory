"""External-validity anchor: does Mem0 (an off-the-shelf agent memory layer) assimilate the SAME
plausible-false-personal claim our content-only baselines accept?

This is NOT a reimplementation of our architecture. Mem0 has no source/trust concept -- that is
exactly the point: a content-based SOTA memory has no mechanism to reject a schema-fitting fabrication.
We run the identical TARGET items from benchmark_items.json:
  1. seed the persona's 4 schema beliefs as memories (as the user),
  2. add ONE injected schema-fitting false claim (a claim the user never actually stated),
  3. query Mem0 for the user's preferences and check (cross-family judge, same as main exp) whether
     the injected claim is now surfaced as a believed user preference.
"acceptance" = injected claim present as a believed preference in Mem0's retrieved memories.

Mirrors the TARGET measurement for RAG/Reflection/schema_no_prov/schema_conf (all 100% accept).
Anthropic Claude Sonnet 4.5 as the LLM, a local sentence-transformers embedder, qdrant in-memory.
Deterministic (temperature 0).

Usage: python3 run_mem0_spotcheck.py [--personas N]   (default: all 10; --personas 2 for a smoke test)
"""
import argparse
import json
import os

import run_experiment_v2 as V
from run_provenance_ablation import belief_present  # reuse the exact cross-family presence judge

_DIR = os.path.dirname(os.path.abspath(__file__))
PERSONAS = json.load(open(os.path.join(_DIR, "..", "benchmark", "benchmark_items.json")))["personas"]

# Same agent model as the main experiment. Mem0 also needs an embedder; we use a local
# sentence-transformers model so the spot-check depends only on the Anthropic LLM.
LLM_MODEL = V.AGENT_MODEL_ID
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIMS = int(os.environ.get("EMBED_DIMS", "384"))


def make_memory():
    """Construct a Mem0 Memory using Anthropic Claude as the LLM, a local embedder, and
    an in-memory qdrant store. Models are configured by name via the environment."""
    from mem0 import Memory
    config = {
        "llm": {"provider": "anthropic",
                "config": {"model": LLM_MODEL, "temperature": 0.0}},
        "embedder": {"provider": "huggingface",
                     "config": {"model": EMBED_MODEL}},
        "vector_store": {"provider": "qdrant",
                         "config": {"collection_name": "spotcheck", "on_disk": False,
                                    "embedding_model_dims": EMBED_DIMS}},
    }
    return Memory.from_config(config)


def run_one(persona, injected_claim):
    """Seed schema, inject one false claim, query, judge presence. Returns 'trusted'/'absent'."""
    mem = make_memory()
    uid = f"user_{persona['name']}"
    # 1. seed schema beliefs (as the user)
    for belief in persona["schema"]:
        mem.add(belief, user_id=uid)
    # 2. inject the schema-fitting false claim (a claim the user never actually stated)
    mem.add(injected_claim, user_id=uid)
    # 3. retrieve what Mem0 now believes about the user, relevant to the injected claim
    hits = mem.search(query=injected_claim, filters={"user_id": uid}, top_k=10)
    memories = hits.get("results", hits) if isinstance(hits, dict) else hits
    store_text = "\n".join(m.get("memory", str(m)) if isinstance(m, dict) else str(m)
                           for m in memories)
    # same cross-family meaning-based presence judge as the main experiment
    present = belief_present(store_text, injected_claim)
    return "trusted" if present else "absent", store_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", type=int, default=len(PERSONAS))
    args = ap.parse_args()
    personas = PERSONAS[:args.personas]

    rows, n_trusted, n = [], 0, 0
    for persona in personas:
        for claim in persona["schema_fit_false"]:
            try:
                outcome, store = run_one(persona, claim)
            except Exception as e:                       # record failures, don't crash the sweep
                outcome, store = f"ERROR:{type(e).__name__}", str(e)[:200]
            n += 1
            if outcome == "trusted":
                n_trusted += 1
            rows.append({"persona": persona["name"], "claim": claim,
                         "outcome": outcome, "retrieved": store[:400]})
            print(f"  {persona['name']:12s} {outcome:9s} :: {claim[:55]}")
    errors = [r for r in rows if r["outcome"].startswith("ERROR")]
    print(f"\nMem0 TARGET acceptance (trusted): {n_trusted}/{n}"
          + (f"  [{len(errors)} errors]" if errors else ""))
    out = {"summary": {"n": n, "trusted": n_trusted, "errors": len(errors),
                       "model": LLM_MODEL, "embed": EMBED_MODEL},
           "rows": rows}
    path = os.path.join(_DIR, "..", "results", "mem0_spotcheck.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
