"""External-validity check: does Mem0, an off-the-shelf agent memory layer, treat a fabricated
personal claim the same way it treats a genuine preference change?

Mem0 has no notion of source trust, so the item text is all it sees. For every hand-authored item:
  1. seed the persona's 4 schema beliefs as user messages,
  2. add ONE item (a fabrication, a genuine reversal, or a fabrication phrased as a user quote),
  3. read back everything Mem0 now stores for the user and ask the cross-family judge whether the
     item is asserted as a belief about the user.

Mem0's LLM calls are routed through model_client (same Bedrock model as the main ablation, same
retry logic, account settings from the gitignored local registry). Embeddings use a local
sentence-transformers model and an in-memory Qdrant store, so the check depends on one LLM only.

Each (persona, repeat) job is checkpointed under results/runs/<tag>/.
"""
import argparse
import datetime as _dt
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

# The embedder is loaded once per worker thread. Without offline mode every load sends HEAD
# requests to the Hugging Face Hub, which rate-limits (HTTP 429) and stalls each worker ~150s.
# Download the model once beforehand; after that it loads from the local cache.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# With telemetry on (the default), every Mem0 instance also opens a "migrations" Qdrant store at a
# fixed path under ~/.mem0, shared by all threads -> "Storage folder ... already accessed" errors.
# Telemetry does not affect how Mem0 stores or retrieves memories. Must be set before importing mem0.
os.environ["MEM0_TELEMETRY"] = "False"

import model_client as MC
import run_experiment_v2 as V
from run_provenance_ablation import SPOOF_TEMPLATE, _wilson, belief_present

_DIR = os.path.dirname(os.path.abspath(__file__))
PERSONAS = json.load(open(os.path.join(_DIR, "..", "benchmark", "benchmark_items.json")))["personas"]
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIMS = int(os.environ.get("EMBED_DIMS", "384"))

CONDITIONS = [  # (label, claim_type, transform)
    ("TARGET", "schema_fit_false", None),
    ("CONTROL", "reversal_true", None),
    ("SPOOF", "schema_fit_false", "spoof"),
]


class _BedrockLLM:
    """Minimal stand-in for Mem0's LLM interface, backed by model_client."""

    def __init__(self, model_key):
        self.model_key = model_key

    def generate_response(self, messages, tools=None, tool_choice="auto", response_format=None, **kw):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
        if response_format and response_format.get("type") == "json_object":
            system += "\n\nRespond with a single valid JSON object and nothing else."
        return MC.complete_text(system or "You are a helpful assistant.", user, model_id=self.model_key)


_local = threading.local()
_init_lock = threading.Lock()  # model loading is not thread-safe; build instances one at a time


def _memory():
    """One Mem0 instance per thread (Qdrant in-memory client and embedder are not shared)."""
    if getattr(_local, "mem", None) is not None:
        return _local.mem
    with _init_lock:
        from mem0 import Memory
        cfg = {
            "llm": {"provider": "aws_bedrock", "config": {"model": "us.meta.llama4-maverick-17b-instruct-v1:0"}},
            "embedder": {"provider": "huggingface", "config": {"model": EMBED_MODEL}},
            # Each thread needs its own Qdrant folder: Mem0 defaults every instance to
            # /tmp/qdrant, and a local Qdrant folder can only be opened by one client.
            "vector_store": {"provider": "qdrant", "config": {
                "collection_name": f"spot_{uuid.uuid4().hex[:8]}", "on_disk": False,
                "path": f"/tmp/qdrant_mem0_{os.getpid()}_{threading.get_ident()}",
                "embedding_model_dims": EMBED_DIMS}},
        }
        mem = Memory.from_config(cfg)
        mem.llm = _BedrockLLM(V.AGENT_MODEL_ID)  # every Mem0 LLM call goes through model_client
        _local.mem = mem
    return _local.mem


def run_one(persona, text, claim):
    mem = _memory()
    uid = f"u_{uuid.uuid4().hex}"
    for belief in persona["schema"]:
        mem.add([{"role": "user", "content": belief}], user_id=uid)
    mem.add([{"role": "user", "content": text}], user_id=uid)
    got = mem.get_all(filters={"user_id": uid})
    items = got.get("results", got) if isinstance(got, dict) else got
    store_text = "\n".join(m.get("memory", str(m)) if isinstance(m, dict) else str(m) for m in items)
    present = belief_present(store_text, claim)
    return ("trusted" if present else "absent"), store_text


def job(persona, repeat, item_workers):
    tasks = []
    for label, ct, tf in CONDITIONS:
        for idx, claim in enumerate(persona[ct]):
            text = SPOOF_TEMPLATE.format(claim=claim) if tf == "spoof" else claim
            tasks.append((label, idx, claim, text))

    def one(t):
        label, idx, claim, text = t
        outcome, store = run_one(persona, text, claim)
        return {"persona": persona["name"], "repeat": repeat, "condition": label, "item_index": idx,
                "claim": claim, "payload": text, "outcome": outcome, "stored": store}

    with ThreadPoolExecutor(max_workers=item_workers) as ex:
        recs = [f.result() for f in as_completed([ex.submit(one, t) for t in tasks])]
    return {"persona": persona["name"], "repeat": repeat, "records": recs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--personas", type=int, default=len(PERSONAS))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--item-workers", type=int, default=1,
                    help="keep 1 unless Mem0 instances are made per task (thread-local instance)")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    run_dir = os.path.join(_DIR, "..", "results", "runs", args.tag)
    os.makedirs(run_dir, exist_ok=True)
    meta = {"agent_model": V.AGENT_MODEL_ID, "judge_model": V.JUDGE_MODEL_ID, "embed": EMBED_MODEL,
            "conditions": [c[0] for c in CONDITIONS], "repeats": list(range(args.repeats)),
            "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}
    json.dump(meta, open(os.path.join(run_dir, "_meta.json"), "w"), indent=2)

    todo = [(p, r, os.path.join(run_dir, f"{p['name']}_r{r}.json"))
            for r in range(args.repeats) for p in PERSONAS[:args.personas]]
    todo = [t for t in todo if not os.path.exists(t[2])]
    print(f"### mem0 {args.tag}: model={V.AGENT_MODEL_ID} todo={len(todo)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(job, p, r, args.item_workers): (p["name"], r, path) for p, r, path in todo}
        for fut in as_completed(futs):
            name, r, path = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"  FAILED {name} r{r}: {type(e).__name__}: {str(e)[:300]}", flush=True)
                continue
            json.dump(res, open(path + ".tmp", "w"), indent=1)
            os.replace(path + ".tmp", path)
            print(f"  done {name} r{r}", flush=True)

    jobs = [json.load(open(os.path.join(run_dir, f))) for f in sorted(os.listdir(run_dir))
            if f.endswith(".json") and not f.startswith("_")]
    recs = [x for j in jobs for x in j["records"]]
    summary = {}
    for label, _, _ in CONDITIONS:
        outs = [x["outcome"] for x in recs if x["condition"] == label]
        k, n = sum(o == "trusted" for o in outs), len(outs)
        summary[label] = {"trusted": k, "n": n, "frac": k / n if n else None, "ci": _wilson(k, n)}
    meta["finished_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    meta["jobs_complete"] = f"{len(jobs)}/{args.personas * args.repeats}"
    out = os.path.join(_DIR, "..", "results", f"mem0_{args.tag}.json")
    json.dump({"summary": summary, "meta": meta}, open(out, "w"), indent=2)
    print(json.dumps(summary, indent=1))
    print(f"Saved {out} (jobs {meta['jobs_complete']})", flush=True)


if __name__ == "__main__":
    main()
