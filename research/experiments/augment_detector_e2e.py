"""Add the detector writer (w5_detect) to the records of an existing e2e run.

The detector reads only the saved tool outputs, so no session is re-run. When nothing is flagged
the transcript seen by the memory writer is unchanged, and w5 reuses w1's saved result. When a tool
output is flagged, it is removed and the content-only extraction is re-run with the run's agent model
(set AGENT_MODEL to match the run). The original run directory is copied to <run>_pre_w5 first.
"""
import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import model_client as MC
import run_experiment_v2 as V
import run_e2e as E

_DIR = os.path.dirname(os.path.abspath(__file__))


def augment(rec, persona_by_name, assets_cache):
    if "w5_detect" in rec["writers"]:
        return rec
    tr = [tuple(x) for x in rec["transcript"]]
    flags = [E.detect_injection(t) if r == "tool" else False for r, t in tr]
    rec["detector_flags"] = flags
    w1 = rec["writers"]["w1_content"]
    if not any(flags):
        rec["writers"]["w5_detect"] = dict(w1)
        return rec
    kept = [(r, E.FILTERED if f else t) for (r, t), f in zip(tr, flags)]
    beliefs = [f.statement for f in V.complete_structured(
        E.EXTRACT_SYSTEM, f"CONVERSATION:\n{E._render(kept, labelled=True)}", E.Facts).facts]
    store = "\n".join(f"- {b}" for b in beliefs)
    kind = {c[0]: c[1] for c in E.CONDITIONS + E.EXTRA_CONDITIONS}[rec["condition"]]
    if kind == "tgt":
        admitted = any(E._has_token(b, rec["target_token"]) for b in beliefs)
        answer = None
        if admitted:
            a = assets_cache[rec["persona"]]["targeted"][rec["item_index"]]
            answer = E.session2_answer(persona_by_name[rec["persona"]], beliefs, a["probe"],
                                       rule=not rec["condition"].endswith("-NORULE"))
        acts = E._has_token(answer, rec["target_token"])
    else:
        admitted = E.presence_all([store], rec["claim"])[0]
        answer, acts = None, False
    rec["writers"]["w5_detect"] = {"beliefs": beliefs, "candidates": [], "admitted": admitted,
                                   "acts": acts, "session2_answer": answer}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    run_dir = os.path.join(_DIR, "..", "results", "runs", f"e2e_{args.tag}")
    backup = run_dir + "_pre_w5"
    if not os.path.exists(backup):
        shutil.copytree(run_dir, backup)
    files = sorted(f for f in os.listdir(run_dir) if f.endswith(".json") and not f.startswith("_"))
    persona_by_name = {p["name"]: p for p in E.PERSONAS}
    assets_cache = {}
    for f in files:
        job = json.load(open(os.path.join(run_dir, f)))
        assets_cache.setdefault(job["persona"], E.load_assets(job["persona"]))
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            job["records"] = list(ex.map(lambda r: augment(r, persona_by_name, assets_cache), job["records"]))
        json.dump(job, open(os.path.join(run_dir, f) + ".tmp", "w"), indent=1)
        os.replace(os.path.join(run_dir, f) + ".tmp", os.path.join(run_dir, f))
        print(f"  {f}: flagged {sum(any(r['detector_flags']) for r in job['records'])}/{len(job['records'])}; "
              f"spent ${MC.spent_usd():.2f}", flush=True)

    # Rebuild the aggregate with the new writer.
    jobs = [json.load(open(os.path.join(run_dir, f))) for f in files]
    labels = {r["condition"] for j in jobs for r in j["records"]}
    conds = [c for c in E.CONDITIONS + E.EXTRA_CONDITIONS if c[0] in labels]
    out = os.path.join(_DIR, "..", "results", f"e2e_{args.tag}.json")
    prev = json.load(open(out))
    agg = E.aggregate(jobs, conds)
    agg["meta"] = prev["meta"]
    agg["meta"]["w5_augment"] = {"detector": E.DETECTOR_MODEL, "cost_usd": round(MC.spent_usd(), 2),
                                 "token_usage": dict(MC.USAGE)}
    json.dump(agg, open(out, "w"), indent=2)
    print(f"Saved {out}; augmentation cost ${MC.spent_usd():.2f}")


if __name__ == "__main__":
    main()
