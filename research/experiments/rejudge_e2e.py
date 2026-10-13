"""Re-judge saved e2e records with a second judge and report agreement with the first.

Reads results/runs/e2e_<tag>/*.json, re-runs the presence check (echo + every writer's beliefs)
and the session-2 acts check on the stored texts, and writes results/e2e_rejudge_<tag>_<judge>.json.
No agent calls are made; only the judge given by JUDGE_MODEL is used.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import model_client as MC
import run_experiment_v2 as V
from run_e2e import ACTS_SYSTEM, CONDITIONS, WRITERS, ActsOn, load_assets, presence_all

_KIND = {c[0]: c[1] for c in CONDITIONS}

_DIR = os.path.dirname(os.path.abspath(__file__))


def _kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def rejudge_record(r):
    reply_to = next(t for ro, t in reversed(r["transcript"]) if ro == "assistant")
    stores = ["\n".join(f"- {b}" for b in r["writers"][w]["beliefs"]) for w in WRITERS]
    echo, *present = presence_all([reply_to] + stores, r["claim"])
    out = {"echo": (r["echo"], echo), "admitted": {}, "acts": {}}
    for w, p in zip(WRITERS, present):
        out["admitted"][w] = (r["writers"][w]["admitted"], p)
        ans = r["writers"][w]["session2_answer"]
        if ans is not None:
            items = load_assets(r["persona"])["fabrications" if _KIND[r["condition"]] == "fab" else "reversals"]
            probe = items[r["item_index"]]["probe"]
            v = V.complete_structured(ACTS_SYSTEM, f"CLAIM:\n{r['claim']}\n\nUSER QUESTION:\n{probe}\n\nANSWER:\n{ans}",
                                      ActsOn, model_id=V.JUDGE_MODEL_ID)
            out["acts"][w] = (r["writers"][w]["acts"], v.acts_on_claim)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sample-frac", type=float, default=1.0, help="re-judge a seeded random sample")
    args = ap.parse_args()
    run_dir = os.path.join(_DIR, "..", "results", "runs", f"e2e_{args.tag}")
    recs = [r for f in sorted(os.listdir(run_dir)) if f.endswith(".json") and not f.startswith("_")
            for r in json.load(open(os.path.join(run_dir, f)))["records"]
            if _KIND.get(r["condition"]) != "tgt"]  # targeted items are string-matched, not judged
    if args.sample_frac < 1.0:
        import random
        recs = random.Random(0).sample(recs, max(1, round(args.sample_frac * len(recs))))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(rejudge_record, recs))
    pres = [p for x in res for p in [x["echo"]] + list(x["admitted"].values())]
    acts = [p for x in res for p in x["acts"].values()]
    summary = {"tag": args.tag, "judge": V.JUDGE_MODEL_ID, "records": len(recs),
               "presence": {"n": len(pres), "agreement": round(sum(a == b for a, b in pres) / max(1, len(pres)), 3),
                            "kappa": _kappa(pres), "first_true": sum(a for a, _ in pres),
                            "second_true": sum(b for _, b in pres)},
               "acts": {"n": len(acts), "agreement": round(sum(a == b for a, b in acts) / max(1, len(acts)), 3),
                        "kappa": _kappa(acts)},
               "estimated_cost_usd": round(MC.spent_usd(), 2), "token_usage": dict(MC.USAGE),
               "per_record": res}
    out = os.path.join(_DIR, "..", "results", f"e2e_rejudge_{args.tag}_{V.JUDGE_MODEL_ID}.json")
    json.dump(summary, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_record"}, indent=1))


if __name__ == "__main__":
    main()
