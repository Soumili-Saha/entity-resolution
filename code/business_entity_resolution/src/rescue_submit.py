"""Add pipeline-v2 pairs that are pipeline-v1 candidates but lost the one-to-one decision, when the stacker still rates them.

A pair is added if (1) the second pipeline predicted it, (2) it is in the base candidate set but not matched,
(3) its pool record is not matched to any S1 yet, and (4) the final stacker probability is >= --min_prob.

  python -m src.rescue_submit --base output/v13_union --out output/v14_last
"""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter

import pandas as pd

from src.io_utils import write_tsv
from src.local_score import read_id_lists
from src.loop_eval import test_country


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="output/v13_union")
    ap.add_argument("--other", default="handoff/lgbm_v2_xlmr_base_v2_matches/matching_results.tsv.gz")
    ap.add_argument("--probs", default="artefacts/exp/E001_mylgbm/final_test_probs.parquet")
    ap.add_argument("--min_prob", type=float, default=0.3)
    ap.add_argument("--out", default="output/v14_last")
    args = ap.parse_args()
    sub = read_id_lists(os.path.join(args.base, "matching_results.tsv"))
    cand = read_id_lists(os.path.join(args.base, "candidate_pairs.tsv"))
    other = read_id_lists(args.other)
    pr = pd.read_parquet(args.probs, columns=["s1_id", "pool_id", "prob"]).drop_duplicates(["s1_id", "pool_id"])
    prob = dict(zip(zip(pr["s1_id"], pr["pool_id"]), pr["prob"]))
    cc = test_country()
    used = {m for ms in sub.values() for m in ms}
    add = {}
    for s, ms in other.items():
        cs, ss = set(cand.get(s, ())), set(sub.get(s, ()))
        for m in ms:
            if m not in ss and m in cs and m not in used and prob.get((s, m), 0.0) >= args.min_prob:
                add.setdefault(s, []).append(m)
                used.add(m)
    n = sum(len(v) for v in add.values())
    print(f"added {n:,} pairs on {len(add):,} S1 | by country {dict(Counter(cc[s] for s, v in add.items() for _ in v))}")
    ids = list(sub)
    os.makedirs(args.out, exist_ok=True)
    write_tsv(ids, {s: sub[s] + add.get(s, []) for s in ids}, os.path.join(args.out, "matching_results.tsv"), "matched_entity_ids")
    write_tsv(ids, cand, os.path.join(args.out, "candidate_pairs.tsv"), "candidate_entity_ids")
    with open(os.path.join(args.out, "rescue.json"), "w") as f:
        json.dump({"base": args.base, "added_from": args.other, "probs": args.probs, "min_prob": args.min_prob,
                   "rule": "pipeline-v2 pair, in base candidates, pool record unused, stacker prob >= min_prob",
                   "added": n}, f, indent=1)
    subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", os.path.join(args.out, "matching_results.tsv"),
                    "--candidate", os.path.join(args.out, "candidate_pairs.tsv"), "--test-dir", "dataset/test", "--check-ids"])


if __name__ == "__main__":
    main()
