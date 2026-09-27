"""Add pairs predicted by the companion pipeline to a stacker output, keeping the one-to-one structure.

Default mode (v13_union): add a companion pair only if the base never had it as a candidate (a blocking miss)
and its S2/S3 record is not matched yet.

--rescue_probs mode (v14_last): add companion pairs that ARE base candidates, were rejected by the base decision,
have a free S2/S3 record and a stacker probability >= --rescue_min.

  python -m entity_matcher.union_submit --base output/v12_mylgbm --out output/v13_union
  python -m entity_matcher.union_submit --base output/v13_union --out output/v14_last \
      --rescue_probs artefacts/exp/E001_mylgbm/final_test_probs.parquet --rescue_min 0.3
"""
import argparse
import os
import subprocess
import sys
from collections import Counter

import pandas as pd

from entity_matcher.io_utils import write_tsv
from entity_matcher.local_score import read_id_lists
from entity_matcher.loop_eval import test_country


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="output/v12_mylgbm")
    ap.add_argument("--other", default="precomputed/my_pipeline/output_ce/matching_results.tsv.gz")
    ap.add_argument("--out", default="output/v13_union")
    ap.add_argument("--countries", default="", help="comma list; only add pairs for these S1 countries (default: all)")
    ap.add_argument("--rescue_probs", default="", help="parquet (s1_id, pool_id, prob): switch to rescue mode")
    ap.add_argument("--rescue_min", type=float, default=0.3)
    args = ap.parse_args()
    sub = read_id_lists(os.path.join(args.base, "matching_results.tsv"))
    cand = read_id_lists(os.path.join(args.base, "candidate_pairs.tsv"))
    other = read_id_lists(args.other)
    assert set(sub) == set(cand), "S1 sets differ"
    prob = None
    if args.rescue_probs:
        p = pd.read_parquet(args.rescue_probs)
        prob = dict(zip(zip(p["s1_id"], p["pool_id"]), p["prob"]))
    cc = test_country()
    used = {m for ms in sub.values() for m in ms}
    add = {}
    keep = set(args.countries.split(",")) if args.countries else None
    for s, ms in other.items():
        if keep is not None and cc[s] not in keep:
            continue
        cs, ss = set(cand.get(s, ())), set(sub.get(s, ()))
        for m in ms:
            if m in ss or m in used:
                continue
            ok = (m in cs and prob.get((s, m), 0.0) >= args.rescue_min) if prob is not None else m not in cs
            if ok:
                add.setdefault(s, []).append(m)
                used.add(m)
    n = sum(len(v) for v in add.values())
    print(f"added {n:,} pairs on {len(add):,} S1 | by country {dict(Counter(cc[s] for s, v in add.items() for _ in v))}")
    ids = list(sub)
    os.makedirs(args.out, exist_ok=True)
    write_tsv(ids, {s: sub[s] + add.get(s, []) for s in ids}, os.path.join(args.out, "matching_results.tsv"), "matched_entity_ids")
    write_tsv(ids, {s: cand[s] + add.get(s, []) for s in ids}, os.path.join(args.out, "candidate_pairs.tsv"), "candidate_entity_ids")
    subprocess.run([sys.executable, "utils/validate_outputs.py", "--matching", os.path.join(args.out, "matching_results.tsv"),
                    "--candidate", os.path.join(args.out, "candidate_pairs.tsv"), "--test-dir", "dataset/test", "--check-ids"])


if __name__ == "__main__":
    main()
