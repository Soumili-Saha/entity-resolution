"""Add pairs that only the second pipeline predicted and the first pipeline never had as candidates.

  python -m src.union_submit --base output/v12_mylgbm --other handoff/lgbm_v2_xlmr_base_v2_matches/matching_results.tsv.gz --out output/v13_union
"""
import argparse
import os
import subprocess
import sys
from collections import Counter

from src.io_utils import write_tsv
from src.local_score import read_id_lists
from src.loop_eval import test_country


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="output/v12_mylgbm")
    ap.add_argument("--other", default="handoff/lgbm_v2_xlmr_base_v2_matches/matching_results.tsv.gz")
    ap.add_argument("--out", default="output/v13_union")
    ap.add_argument("--countries", default="", help="comma list; only add pairs for these S1 countries (default: all)")
    args = ap.parse_args()
    sub = read_id_lists(os.path.join(args.base, "matching_results.tsv"))
    cand = read_id_lists(os.path.join(args.base, "candidate_pairs.tsv"))
    other = read_id_lists(os.path.expanduser(args.other))
    assert set(sub) == set(cand), "S1 sets differ"
    cc = test_country()
    used = {m for ms in sub.values() for m in ms}
    add = {}
    keep = set(args.countries.split(",")) if args.countries else None
    for s, ms in other.items():
        if keep is not None and cc[s] not in keep:
            continue
        cs, ss = set(cand.get(s, ())), set(sub.get(s, ()))
        for m in ms:
            if m not in ss and m not in cs and m not in used:  # never a candidate, pool record still free
                add.setdefault(s, []).append(m)
                used.add(m)
    n = sum(len(v) for v in add.values())
    print(f"added {n:,} pairs on {len(add):,} S1 | by country {dict(Counter(cc[s] for s, v in add.items() for _ in v))}")
    ids = list(sub)
    os.makedirs(args.out, exist_ok=True)
    write_tsv(ids, {s: sub[s] + add.get(s, []) for s in ids}, os.path.join(args.out, "matching_results.tsv"), "matched_entity_ids")
    write_tsv(ids, {s: cand[s] + add.get(s, []) for s in ids}, os.path.join(args.out, "candidate_pairs.tsv"), "candidate_entity_ids")
    subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", os.path.join(args.out, "matching_results.tsv"),
                    "--candidate", os.path.join(args.out, "candidate_pairs.tsv"), "--test-dir", "dataset/test", "--check-ids"])


if __name__ == "__main__":
    main()
