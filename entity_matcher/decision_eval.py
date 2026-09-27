"""Compare decision rules on saved stacker out-of-fold probabilities (artefacts/exp/stack_oof_<tag>.parquet)
with the cross-fitted halves protocol of entity_matcher.blend_eval: the rule's parameter is tuned on one half of the
fold-0 S1s and scored on the other, both ways.

Usage: python -m entity_matcher.decision_eval [--tag sample] [--cols stack_pool,blend]
"""
import argparse
import os
import glob
import zlib

import pandas as pd

from entity_matcher.blend_eval import held_out
from entity_matcher.decide import FAST_GRID
from entity_matcher.io_utils import load_ground_truth

RULES = {
    "expf": FAST_GRID,
    "expf_exact": [{"method": "expf_exact", "alpha": a} for a in (0.8, 1.0, 1.2, 1.5, 2.0, 3.0)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sample")
    ap.add_argument("--cols", default="stack_pool")
    ap.add_argument("--pairs", default="precomputed/ce/train_pairs_part*.parquet", help="pair files defining the fold-0 S1s")
    args = ap.parse_args()
    if os.name == "nt":  # keep full CPU when the laptop is locked (Windows EcoQoS)
        from entity_matcher.no_throttle import disable_throttling
        disable_throttling()
    gold = load_ground_truth("dataset")
    df = pd.read_parquet(f"artefacts/exp/stack_oof_{args.tag}.parquet")
    # all fold-0 S1s of the pair table, including those without any pair >= 0.001 (they get empty predictions)
    s1s = pd.concat([pd.read_parquet(f, columns=["s1_id"]) for f in sorted(glob.glob(args.pairs))])["s1_id"].unique()
    ids = [s for s in s1s if zlib.crc32(s.encode()) % 5 == 0]
    s1 = pd.read_parquet("artefacts/train/s1.parquet", columns=["entity_id", "country_norm"])
    country = dict(zip(s1["entity_id"], s1["country_norm"]))
    hh = {s: zlib.crc32(s.encode()) % 10 for s in ids}
    halves = [[s for s in ids if hh[s] == 0], [s for s in ids if hh[s] == 5]]
    for col in args.cols.split(","):
        for rule, grid in RULES.items():
            score, by_c, p = held_out(df, col, gold, halves, country, grid)
            print(f"  {col:12s} {rule:10s} held-out macro F0.5 {score:.4f} | " +
                  " | ".join(f"{c} {x:.4f}" for c, x in sorted(by_c.items())) + f" | params(all) {p}", flush=True)


if __name__ == "__main__":
    main()
