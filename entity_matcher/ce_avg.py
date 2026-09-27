"""Average several cross-encoders' scores (mean of logits) into one CE folder, so an adapter ensemble enters the
stacker as a single column with exactly the coverage of its members (pairs missing in any member are dropped).

Usage: python -m entity_matcher.ce_avg --out precomputed/ce_qwen_ens_out precomputed/ce_qwen_out precomputed/ce_qwen_s2_out
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from entity_matcher.blend_eval import logit

KINDS = {"ce_oof_fold0": ["ce_oof_fold0_part*.parquet"],
         "ce_test": ["ce_test_part*.parquet", "ce_test_contested_part*.parquet"],
         "ce_test_france": ["ce_test_france_part*.parquet"]}


def read(d, pats):
    files = [f for p in pats for f in sorted(glob.glob(os.path.join(d, p)))]
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).drop_duplicates(["s1_id", "pool_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for kind, pats in KINDS.items():
        parts = [read(d, pats) for d in args.dirs]
        if any(p is None for p in parts):
            print(f"{kind}: missing in some member, skipped", flush=True)
            continue
        m = parts[0].rename(columns={"ce_prob": "p0"})
        for i, p in enumerate(parts[1:], 1):
            m = m.merge(p.rename(columns={"ce_prob": f"p{i}"}), on=["s1_id", "pool_id"])
        z = np.mean([logit(m[f"p{i}"].to_numpy()) for i in range(len(parts))], axis=0)
        corr = np.corrcoef(logit(m["p0"].to_numpy()), logit(m["p1"].to_numpy()))[0, 1]
        out = m[["s1_id", "pool_id"]].assign(ce_prob=(1 / (1 + np.exp(-z))).astype(np.float32))
        out.to_parquet(os.path.join(args.out, f"{kind}_part00.parquet"))
        open(os.path.join(args.out, f"{kind}.DONE"), "w").close()
        print(f"{kind}: {len(out):,} pairs (members {[len(p) for p in parts]}), logit corr p0-p1 {corr:.4f}", flush=True)


if __name__ == "__main__":
    main()
