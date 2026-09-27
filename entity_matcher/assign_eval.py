"""Empty-address assignment rule, evaluated on saved stacker out-of-fold probabilities.

Training ground truth: 97.7% of empty-address pool records belong to some S1 (74% of all records).
Rule: after the normal decision, add every still-unassigned empty-address candidate for which this S1
is the record's top S1 (by LightGBM probability over all S1s of the pair table) and the stacker
probability is >= t. t is tuned on one half of the fold-0 S1s and scored on the other, both ways.

On the precomputed sample only ~23% of training S1s are present, so a record's true S1 is often missing
and the rule is penalised more than it would be on test (where all S1s are present): conservative.

Usage: python -m entity_matcher.assign_eval --tag large --col stack_extra --alpha 2.0
"""
import argparse
import os
import glob
import zlib

import numpy as np
import pandas as pd

from entity_matcher.decide import decide
from entity_matcher.evaluate import f05
from entity_matcher.io_utils import load_ground_truth
from entity_matcher.stack_eval import pool_context


def apply_rule(df, pred, t):
    """pred + unassigned empty-address candidates where this S1 is the record's top S1 and prob >= t."""
    taken = {p for v in pred.values() for p in v}
    add = df[(df["eaddr"]) & (df["top"]) & (df["prob"] >= t) & (~df["pool_id"].isin(taken))]
    out = {s: list(v) for s, v in pred.items()}
    for s, p in zip(add["s1_id"], add["pool_id"]):
        if s in out and p not in out[s]:
            out[s].append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="large")
    ap.add_argument("--col", default="stack_extra")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--pairs", default="precomputed/ce/train_pairs_part*.parquet")
    ap.add_argument("--s1_from", default="", help="glob of pair files defining the evaluated fold-0 S1s")
    args = ap.parse_args()
    if os.name == "nt":  # keep full CPU when the laptop is locked (Windows EcoQoS)
        from entity_matcher.no_throttle import disable_throttling
        disable_throttling()
    gold = load_ground_truth("dataset")
    df = pd.read_parquet(f"artefacts/exp/stack_oof_{args.tag}.parquet")[["s1_id", "pool_id", args.col]]
    df = df.rename(columns={args.col: "prob"})
    files = sorted(glob.glob(args.pairs))
    pcol = "lgbm_prob" if "train_pairs" in args.pairs else "prob"
    allp = pd.concat([pd.read_parquet(f, columns=["s1_id", "pool_id", pcol], filters=[(pcol, ">=", 0.001)])
                      for f in files]).rename(columns={"prob": "lgbm_prob"})
    df = df.merge(allp, on=["s1_id", "pool_id"], how="left")
    df["top"] = pool_context(allp, df)["pool_is_top"].to_numpy() > 0
    pool = pd.concat([pd.read_parquet(f"artefacts/train/s{k}.parquet", columns=["entity_id", "addr_clean"]) for k in (2, 3)])
    df["eaddr"] = df["pool_id"].isin(set(pool.loc[pool["addr_clean"].str.len() == 0, "entity_id"])).to_numpy()
    src = sorted(glob.glob(args.s1_from)) if args.s1_from else files
    ids = [s for s in pd.concat([pd.read_parquet(f, columns=["s1_id"]) for f in src])["s1_id"].unique()
           if zlib.crc32(s.encode()) % 5 == 0]
    hh = {s: zlib.crc32(s.encode()) % 10 for s in ids}
    halves = [[s for s in ids if hh[s] == 0], [s for s in ids if hh[s] == 5]]
    params = {"method": "expf", "alpha": args.alpha, "one_to_one": True}
    ts = [1.01, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]  # 1.01 = rule off
    res = {}
    for h in (0, 1):
        sub = df[df["s1_id"].isin(set(halves[h]))]
        pred = decide(sub[["s1_id", "pool_id", "prob"]], halves[h], params)
        for t in ts:
            p2 = apply_rule(sub, pred, t)
            res[(h, t)] = np.mean([f05(p2[s], gold.get(s, ())) for s in halves[h]])
    for t in ts:
        print(f"  t={t:<5} half0 {res[(0, t)]:.4f} half1 {res[(1, t)]:.4f} mean {(res[(0, t)] + res[(1, t)]) / 2:.4f}", flush=True)
    cross = []
    for a, b in ((0, 1), (1, 0)):
        tb = max(ts, key=lambda t: res[(a, t)])
        cross.append((res[(b, tb)], res[(b, 1.01)], tb))
    print(f"cross-fitted: rule {np.mean([c[0] for c in cross]):.4f} vs off {np.mean([c[1] for c in cross]):.4f} "
          f"(t chosen {[c[2] for c in cross]})", flush=True)


if __name__ == "__main__":
    main()
