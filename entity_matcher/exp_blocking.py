"""Blocking recall experiment on a sample of validation S1 entities against the FULL train pool
(the full pool matters: recall at top-k depends on how many distractors exist).

Usage: python -m entity_matcher.exp_blocking --n 20000 [--emb]   (--emb needs cached train embeddings)
"""
import argparse
import os
import time

import numpy as np
import yaml

from entity_matcher.evaluate import is_val
from entity_matcher.io_utils import load_ground_truth
from entity_matcher.run import art_dir, load_normalized

COLS = ["entity_id", "name_skel", "name_alt_skel", "name_ns", "addr_clean", "country_norm"]


def recall_at(cands, rank_col, gold_pairs, ks):
    """Recall of true pairs within rank <= k for each k."""
    c = cands[["s1_id", "pool_id", rank_col]]
    m = gold_pairs.merge(c, on=["s1_id", "pool_id"], how="left")
    return {k: float((m[rank_col] <= k).mean()) for k in ks}


def main():
    """Run token (and optionally embedding) blocking on a validation sample and print recall."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--max_df", type=int, default=3000)
    ap.add_argument("--emb", action="store_true")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    t0 = time.time()
    gold = load_ground_truth(cfg["paths"]["data_dir"])
    s1, pool = load_normalized(cfg, "train", columns=COLS)
    val_idx = np.flatnonzero([is_val(i) for i in s1["entity_id"]])
    rng = np.random.RandomState(0)
    samp = np.sort(rng.choice(val_idx, args.n, replace=False))
    q = s1.iloc[samp].reset_index(drop=True)
    print(f"loaded [{time.time() - t0:.0f}s]", flush=True)

    import pandas as pd
    rows = [(i, m) for i in q["entity_id"] for m in gold.get(i, ())]
    gold_pairs = pd.DataFrame(rows, columns=["s1_id", "pool_id"])
    print(f"{len(q)} S1, {len(gold_pairs)} true pairs", flush=True)

    from entity_matcher.blocking import block_all
    e_s1 = e_pool = None
    if args.emb:
        d = art_dir(cfg, "train")
        e_s1 = np.ascontiguousarray(np.load(os.path.join(d, "emb_s1.npy"), mmap_mode="r")[samp])
        e_pool = np.load(os.path.join(d, "emb_pool.npy"), mmap_mode="r")
    cands = block_all(q, pool, e_s1, e_pool, token_k=args.k, token_max_df=args.max_df, embed_k=args.k, cap=None)
    parts = ["tok"] + (["emb"] if args.emb else [])
    print(f"blocking done: {len(cands):,} pairs [{time.time() - t0:.0f}s]", flush=True)
    cands["s1_id"] = q["entity_id"].to_numpy()[cands["s1_idx"].to_numpy()]
    cands["pool_id"] = pool["entity_id"].to_numpy()[cands["pool_idx"].to_numpy()]
    ks = [5, 10, 20, 30, 50]
    for col in [c for c in cands.columns if c.endswith("_rank")]:
        print(col, {k: round(v, 4) for k, v in recall_at(cands, col, gold_pairs, ks).items()})
    for cap in (20, 30, 40, 60, 80):
        sub = cands[cands["best_rank"] <= cap // len(parts) + (cap % len(parts) > 0)]
        n = len(sub) / len(q)
        rec = gold_pairs.merge(sub[["s1_id", "pool_id"]], on=["s1_id", "pool_id"], how="inner").shape[0] / len(gold_pairs)
        print(f"union with per-pass rank <= {cap // len(parts)}: {n:.1f} cands/S1, recall {rec:.4f}")
    print(f"union all: {len(cands) / len(q):.1f} cands/S1 [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
