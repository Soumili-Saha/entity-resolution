"""Test candidate pairs of S1s whose country is absent from training (open set, derived from the data).

unseen_s1()     : test S1 ids of countries not present in train_source1.tsv
unseen_pairs()  : handoff test pairs (lgbm >= 0.001) UNION step-3 test candidates (prob >= min_prob) of those S1s

  python -m src.unseen_pairs --out artefacts/qwen_ce/inputs/unseen_all.parquet   # pair list for qwen_score.py
"""
import argparse
import glob
import os

import pandas as pd

from src.io_utils import read_tsv

KEY = ["s1_id", "pool_id"]


def read_parts(pattern, columns=None):
    return pd.concat([pd.read_parquet(f, columns=columns) for f in sorted(glob.glob(pattern))], ignore_index=True)


def unseen_s1(data_dir="dataset"):
    s1 = read_tsv(os.path.join(data_dir, "test", "test_source1.tsv"), usecols=["entity_id", "country"])
    seen = set(read_tsv(os.path.join(data_dir, "train", "train_source1.tsv"), usecols=["country"])["country"])
    return set(s1.loc[~s1["country"].isin(seen), "entity_id"])


def unseen_pairs(data_dir="dataset", min_prob=0.001, with_step3=True):
    uns = unseen_s1(data_dir)
    h = read_parts("handoff/ce_training_pairs/test_pairs_part*.parquet", columns=KEY + ["lgbm_prob"])
    parts = [h[h["s1_id"].isin(uns)][KEY]]
    if with_step3:
        s3 = read_parts("handoff/lgbm_ranker_v1/probs_model_full_part*.parquet", columns=KEY + ["prob"])
        parts.append(s3[s3["s1_id"].isin(uns) & (s3["prob"] >= min_prob)][KEY])
    return pd.concat(parts).drop_duplicates().reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--min_prob", type=float, default=0.001)
    ap.add_argument("--out", default="artefacts/qwen_ce/inputs/unseen_all.parquet")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    p = unseen_pairs(args.data_dir, args.min_prob)
    p.to_parquet(args.out, index=False)
    print(f"{len(p):,} unseen-country test pairs on {p['s1_id'].nunique():,} S1 -> {args.out}")


if __name__ == "__main__":
    main()
