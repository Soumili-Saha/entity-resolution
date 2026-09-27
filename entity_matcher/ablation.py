"""Feature ablation on cached training features: stage-1 LightGBM trained on folds 1-4 and scored on
fold 0 (macro F0.5 with the decision tuned on fold 0), with and without a group of feature columns.

Usage: python -m entity_matcher.ablation --config configs/experiment.yaml --drop n_extra_b,n_extra_a,...
"""
import argparse
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from entity_matcher.decide import tune
from entity_matcher.evaluate import fold_of
from entity_matcher.io_utils import load_ground_truth
from entity_matcher.ranker import ID_COLS, lgb_params
from entity_matcher.run import art_dir

DECOY = ["n_extra_b", "n_extra_a", "n_extra_b_len", "a_extra_b", "a_extra_a", "a_extra_b_frac",
         "legal_conflict", "legal_both", "house_prefix", "house_absdiff", "house_lev", "house_samelen"]


def run(feats, y, val, cols, cfg, gold, val_ids):
    """Train on non-val rows, return (fold-0 macro F0.5, logloss, trees)."""
    params = lgb_params(cfg)
    dtr = lgb.Dataset(feats.loc[~val, cols].to_numpy(np.float32), y[~val], params=params)
    Xva = feats.loc[val, cols].to_numpy(np.float32)
    dva = lgb.Dataset(Xva, y[val], reference=dtr, params=params)
    m = lgb.train(params, dtr, 5000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    df = feats.loc[val, ["s1_id", "pool_id"]].copy()
    df["prob"] = m.predict(Xva, num_iteration=m.best_iteration)
    _, score = tune(df, gold, val_ids, verbose=False)
    return score, m.best_score["valid_0"]["binary_logloss"], m.best_iteration


def main():
    """Compare all features vs. all minus the dropped group on the same fold-0 validation S1s."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--drop", default=",".join(DECOY))
    args = ap.parse_args()
    if os.name == "nt":
        from entity_matcher.no_throttle import disable_throttling
        disable_throttling()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    gold = load_ground_truth(cfg["paths"]["data_dir"])
    fd = os.path.join(art_dir(cfg, "train"), "feats")
    feats = pd.concat([pd.read_parquet(os.path.join(fd, p)) for p in sorted(os.listdir(fd))], ignore_index=True)
    y = np.array([int(p in gold.get(s, ())) for s, p in zip(feats["s1_id"], feats["pool_id"])], dtype=np.int8)
    uniq = pd.unique(feats["s1_id"])
    val_ids = [s for s in uniq if fold_of(s) == 0]
    val = feats["s1_id"].isin(set(val_ids)).to_numpy()
    all_cols = [c for c in feats.columns if c not in ID_COLS]
    drop = [c for c in args.drop.split(",") if c in all_cols]
    print(f"{len(uniq):,} S1 ({len(val_ids):,} validation), {len(all_cols)} features, dropping {len(drop)}", flush=True)
    for name, cols in (("without group", [c for c in all_cols if c not in drop]), ("with group", all_cols)):
        s, ll, it = run(feats, y, val, cols, cfg, gold, val_ids)
        print(f"  {name:14s}: fold-0 macro F0.5 {s:.4f} | logloss {ll:.5f} | trees {it}", flush=True)


if __name__ == "__main__":
    main()
