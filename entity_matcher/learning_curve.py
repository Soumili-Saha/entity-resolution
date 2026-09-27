"""Learning curve: does more training data still help the matcher?

Fixed validation = the S1s of fold 0 in the cached training features. Stage-1 LightGBM models are
trained on growing subsets of the remaining S1s and scored (macro F0.5, decision tuned on the
validation probabilities) on the SAME validation S1s, so the numbers are directly comparable.

Usage: python -m entity_matcher.learning_curve [--feats artefacts/train/feats]
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


def main():
    """Train on 12.5% / 25% / 50% / 100% of the non-validation S1s and print validation F0.5."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--feats", default="artefacts/train/feats")
    ap.add_argument("--lr", type=float, default=0.1)
    args = ap.parse_args()
    if os.name == "nt":  # keep all cores when the laptop is locked (Windows EcoQoS throttling)
        from entity_matcher.no_throttle import disable_throttling
        disable_throttling()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    gold = load_ground_truth(cfg["paths"]["data_dir"])
    feats = pd.concat([pd.read_parquet(os.path.join(args.feats, p)) for p in sorted(os.listdir(args.feats))],
                      ignore_index=True)
    cols = [c for c in feats.columns if c not in ID_COLS]
    y = np.array([int(p in gold.get(s, ())) for s, p in zip(feats["s1_id"], feats["pool_id"])], dtype=np.int8)
    s1 = feats["s1_id"].to_numpy()
    uniq = pd.unique(s1)
    fold = {i: fold_of(i, cfg["validation"]["n_folds"]) for i in uniq}
    val_ids = [i for i in uniq if fold[i] == 0]
    train_ids = np.array([i for i in uniq if fold[i] != 0])
    np.random.RandomState(0).shuffle(train_ids)
    va = feats["s1_id"].isin(set(val_ids)).to_numpy()  # hash-based (np.isin on strings is O(n*m))
    Xva = feats.loc[va, cols].to_numpy(np.float32)
    params = dict(lgb_params(cfg), learning_rate=args.lr)
    print(f"validation: {len(val_ids):,} S1 fixed | training pool: {len(train_ids):,} S1", flush=True)
    for frac in (0.125, 0.25, 0.5, 1.0):
        sub = set(train_ids[:int(len(train_ids) * frac)])
        tr = feats["s1_id"].isin(sub).to_numpy()
        dtr = lgb.Dataset(feats.loc[tr, cols].to_numpy(np.float32), y[tr])
        dva = lgb.Dataset(Xva, y[va], reference=dtr)
        m = lgb.train(params, dtr, 5000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        df = feats.loc[va, ["s1_id", "pool_id"]].copy()
        df["prob"] = m.predict(Xva, num_iteration=m.best_iteration)
        _, score = tune(df, gold, val_ids, verbose=False)
        print(f"  train S1 {len(sub):>7,}: val logloss {m.best_score['valid_0']['binary_logloss']:.5f} | "
              f"val macro F0.5 {score:.4f} | trees {m.best_iteration}", flush=True)


if __name__ == "__main__":
    main()
