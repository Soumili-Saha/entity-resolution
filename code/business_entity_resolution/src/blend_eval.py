"""Entity-level evaluation of the cross-encoder (CE) and CE + LightGBM blends on the fold-0 S1s of the
500k-S1 handoff sample (handoff/ce_training_pairs/train_pairs_*, CE scores from handoff/xlmr_base_ce_v1).

Pair AUC/logloss do not show the leaderboard metric, so every scorer is pushed through the same
decision layer and scored as macro F0.5 over ALL fold-0 S1s (singletons included). To keep the
decision tuning honest, fold-0 S1s are split in two halves (crc32 % 10 == 0 vs == 5): the decision
is tuned on one half and scored on the other, both ways, and the two held-out scores are averaged.

Usage: python -m src.blend_eval [--ce handoff/xlmr_base_ce_v1]
"""
import argparse
import glob
import os
import zlib

import numpy as np
import pandas as pd

from src.decide import FAST_GRID, decide, tune
from src.evaluate import f05
from src.io_utils import load_ground_truth


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def held_out(df, col, gold, halves, country, grid=FAST_GRID):
    """(cross-fitted macro F0.5, {country: score}, params tuned on all) for probability column col."""
    d = df[["s1_id", "pool_id", col]].rename(columns={col: "prob"})
    per = {}
    for a, b in ((0, 1), (1, 0)):
        ta = d[d["s1_id"].isin(set(halves[a]))]
        tb = d[d["s1_id"].isin(set(halves[b]))]
        params, _ = tune(ta, gold, halves[a], verbose=False, grid=grid, o2o_opts=(True,))
        pred = decide(tb, halves[b], params)
        per.update({s: f05(pred[s], gold.get(s, ())) for s in halves[b]})
    all_ids = halves[0] + halves[1]
    params, _ = tune(d, gold, all_ids, verbose=False, grid=grid, o2o_opts=(True,))
    by_c = pd.Series(per).groupby(pd.Series({s: country.get(s) for s in per})).mean().to_dict()
    return float(np.mean(list(per.values()))), by_c, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce", default="handoff/xlmr_base_ce_v1")
    args = ap.parse_args()
    gold = load_ground_truth("dataset")
    pairs = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob("handoff/ce_training_pairs/train_pairs_part*.parquet"))],
                      ignore_index=True)
    pairs = pairs[pairs["fold"] == 0].drop(columns=["fold"])
    ce = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(args.ce, "ce_oof_fold0_part*.parquet")))],
                   ignore_index=True)
    ids = list(pd.unique(pairs["s1_id"]))
    df = pairs[pairs["lgbm_prob"] >= 0.001].merge(ce, on=["s1_id", "pool_id"], how="left")
    print(f"fold-0: {len(ids):,} S1 | {len(df):,} pairs with lgbm_prob >= 0.001 | "
          f"CE missing on {df['ce_prob'].isna().sum():,}", flush=True)
    df["ce_prob"] = df["ce_prob"].fillna(0.0)
    s1 = pd.read_parquet("artefacts/train/s1.parquet", columns=["entity_id", "country_norm"])
    country = dict(zip(s1["entity_id"], s1["country_norm"]))
    h = {s: zlib.crc32(s.encode()) % 10 for s in ids}
    halves = [[s for s in ids if h[s] == 0], [s for s in ids if h[s] == 5]]
    lg, lc = logit(df["lgbm_prob"].to_numpy()), logit(df["ce_prob"].to_numpy())
    cols = {"lgbm": df["lgbm_prob"].to_numpy(), "ce": df["ce_prob"].to_numpy()}
    for w in (0.3, 0.5, 0.7, 0.85):
        cols[f"prob_avg_w{w}"] = w * cols["ce"] + (1 - w) * cols["lgbm"]
        cols[f"logit_avg_w{w}"] = 1 / (1 + np.exp(-(w * lc + (1 - w) * lg)))
    for name, v in cols.items():
        df[name] = v
        score, by_c, params = held_out(df, name, gold, halves, country)
        print(f"  {name:16s} held-out macro F0.5 {score:.4f} | " +
              " | ".join(f"{c} {x:.4f}" for c, x in sorted(by_c.items())) + f" | params(all) {params}", flush=True)


if __name__ == "__main__":
    main()
