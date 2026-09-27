"""Leave-one-country-out LightGBM on the local training feature sample: how much of the pair scorer's
unseen-country loss comes from country-specific features? Trains on India only and scores US (the same US half
for every variant), next to a US-trained reference. Feature sets are named groups of stage-3 features.

Usage: python -m entity_matcher.loco_lgbm
"""
import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from entity_matcher.blend_eval import logit
from entity_matcher.decide import decide
from entity_matcher.evaluate import f05
from entity_matcher.io_utils import load_ground_truth

PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
          "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": 42}
EMB = ["emb_score", "emb_rank", "s1_rank_emb_score", "s1_gap_emb_score", "cand_rank_emb_score", "cand_gap_emb_score",
       "best_rank"]
COUNTRY = ["lm_jac", "lm_inter", "legal_conflict", "legal_both", "n_alt_tset", "b_has_alt"]
META = ["source", "len_name_a", "len_name_b", "len_addr_a", "len_addr_b"]


def main():
    d = pd.read_parquet("artefacts/train/feats/part_000.parquet")
    s1 = pd.read_parquet("artefacts/train/s1.parquet", columns=["entity_id", "country_norm"])
    d["c"] = d["s1_id"].map(dict(zip(s1["entity_id"], s1["country_norm"])))
    gold = load_ground_truth("dataset")
    d["y"] = [int(p in gold.get(s, ())) for s, p in zip(d["s1_id"], d["pool_id"])]
    half = d["s1_id"].map(lambda s: zlib.crc32(s.encode()) % 2)
    us_test = (d["c"] == "us") & (half == 0)
    us_train = (d["c"] == "us") & (half == 1)
    india = d["c"] == "india"
    feats = [c for c in d.columns if c not in ("s1_id", "pool_id", "c", "y")]
    sets = {"all": feats, "no_emb": [f for f in feats if f not in EMB],
            "no_country": [f for f in feats if f not in COUNTRY],
            "neutral": [f for f in feats if f not in EMB + COUNTRY + META]}
    te = d.loc[us_test, ["s1_id", "pool_id", "y"]].copy()
    ids = list(te["s1_id"].unique())
    print(f"rows {len(d):,} | India {int(india.sum()):,} | US train {int(us_train.sum()):,} | US test {len(te):,} "
          f"({len(ids):,} S1)", flush=True)
    runs = [("US-trained ref", us_train, "all")] + [(f"India-only {k}", india, k) for k in sets]
    for name, tr, fs in runs:
        X, y = d.loc[tr, sets[fs]], d.loc[tr, "y"]
        va = d.loc[tr, "s1_id"].map(lambda s: zlib.crc32(s.encode()) % 10 == 3).to_numpy()
        m = lgb.train(PARAMS, lgb.Dataset(X[~va], y[~va]), 3000, valid_sets=[lgb.Dataset(X[va], y[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        p = m.predict(d.loc[us_test, sets[fs]], num_iteration=m.best_iteration)
        z = logit(p)
        res = []
        for sh in (-1.0, -0.5, 0.0, 0.5):
            dd = te[["s1_id", "pool_id"]].assign(prob=1 / (1 + np.exp(-(z + sh))))
            pred = decide(dd, ids, {"method": "expf", "alpha": 1.5, "one_to_one": True})
            res.append(f"{sh:+.1f}: {np.mean([f05(pred[s], gold.get(s, ())) for s in ids]):.4f}")
        print(f"  {name:22s} ({len(sets[fs])} feats, {m.best_iteration} trees) US AUC {roc_auc_score(te['y'], p):.4f} | "
              f"F0.5 by shift " + " | ".join(res), flush=True)


if __name__ == "__main__":
    main()
