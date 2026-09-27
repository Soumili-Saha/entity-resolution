"""Validation analysis for the write-up: error breakdown of the OOF predictions, F0.5 by country
and by number of true matches, example false merges / misses, and leave-one-country-out
(train on US -> score India, and the reverse) as a proxy for the unseen test country (France).

Usage: python -m entity_matcher.analysis [--loco]
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import yaml

from entity_matcher.decide import decide
from entity_matcher.evaluate import f05
from entity_matcher.io_utils import load_ground_truth
from entity_matcher.run import art_dir, load_features, load_normalized, model_dir


def score_by(preds, gold, ids, key):
    """Mean F0.5 per group given {s1_id: group}."""
    rows = [(key[i], f05(preds.get(i, ()), gold.get(i, ()))) for i in ids]
    return pd.DataFrame(rows, columns=["group", "f05"]).groupby("group")["f05"].agg(["mean", "size"])


def main():
    """Print the analysis report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--loco", action="store_true")
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    d = art_dir(cfg, "train")
    md = model_dir(cfg)
    with open(os.path.join(md, "config.json")) as f:
        mc = json.load(f)
    gold = load_ground_truth(cfg["paths"]["data_dir"])
    s1, pool = load_normalized(cfg, "train", columns=["entity_id", "business_name", "business_address", "country"])
    q = pd.read_parquet(os.path.join(d, "query_idx.parquet"))["s1_idx"].to_numpy()
    ids = s1["entity_id"].to_numpy()[q].tolist()
    country = dict(zip(s1["entity_id"], s1["country"]))
    oof = pd.read_parquet(os.path.join(d, "oof.parquet"))
    preds = decide(oof, ids, mc["decision"])

    total = float(np.mean([f05(preds.get(i, ()), gold.get(i, ())) for i in ids]))
    print(f"OOF macro F0.5 = {total:.4f} on {len(ids):,} S1  (decision {mc['decision']})")
    print("\nby country:\n", score_by(preds, gold, ids, country).round(4))
    n_gold = {i: min(len(gold.get(i, ())), 7) for i in ids}
    print("\nby number of true matches (7 = 7+):\n", score_by(preds, gold, ids, n_gold).round(4))

    # error accounting
    cand = oof.groupby("s1_id")["pool_id"].apply(set).to_dict()
    tp = fp = fn_blocked = fn_rejected = 0
    fp_single = 0
    fp_ex, fn_ex = [], []
    for i in ids:
        g, p, c = gold.get(i, set()), set(preds.get(i, ())), cand.get(i, set())
        tp += len(p & g)
        for x in p - g:
            fp += 1
            fp_single += int(not g)
            fp_ex.append((i, x))
        for x in g - p:
            if x in c:
                fn_rejected += 1
                fn_ex.append((i, x, "rejected"))
            else:
                fn_blocked += 1
                fn_ex.append((i, x, "not in candidates"))
    tot_gold = sum(len(gold.get(i, ())) for i in ids)
    print(f"\npairs: gold {tot_gold:,} | TP {tp:,} | FP {fp:,} ({fp_single:,} on true singletons) | "
          f"FN {fn_blocked + fn_rejected:,} (blocking {fn_blocked:,}, model/decision {fn_rejected:,})")
    print(f"pair precision {tp / max(tp + fp, 1):.4f} | pair recall {tp / max(tot_gold, 1):.4f}")

    name = dict(zip(s1["entity_id"], s1["business_name"] + " | " + s1["business_address"]))
    name.update(zip(pool["entity_id"], pool["business_name"] + " | " + pool["business_address"]))
    rng = random.Random(0)
    print("\nexample FALSE MERGES (S1  ->  wrongly matched record):")
    for i, x in rng.sample(fp_ex, min(args.examples, len(fp_ex))):
        truth = sorted(gold.get(i, ()))[:1]
        print(f"  {name[i]}\n    -> {name[x]}\n    true match e.g.: {name[truth[0]] if truth else '(singleton)'}")
    print("\nexample MISSES (S1  ->  missed true record):")
    for i, x, why in rng.sample(fn_ex, min(args.examples, len(fn_ex))):
        print(f"  [{why}] {name[i]}\n    -> {name[x]}")

    if args.loco:
        import lightgbm as lgb
        from entity_matcher.ranker import lgb_params
        feats = load_features(cfg, "train")
        cols = mc["features"]
        y = np.array([int(p in gold.get(s, ())) for s, p in zip(feats["s1_id"], feats["pool_id"])], dtype=np.int8)
        pc = feats["s1_id"].map(country).to_numpy()
        print("\nleave-one-country-out (stage 1 only, 600 rounds):")
        for held in sorted(set(pc)):
            tr, te = pc != held, pc == held
            m = lgb.train(lgb_params(cfg), lgb.Dataset(feats.loc[tr, cols].to_numpy(np.float32), y[tr]), 600)
            df = feats.loc[te, ["s1_id", "pool_id"]].copy()
            df["prob"] = m.predict(feats.loc[te, cols].to_numpy(np.float32))
            held_ids = [i for i in ids if country[i] == held]
            p = decide(df, held_ids, {"method": "expf", "alpha": 3.0, "one_to_one": True})
            s = float(np.mean([f05(p.get(i, ()), gold.get(i, ())) for i in held_ids]))
            print(f"  train on others -> score {held}: F0.5 {s:.4f} ({len(held_ids):,} S1)")


if __name__ == "__main__":
    main()
