"""Unseen-country proxy for France: every scorer is trained WITHOUT the unseen country (default: India-only
models, US unseen), the stacker is trained on the seen country's fold-0 S1s and applied to the unseen country's
fold-0 S1s, and the unseen country is scored under a grid of logit shifts. A France technique (synthetic pairs,
self-training, rescoring all pairs, ...) counts as measured when it raises the unseen-country score here.

Inputs are fold-0 pair files of the full-data run scored by seen-country-only models:
  --lgbm  glob of (s1_id, pool_id, prob) from the seen-only LightGBM (both countries' fold-0 S1s)
  --ce    folder with ce_oof_fold0_part*.parquet (s1_id, pool_id, ce_prob) from the seen-only CE
  --extra name=folder for further seen-only CEs (e.g. a CE also trained on synthetic unseen-country pairs)

Usage: python -m entity_matcher.proxy_eval --lgbm "precomputed/loco_india_out/oof_part*.parquet" --ce precomputed/ce_loco_india_out
       [--extra synth=precomputed/ce_synth_out] [--fill synth=ce:0.02:0.98] [--tag base]
"""
import argparse
import glob
import os
import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from entity_matcher.blend_eval import logit
from entity_matcher.decide import decide
from entity_matcher.evaluate import f05
from entity_matcher.io_utils import load_ground_truth
from entity_matcher.stack_eval import STACK_PARAMS, apply_fill, build_X, load_ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lgbm", required=True)
    ap.add_argument("--ce", required=True)
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--fill", action="append", default=[])
    ap.add_argument("--seen", default="india")
    ap.add_argument("--unseen", default="us")
    ap.add_argument("--ids_from", default="precomputed/full_out/oof_train_full_part*.parquet",
                    help="pair files whose fold-0 S1s (all, including those without a pair >= 0.001) are scored")
    ap.add_argument("--shifts", default="-2,-1.5,-1,-0.5,0,0.5")
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--drop", default="", help="comma-separated feature-name prefixes left out of the stacker")
    ap.add_argument("--raw", action="store_true", help="add entity_matcher.raw_feats pair evidence to the stacker")
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()
    if os.name == "nt":
        from entity_matcher.no_throttle import disable_throttling
        disable_throttling()
    extra = dict(e.split("=") for e in args.extra)
    gold = load_ground_truth("dataset")
    s1 = pd.read_parquet("artefacts/train/s1.parquet", columns=["entity_id", "country_norm"])
    country = dict(zip(s1["entity_id"], s1["country_norm"]))

    files = sorted(glob.glob(args.lgbm))
    col = "lgbm_prob" if "lgbm_prob" in pd.read_parquet(files[0]).columns else "prob"
    lg = pd.concat([pd.read_parquet(f, columns=["s1_id", "pool_id", col], filters=[(col, ">=", 0.001)]) for f in files],
                   ignore_index=True).rename(columns={col: "lgbm_prob"}).drop_duplicates(["s1_id", "pool_id"])
    lg = lg[lg["s1_id"].map(lambda s: zlib.crc32(s.encode()) % 5 == 0)]
    lg["c"] = lg["s1_id"].map(country)
    lg = lg[lg["c"].isin([args.seen, args.unseen])].reset_index(drop=True)
    df = lg.merge(load_ce(args.ce, "oof_fold0"), on=["s1_id", "pool_id"], how="left")
    print(f"pairs {len(df):,} | ce missing {df['ce_prob'].isna().sum():,}", flush=True)
    df["ce_prob"] = df["ce_prob"].fillna(df["lgbm_prob"])
    for name, d in extra.items():
        df = df.merge(load_ce(d, "oof_fold0").rename(columns={"ce_prob": f"{name}_prob"}), on=["s1_id", "pool_id"],
                      how="left")
        print(f"{name}: missing {df[f'{name}_prob'].isna().sum():,}", flush=True)
        if not any(f.startswith(f"{name}=") for f in args.fill):
            df[f"{name}_prob"] = df[f"{name}_prob"].fillna(df["ce_prob"])
    df = apply_fill(df, args.fill).reset_index(drop=True)
    df["label"] = [int(p in gold.get(s, ())) for s, p in zip(df["s1_id"], df["pool_id"])]
    # pool competition over the fold-0 pairs of both countries (the same table for every variant)
    X_all = build_X(df, df[["s1_id", "pool_id", "lgbm_prob"]], "train", tuple(extra))
    if args.raw:
        from entity_matcher.raw_feats import raw_features
        X_all = X_all.join(raw_features(df, "train"))
    seen = (df["c"] == args.seen).to_numpy()
    y = df["label"].to_numpy()
    va = seen & df["s1_id"].map(lambda s: zlib.crc32(s.encode()) % 20 == 15).to_numpy()
    un = df.loc[~seen, ["s1_id", "pool_id"]].copy()
    un["blend"] = X_all["blend"].to_numpy()[~seen]
    cols = ["blend"]
    # --drop "a,b;c": one stacker per ';'-separated set of dropped feature prefixes ("" = all features)
    for k, drop in enumerate(args.drop.split(";")):
        pre = [p for p in drop.split(",") if p]
        X = X_all[[c for c in X_all.columns if not any(c.startswith(p) for p in pre)]]
        m = lgb.train(STACK_PARAMS, lgb.Dataset(X[seen & ~va], y[seen & ~va]), 3000,
                      valid_sets=[lgb.Dataset(X[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        name = f"stack{k}"
        un[name] = m.predict(X[~seen], num_iteration=m.best_iteration)
        cols.insert(k, name)
        print(f"{name}: drop {pre or 'nothing'} -> {X.shape[1]} features, {m.best_iteration} trees", flush=True)

    allids = pd.concat([pd.read_parquet(f, columns=["s1_id"]) for f in sorted(glob.glob(args.ids_from))])["s1_id"].unique()
    ids = [s for s in allids if zlib.crc32(s.encode()) % 5 == 0 and country.get(s) == args.unseen]
    print(f"stackers trained on {int(seen.sum()):,} {args.seen} pairs | {args.unseen}: "
          f"{len(un):,} pairs, {len(ids):,} fold-0 S1", flush=True)
    params = {"method": "expf", "alpha": args.alpha, "one_to_one": True}
    for colname in cols:
        z = logit(un[colname].to_numpy())
        res = []
        for sh in map(float, args.shifts.split(",")):
            d = un[["s1_id", "pool_id"]].assign(prob=1 / (1 + np.exp(-(z + sh))))
            pred = decide(d, ids, params)
            res.append(f"{sh:+.1f}: {np.mean([f05(pred[s], gold.get(s, ())) for s in ids]):.4f}")
        print(f"  [{args.tag}] {args.unseen} unseen, {colname:5s} F0.5 by shift | " + " | ".join(res), flush=True)
    os.makedirs("artefacts/exp", exist_ok=True)
    un.to_parquet(f"artefacts/exp/proxy_{args.tag}.parquet")


if __name__ == "__main__":
    main()
