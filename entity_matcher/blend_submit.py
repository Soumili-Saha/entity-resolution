"""Submission from a CE + LightGBM blend: blend the cross-encoder test scores (precomputed/ce_out/ce_test_*)
with a LightGBM test-probability file, apply decision parameters chosen by entity_matcher.blend_eval on fold-0
entities, and write + validate matching_results.tsv / candidate_pairs.tsv.

Pairs with lgbm_prob < 0.001 were not CE-scored; they keep blend = 0 (never selected).

Usage: python -m entity_matcher.blend_submit --blend logit --w 0.7 --params '{"method": "expf", "alpha": 2.0, "one_to_one": true}'
       --out output/v3_ce [--lgbm artefacts/test/probs_model_v2.parquet]
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import yaml

from entity_matcher.blend_eval import logit
from entity_matcher.decide import decide
from entity_matcher.run import art_dir, write_submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--lgbm", default="artefacts/test/probs_model_v2.parquet")
    ap.add_argument("--ce", default="precomputed/ce_out")
    ap.add_argument("--blend", choices=["prob", "logit", "ce"], required=True)
    ap.add_argument("--w", type=float, default=1.0, help="CE weight")
    ap.add_argument("--params", required=True, help="decision params JSON")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shift", default="", help="country:delta logit shift (unseen-country probe)")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    params = json.loads(args.params)
    lg = pd.read_parquet(args.lgbm)
    lg = lg[lg["prob"] >= 0.001].rename(columns={"prob": "lgbm_prob"})
    ce = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(args.ce, "ce_test_part*.parquet")))],
                   ignore_index=True)
    df = lg.merge(ce, on=["s1_id", "pool_id"], how="left")
    miss = int(df["ce_prob"].isna().sum())
    print(f"{len(df):,} test pairs, {len(ce):,} CE scores, {miss:,} without CE score", flush=True)
    if miss > 0.001 * len(df):
        raise SystemExit("CE test scores do not cover the LightGBM pairs - wrong probability file?")
    df["ce_prob"] = df["ce_prob"].fillna(df["lgbm_prob"])
    a, b = df["ce_prob"].to_numpy(), df["lgbm_prob"].to_numpy()
    if args.blend == "ce":
        df["prob"] = a
    elif args.blend == "prob":
        df["prob"] = args.w * a + (1 - args.w) * b
    else:
        df["prob"] = 1 / (1 + np.exp(-(args.w * logit(a) + (1 - args.w) * logit(b))))
    s1 = pd.read_parquet(os.path.join(art_dir(cfg, "test"), "s1.parquet"), columns=["entity_id", "country_norm"])
    s1_ids = s1["entity_id"].tolist()
    if args.shift:  # probe: shift the blended logit of one country's S1s (e.g. france:-1)
        c, delta = args.shift.split(":")
        in_c = df["s1_id"].isin(set(s1.loc[s1["country_norm"] == c, "entity_id"])).to_numpy()
        z = logit(df["prob"].to_numpy())
        df["prob"] = np.where(in_c, 1 / (1 + np.exp(-(z + float(delta)))), df["prob"].to_numpy())
        print(f"shifted logit by {delta} on {in_c.sum():,} pairs of country {c}", flush=True)
    matches = decide(df[["s1_id", "pool_id", "prob"]], s1_ids, params)
    write_submission(cfg, s1_ids, matches, args.out)
    with open(os.path.join(args.out, "blend.json"), "w") as f:
        json.dump({"lgbm": args.lgbm, "blend": args.blend, "w": args.w, "decision": params, "shift": args.shift},
                  f, indent=1)


if __name__ == "__main__":
    main()
