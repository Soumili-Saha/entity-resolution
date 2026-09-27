"""Submission from the stacker (entity_matcher.stack_eval): train one stacker on all fold-0 pairs of the 500k-S1
sample, apply it to the test pairs (v2 LightGBM + CE test scores, competition computed over all test
S1s), decide, write, validate.

Usage: python -m entity_matcher.stack_submit --out output/v5 --shift france:-1 [--features pool|meta|extra]
       [--extra ce_large=precomputed/ce_large_out]
"""
import argparse
import glob
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from entity_matcher.blend_eval import logit
from entity_matcher.decide import decide
from entity_matcher.evaluate import fold_of
from entity_matcher.io_utils import load_ground_truth
from entity_matcher.run import art_dir, write_submission
from entity_matcher.stack_eval import STACK_PARAMS, apply_fill, build_X, load_ce

META = ["b_source", "len_name_a", "len_addr_a", "len_name_b", "len_addr_b"]


def read_pairs(pattern):
    """LightGBM pair probabilities (prob or lgbm_prob column) with prob >= 0.001, as lgbm_prob."""
    files = sorted(glob.glob(pattern))
    col = "lgbm_prob" if "lgbm_prob" in pq.read_schema(files[0]).names else "prob"
    df = pd.concat([pd.read_parquet(f, filters=[(col, ">=", 0.001)]) for f in files], ignore_index=True)
    return df.rename(columns={"prob": "lgbm_prob"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shift", default="")
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--features", default="pool", choices=["pool", "meta", "extra"])
    ap.add_argument("--extra", action="append", default=[], help="name=folder of an extra cross-encoder")
    ap.add_argument("--swap", default="", help="col=folder: replace <col>_prob on unseen-country test pairs")
    ap.add_argument("--unseen_prob", default="", help="glob of parquet (s1_id, pool_id, prob): final probability "
                    "for unseen-country S1 pairs (e.g. a France-specialised model); pairs not in it keep the stacker prob")
    ap.add_argument("--pairs", default="precomputed/ce/train_pairs_part*.parquet", help="LightGBM OOF pair files (train)")
    ap.add_argument("--s1_from", default="", help="glob of pair files: train only on their fold-0 S1s")
    ap.add_argument("--lgbm_test", default="artefacts/test/probs_model_v2.parquet", help="LightGBM test prob files")
    ap.add_argument("--ce_fill", default="zero", choices=["zero", "lgbm"], help="train pairs without a CE score")
    ap.add_argument("--fill", action="append", default=[], help="name=src[:lo:hi], as in entity_matcher.stack_eval")
    ap.add_argument("--raw", action="store_true", help="add entity_matcher.raw_feats pair evidence to the stacker")
    ap.add_argument("--unseen_mix", default="", help="col:w1,w2,...: on unseen-country pairs, logit-mix the stacker "
                    "prob with <col>_prob at each weight w (one submission per weight, written to <out>_mix<w>)")
    args = ap.parse_args()
    if os.name == "nt":  # keep full CPU when the laptop is locked (Windows EcoQoS)
        from entity_matcher.no_throttle import disable_throttling
        disable_throttling()
    extra = dict(e.split("=") for e in args.extra) if args.features == "extra" else {}
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    def attach(df, kind):
        df = df.merge(load_ce("precomputed/ce_out", kind), on=["s1_id", "pool_id"], how="left")
        for name, d in extra.items():
            df = df.merge(load_ce(d, kind).rename(columns={"ce_prob": f"{name}_prob"}), on=["s1_id", "pool_id"],
                          how="left")
        return df.reset_index(drop=True)

    def select(X):
        return X if args.features != "pool" else X.drop(columns=META)

    # train on all fold-0 pairs (the only S1s with out-of-fold CE scores)
    pairs = read_pairs(args.pairs)
    if "fold" not in pairs:
        pairs["fold"] = pairs["s1_id"].map(lambda s: fold_of(s, 5))
    f0 = pairs[pairs["fold"] == 0]
    if args.s1_from:
        keep = set(pd.concat([pd.read_parquet(f, columns=["s1_id"]) for f in sorted(glob.glob(args.s1_from))])["s1_id"])
        f0 = f0[f0["s1_id"].isin(keep)]
    if "label" not in f0:
        gold = load_ground_truth(cfg["paths"]["data_dir"])
        f0 = f0.assign(label=[int(p in gold.get(s, ())) for s, p in zip(f0["s1_id"], f0["pool_id"])])
    tr = attach(f0, "oof_fold0")
    for c in ["ce_prob"] + [f"{e}_prob" for e in extra]:
        print(f"train {c}: missing on {int(tr[c].isna().sum()):,} of {len(tr):,} pairs", flush=True)
        if not any(f.startswith(f"{c[:-5]}=") for f in args.fill):
            tr[c] = tr[c].fillna(0.0 if args.ce_fill == "zero" else tr["lgbm_prob"])
    tr = apply_fill(tr, args.fill)
    Xtr = select(build_X(tr, pairs[["s1_id", "pool_id", "lgbm_prob"]], "train", tuple(extra)))
    if args.raw:
        from entity_matcher.raw_feats import raw_features
        Xtr = Xtr.join(raw_features(tr, "train"))
    y = tr["label"].to_numpy()
    del pairs
    va = tr["s1_id"].map(lambda s: fold_of(s, 20) == 15).to_numpy()  # early-stopping slice
    m = lgb.train(STACK_PARAMS, lgb.Dataset(Xtr[~va], y[~va]), 3000, valid_sets=[lgb.Dataset(Xtr[va], y[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    m = lgb.train(STACK_PARAMS, lgb.Dataset(Xtr, y), m.best_iteration)
    print(f"stacker ({args.features}): {len(tr):,} training pairs, {Xtr.shape[1]} features, {m.num_trees()} trees",
          flush=True)

    # test
    lg = read_pairs(args.lgbm_test)
    te = attach(lg, "test")
    for c in ["ce_prob"] + [f"{e}_prob" for e in extra]:
        miss = int(te[c].isna().sum())
        print(f"test {c}: missing on {miss:,} of {len(te):,} pairs", flush=True)
        if not any(f.startswith(f"{c[:-5]}=") for f in args.fill):
            te[c] = te[c].fillna(te["lgbm_prob"])
    te = apply_fill(te, args.fill)
    if args.swap:  # e.g. ce_large=precomputed/ce_france_out: that CE's scores on unseen-country S1s replace the column
        col, d = args.swap.split("=")
        files = sorted(glob.glob(os.path.join(d, "ce_test_unseen_part*.parquet")))
        files += sorted(glob.glob(os.path.join(d, "ce_extra_test_part*.parquet")))  # new step-3 pairs, all countries
        sw = pd.concat([pd.read_parquet(p) for p in files]).drop_duplicates(["s1_id", "pool_id"])
        # only S1s of countries absent from training (open set, no hard-coded names)
        tr_c = set(pd.read_parquet(os.path.join(art_dir(cfg, "train"), "s1.parquet"), columns=["country_norm"])["country_norm"])
        te_c = pd.read_parquet(os.path.join(art_dir(cfg, "test"), "s1.parquet"), columns=["entity_id", "country_norm"])
        unseen = set(te_c.loc[~te_c["country_norm"].isin(tr_c), "entity_id"])
        sw = sw[sw["s1_id"].isin(unseen)].set_index(["s1_id", "pool_id"])["ce_prob"]
        new = pd.Series(pd.MultiIndex.from_frame(te[["s1_id", "pool_id"]]).map(sw), index=te.index)
        print(f"swap {col}: {new.notna().sum():,} test pairs replaced from {d}", flush=True)
        te[f"{col}_prob"] = new.fillna(te[f"{col}_prob"])
    Xte = select(build_X(te, lg, "test", tuple(extra)))
    if args.raw:
        Xte = Xte.join(raw_features(te, "test"))
    del lg
    te["prob"] = m.predict(Xte[Xtr.columns])
    s1 = pd.read_parquet(os.path.join(art_dir(cfg, "test"), "s1.parquet"), columns=["entity_id", "country_norm"])
    if args.unseen_prob:
        tr_c = set(pd.read_parquet(os.path.join(art_dir(cfg, "train"), "s1.parquet"), columns=["country_norm"])["country_norm"])
        unseen = set(s1.loc[~s1["country_norm"].isin(tr_c), "entity_id"])
        up = pd.concat([pd.read_parquet(f, columns=["s1_id", "pool_id", "prob"]) for f in sorted(glob.glob(args.unseen_prob))])
        up = up[up["s1_id"].isin(unseen)].drop_duplicates(["s1_id", "pool_id"])
        new = te[["s1_id", "pool_id"]].merge(up, on=["s1_id", "pool_id"], how="left")["prob"].to_numpy()
        # pairs only in the override file (not in the stacker table) are appended as new candidates
        extra_rows = up.merge(te[["s1_id", "pool_id"]], on=["s1_id", "pool_id"], how="left", indicator=True)
        extra_rows = extra_rows[extra_rows["_merge"] == "left_only"][["s1_id", "pool_id", "prob"]]
        te["prob"] = np.where(np.isnan(new), te["prob"].to_numpy(), new)
        te = pd.concat([te, extra_rows], ignore_index=True)
        print(f"unseen_prob: {np.isfinite(new).sum():,} pairs overridden, {len(extra_rows):,} added", flush=True)
    variants = [(args.out, None)]
    shifts = {args.out: args.shift}
    if args.shift and "," in args.shift:  # country:d1,d2,...: one submission per shift, written to <out>_s<d>
        c, ds = args.shift.split(":")
        variants = [(f"{args.out}_s{d}", None) for d in ds.split(",")]
        shifts = {f"{args.out}_s{d}": f"{c}:{d}" for d in ds.split(",")}
    if args.unseen_mix:  # the stacker learned CE weights on seen countries; re-weight one CE on unseen ones
        mcol, ws = args.unseen_mix.split(":")
        variants = [(f"{args.out}_mix{w}", float(w)) for w in ws.split(",")]
        tr_c = set(pd.read_parquet(os.path.join(art_dir(cfg, "train"), "s1.parquet"), columns=["country_norm"])["country_norm"])
        in_u = te["s1_id"].isin(set(s1.loc[~s1["country_norm"].isin(tr_c), "entity_id"])).to_numpy()
        z_stack, z_col = logit(te["prob"].to_numpy()), logit(te[f"{mcol}_prob"].to_numpy())
    params_d = {"method": "expf", "alpha": args.alpha, "one_to_one": True}
    s1_ids = s1["entity_id"].tolist()
    base_prob = te["prob"].to_numpy().copy()
    for out, w in variants:
        te["prob"] = base_prob
        if w is not None:
            te["prob"] = np.where(in_u, 1 / (1 + np.exp(-((1 - w) * z_stack + w * z_col))), 1 / (1 + np.exp(-z_stack)))
            print(f"[{out}] unseen_mix {mcol} w={w} on {in_u.sum():,} pairs", flush=True)
        shift = shifts.get(out, args.shift)
        if shift:
            c, delta = shift.split(":")
            in_c = te["s1_id"].isin(set(s1.loc[s1["country_norm"] == c, "entity_id"])).to_numpy()
            z = logit(te["prob"].to_numpy())
            te["prob"] = np.where(in_c, 1 / (1 + np.exp(-(z + float(delta)))), te["prob"].to_numpy())
            print(f"shifted logit by {delta} on {in_c.sum():,} pairs of country {c}", flush=True)
        write_submission(cfg, s1_ids, decide(te[["s1_id", "pool_id", "prob"]], s1_ids, params_d), out,
                         extra_cands=te[["s1_id", "pool_id"]])
        with open(os.path.join(out, "blend.json"), "w") as f:
            json.dump({"model": f"stack_{args.features}", "extra": extra, "swap": args.swap, "decision": params_d,
                       "shift": shift, "unseen_mix": args.unseen_mix, "mix_w": w, "pairs": args.pairs,
                       "lgbm_test": args.lgbm_test}, f, indent=1)


if __name__ == "__main__":
    main()
