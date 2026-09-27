"""Stacker over LightGBM + cross-encoder scores with per-S1 and per-candidate competition context,
evaluated on the fold-0 S1s of the 500k-S1 cross-encoder sample with the same cross-fitted protocol as entity_matcher.blend_eval
(so scores are directly comparable).

Stacker OOF: fold-0 S1s are split in 4 quarters (crc32 % 20 in {0, 5, 10, 15}); each quarter is
predicted by a small LightGBM trained on the other three. The decision is then tuned / scored on the
two halves exactly as in blend_eval.

Feature sets compared (each a superset of the previous):
  pool  : scores + S1 context + empty-address flag + candidate competition
  meta  : + record source (S2/S3) and name / address lengths of both records
  extra : + every extra cross-encoder given with --extra (e.g. ce_large=precomputed/ce_large_out)

Usage: python -m entity_matcher.stack_eval [--extra ce_large=precomputed/ce_large_out]
"""
import argparse
import glob
import os
import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from entity_matcher.blend_eval import held_out, logit
from entity_matcher.io_utils import load_ground_truth

STACK_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 100,
                "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": 42}


def context(df, col, prefix):
    """Per-S1 rank, gap to best, second best, sum and count > 0.5 of score column col."""
    g = df.groupby("s1_id", sort=False)[col]
    out = pd.DataFrame(index=df.index)
    rank = g.rank(ascending=False, method="first")
    out[f"{prefix}_rank"] = rank
    mx = g.transform("max")
    out[f"{prefix}_gap"] = mx - df[col]
    out[f"{prefix}_max"] = mx
    out[f"{prefix}_sum"] = g.transform("sum")
    out[f"{prefix}_n50"] = (df[col] > 0.5).groupby(df["s1_id"], sort=False).transform("sum")
    second = df.loc[rank == 2].set_index("s1_id")[col]
    out[f"{prefix}_second"] = df["s1_id"].map(second).fillna(0.0)
    return out


def pool_context(all_pairs, df):
    """Candidate-side competition from LightGBM probabilities over ALL S1s of the table (every fold):
    how many S1s compete for the pool record, whether this S1 is the top one, and the best other S1's prob."""
    a = all_pairs[["s1_id", "pool_id", "lgbm_prob"]]
    a = a[a["lgbm_prob"] >= 0.001]
    a = a.sort_values(["pool_id", "lgbm_prob"], ascending=[True, False])
    r = a.groupby("pool_id", sort=False).cumcount().to_numpy()
    first = a.loc[r == 0].set_index("pool_id")["lgbm_prob"]
    second = a.loc[r == 1].set_index("pool_id")["lgbm_prob"]
    n = a.groupby("pool_id", sort=False).size()
    out = pd.DataFrame(index=df.index)
    out["pool_n"] = df["pool_id"].map(n).fillna(1).to_numpy()
    f1, f2 = df["pool_id"].map(first).to_numpy(), df["pool_id"].map(second).fillna(0.0).to_numpy()
    p = df["lgbm_prob"].to_numpy()
    out["pool_best_other"] = np.where(p >= f1, f2, f1)
    out["pool_margin"] = p - out["pool_best_other"]
    out["pool_is_top"] = (p >= f1).astype(np.float32)
    return out


def score_features(df, extra=()):
    """Scores, their logits, blends and per-S1 context for LightGBM, CE base and any extra CE columns."""
    ces = ["ce_prob"] + [f"{e}_prob" for e in extra]
    X = pd.DataFrame({"lgbm": df["lgbm_prob"], "lgbm_logit": logit(df["lgbm_prob"].to_numpy())}, index=df.index)
    for c in ces:
        X[c] = df[c]
        X[f"{c}_logit"] = logit(df[c].to_numpy())
    X["blend"] = 1 / (1 + np.exp(-(0.5 * X["lgbm_logit"] + 0.5 * X["ce_prob_logit"])))
    if extra:
        X["blend_all"] = 1 / (1 + np.exp(-np.mean([X["lgbm_logit"]] + [X[f"{c}_logit"] for c in ces], axis=0)))
    X["n_cands"] = df.groupby("s1_id", sort=False)["pool_id"].transform("size")
    tmp = df[["s1_id"]].copy()
    for col in ["lgbm", *ces, "blend"] + (["blend_all"] if extra else []):
        tmp[col] = X[col].to_numpy()
        X = X.join(context(tmp, col, f"ctx_{col}"))
    return X


def meta_features(df, split):
    """Record source and name / address lengths (normalised views) of both records."""
    cols = ["entity_id", "name_clean", "addr_clean"]
    s1 = pd.read_parquet(f"artefacts/{split}/s1.parquet", columns=cols).set_index("entity_id")
    pool = pd.concat([pd.read_parquet(f"artefacts/{split}/s{k}.parquet", columns=cols) for k in (2, 3)]).set_index("entity_id")
    out = pd.DataFrame(index=df.index)
    out["b_source"] = (df["pool_id"].str[:2] == "S3").astype(np.float32).to_numpy()
    for side, tab, key in (("a", s1, "s1_id"), ("b", pool, "pool_id")):
        out[f"len_name_{side}"] = df[key].map(tab["name_clean"].str.len()).fillna(0).to_numpy()
        out[f"len_addr_{side}"] = df[key].map(tab["addr_clean"].str.len()).fillna(0).to_numpy()
    return out


def build_X(df, all_pairs, split, extra=(), meta=True):
    """Full stacker matrix for pairs df (with lgbm_prob, ce_prob and <extra>_prob columns)."""
    X = score_features(df, extra)
    pool = pd.concat([pd.read_parquet(f"artefacts/{split}/s{k}.parquet", columns=["entity_id", "addr_clean"])
                      for k in (2, 3)])
    empty = set(pool.loc[pool["addr_clean"].str.len() == 0, "entity_id"])
    X["b_empty_addr"] = df["pool_id"].isin(empty).to_numpy().astype(np.float32)
    X = X.join(pool_context(all_pairs, df))
    if meta:
        X = X.join(meta_features(df, split))
    return X.astype(np.float32)


def load_ce(d, kind):
    """CE scores of kind 'oof_fold0' or 'test' from folder d, including extra-pair files when present."""
    files = sorted(glob.glob(os.path.join(d, f"ce_{kind}_part*.parquet")))
    files += sorted(glob.glob(os.path.join(d, f"ce_{kind}_rest_part*.parquet")))  # remainder scored later
    files += sorted(glob.glob(os.path.join(d, f"ce_extra_{'fold0' if kind == 'oof_fold0' else 'test'}_part*.parquet")))
    if kind == "test":  # unseen-country pairs scored beyond the contested set (e.g. Qwen on ALL France pairs)
        files += sorted(glob.glob(os.path.join(d, "ce_test_contested_part*.parquet")))
        files += sorted(glob.glob(os.path.join(d, "ce_test_france_part*.parquet")))
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True).drop_duplicates(["s1_id", "pool_id"])


def apply_fill(df, specs):
    """--fill name=src[:lo:hi]: <name>_prob takes <src>_prob where missing and, with lo:hi, also where the
    0.5/0.5 logit blend of lgbm_prob and <src>_prob is outside [lo, hi] (a CE scored only on contested pairs,
    e.g. xlm-roberta-xl on test, gets the same coverage on train)."""
    for spec in specs:
        name, rest = spec.split("=")
        src, *rng = rest.split(":")
        col, fill = f"{name}_prob", df[f"{src}_prob"]
        keep = df[col].notna()
        if rng:
            b = 1 / (1 + np.exp(-(0.5 * logit(df["lgbm_prob"].to_numpy()) + 0.5 * logit(fill.to_numpy()))))
            keep &= (b >= float(rng[0])) & (b <= float(rng[1]))
        print(f"fill {col}: own score on {int(keep.sum()):,} of {len(df):,} pairs, rest from {src}", flush=True)
        df[col] = df[col].where(keep, fill)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", action="append", default=[], help="name=folder of an extra cross-encoder")
    ap.add_argument("--tag", default="sample")
    ap.add_argument("--param_grid", action="store_true",
                    help="stacker with all features under several LightGBM settings (leaves / learning rate)")
    ap.add_argument("--compare_extras", action="store_true",
                    help="only two stackers: with the first --extra only vs with all --extra (skip blends/pool/meta)")
    ap.add_argument("--only_all", action="store_true", help="with --compare_extras: only the all-extras stacker")
    ap.add_argument("--s1_from", default="", help="glob of pair files: evaluate only their fold-0 S1s "
                    "(e.g. the 500k-S1 cross-encoder sample, whose pairs all have CE scores)")
    ap.add_argument("--ce_fill", default="zero", choices=["zero", "lgbm"],
                    help="value for pairs without a CE score (new candidates of a re-blocked run)")
    ap.add_argument("--raw", action="store_true", help="add entity_matcher.raw_feats pair evidence to the stacker")
    ap.add_argument("--fill", action="append", default=[],
                    help="name=src[:lo:hi]: fill extra <name> from extra <src> (see apply_fill); src listed first")
    ap.add_argument("--pairs", default="precomputed/ce/train_pairs_part*.parquet",
                    help="glob of LightGBM OOF pair files; precomputed/full_out/oof_train_full_part*.parquet = all 2.2M "
                         "training S1, so candidate competition is complete (as on test)")
    args = ap.parse_args()
    if os.name == "nt":  # keep full CPU when the laptop is locked (Windows EcoQoS)
        from entity_matcher.no_throttle import disable_throttling
        disable_throttling()
    extra = dict(e.split("=") for e in args.extra)
    gold = load_ground_truth("dataset")
    files = sorted(glob.glob(args.pairs))
    pcol = "lgbm_prob" if "lgbm_prob" in pq.read_schema(files[0]).names else "prob"
    # only pairs with prob >= 0.001 are ever used (stacker rows and candidate competition); filtering at read
    # time keeps the 54.6M-row full-data table within laptop memory
    all_pairs = pd.concat([pd.read_parquet(p, filters=[(pcol, ">=", 0.001)]) for p in files], ignore_index=True)
    all_pairs = all_pairs.rename(columns={"prob": "lgbm_prob"})
    if "fold" not in all_pairs:
        all_pairs["fold"] = all_pairs["s1_id"].map(lambda s: zlib.crc32(s.encode()) % 5)
    pairs = all_pairs[all_pairs["fold"] == 0].drop(columns=["fold"])
    if args.s1_from:
        keep = set(pd.concat([pd.read_parquet(f, columns=["s1_id"]) for f in sorted(glob.glob(args.s1_from))])["s1_id"])
        pairs = pairs[pairs["s1_id"].isin(keep)]
    if "label" not in pairs:
        pairs = pairs.assign(label=[int(p in gold.get(s, ())) for s, p in zip(pairs["s1_id"], pairs["pool_id"])])
    print(f"{len(all_pairs):,} pairs, {pairs['s1_id'].nunique():,} fold-0 S1", flush=True)
    if args.s1_from:  # every fold-0 S1 of the reference sample, including those without a pair >= 0.001
        ids = [x for x in keep if zlib.crc32(x.encode()) % 5 == 0]
    else:  # every fold-0 S1 of the pair table, including those without a pair >= 0.001 (read unfiltered)
        allids = pd.concat([pd.read_parquet(f, columns=["s1_id"]) for f in files])["s1_id"].unique()
        ids = [x for x in allids if zlib.crc32(x.encode()) % 5 == 0]
    df = pairs[pairs["lgbm_prob"] >= 0.001].merge(load_ce("precomputed/ce_out", "oof_fold0"), on=["s1_id", "pool_id"],
                                                  how="left").reset_index(drop=True)
    print(f"ce_prob: missing on {df['ce_prob'].isna().sum():,} of {len(df):,} pairs", flush=True)
    df["ce_prob"] = df["ce_prob"].fillna(0.0 if args.ce_fill == "zero" else df["lgbm_prob"])
    for name, d in extra.items():
        e = load_ce(d, "oof_fold0").rename(columns={"ce_prob": f"{name}_prob"})
        df = df.merge(e, on=["s1_id", "pool_id"], how="left")
        print(f"{name}: missing on {df[f'{name}_prob'].isna().sum():,} of {len(df):,} pairs", flush=True)
        if not any(f.startswith(f"{name}=") for f in args.fill):
            df[f"{name}_prob"] = df[f"{name}_prob"].fillna(0.0 if args.ce_fill == "zero" else df["lgbm_prob"])
    df = apply_fill(df, args.fill)
    X = build_X(df, all_pairs, "train", tuple(extra))
    if args.raw:  # raw address / name evidence so the stacker can arbitrate LightGBM vs CE by pair type
        from entity_matcher.raw_feats import raw_features
        X = X.join(raw_features(df, "train"))
    base_score_cols = {"lgbm", "lgbm_logit", "ce_prob", "ce_prob_logit", "blend", "n_cands"}
    extra_cols = [c for c in X.columns if any(c.startswith(p) for e in extra for p in (f"{e}_prob", f"ctx_{e}_prob"))
                  or c.startswith("blend_all") or c.startswith("ctx_blend_all")]
    meta_cols = ["b_source", "len_name_a", "len_addr_a", "len_name_b", "len_addr_b"]
    sets = {"pool": [c for c in X.columns if c not in extra_cols and c not in meta_cols],
            "meta": [c for c in X.columns if c not in extra_cols]}
    if extra:
        sets["extra"] = list(X.columns)
    if args.compare_extras:
        later = list(extra)[1:]
        drop = [c for c in X.columns if any(c.startswith(p) for e in later for p in (f"{e}_prob", f"ctx_{e}_prob"))
                or c.startswith("blend_all") or c.startswith("ctx_blend_all")]
        sets = {f"x_{list(extra)[0]}": [c for c in X.columns if c not in drop], "x_all": list(X.columns)}
        if args.only_all:  # the with-first-extra-only reference is already known
            sets = {"x_all": list(X.columns)}
    y = df["label"].to_numpy()
    q = df["s1_id"].map({s: zlib.crc32(s.encode()) % 20 for s in ids}).to_numpy()

    def oof_stack(M, params=STACK_PARAMS):
        oof = np.zeros(len(df))
        for k in (0, 5, 10, 15):
            te = q == k
            va_q = [x for x in (0, 5, 10, 15) if x != k][0]  # early stopping on one training quarter
            fit, va = ~te & (q != va_q), q == va_q
            m = lgb.train(params, lgb.Dataset(M[fit], y[fit]), 5000, valid_sets=[lgb.Dataset(M[va], y[va])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[te] = m.predict(M[te], num_iteration=m.best_iteration)
        return oof

    s1 = pd.read_parquet("artefacts/train/s1.parquet", columns=["entity_id", "country_norm"])
    country = dict(zip(s1["entity_id"], s1["country_norm"]))
    hh = {s: zlib.crc32(s.encode()) % 10 for s in ids}
    halves = [[s for s in ids if hh[s] == 0], [s for s in ids if hh[s] == 5]]
    df["blend"] = X["blend"].to_numpy()
    names = [] if (args.compare_extras or args.param_grid) else ["blend"]
    if extra and not (args.compare_extras or args.param_grid):
        df["blend_all"] = X["blend_all"].to_numpy()
        names.append("blend_all")
    if args.param_grid:
        M = X.to_numpy()
        for leaves, lr, mdl in ((31, 0.05, 100), (63, 0.05, 100), (127, 0.03, 200), (255, 0.03, 400)):
            name = f"stack_l{leaves}_lr{lr}"
            df[name] = oof_stack(M, dict(STACK_PARAMS, num_leaves=leaves, learning_rate=lr, min_data_in_leaf=mdl))
            names.append(name)
            print(f"  trained {name}", flush=True)
        sets = {}
    for name, cols in sets.items():
        df[f"stack_{name}"] = oof_stack(X[cols].to_numpy())
        names.append(f"stack_{name}")
    os.makedirs("artefacts/exp", exist_ok=True)
    df[["s1_id", "pool_id", "label", *names]].to_parquet(os.path.join("artefacts/exp", f"stack_oof_{args.tag}.parquet"))
    for name in names:
        score, by_c, p = held_out(df, name, gold, halves, country)
        print(f"  {name:12s} held-out macro F0.5 {score:.4f} | " +
              " | ".join(f"{c} {x:.4f}" for c, x in sorted(by_c.items())) + f" | params(all) {p}", flush=True)


if __name__ == "__main__":
    main()
