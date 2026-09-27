"""Fixed-protocol experiments on the cached training features (artefacts/train/feats).

Every run is scored on the SAME validation S1s (fold 0 = crc32(s1_id) % 5 == 0) so numbers are
comparable across runs. Modes:
  fold0 : train stage-1 LightGBM on folds 1-4 (all countries), tune the decision on fold 0,
          report macro F0.5 overall and per country, plus an error decomposition.
  loco  : leave-one-country-out, a proxy for an unseen country (France is test-only): train on
          folds 1-4 of country A, score fold 0 of country B with the decision tuned on A's fold 0
          ("transfer") and, for reference, tuned on B's fold 0 ("tuned on target").
Results are appended to logs/experiments.csv; fold-0 predictions go to artefacts/exp/preds_<tag>.parquet.

Usage: python -m entity_matcher.experiment --tag base [--mode fold0|loco] [--drop c1,c2] [--lr 0.05] [--leaves 255]
"""
import argparse
import csv
import datetime
import os
import subprocess

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from entity_matcher.decide import decide, tune
from entity_matcher.evaluate import f05, fold_of
from entity_matcher.io_utils import load_ground_truth
from entity_matcher.ranker import ID_COLS, lgb_params


def load(cfg):
    """Features, labels, S1 country and fold-0 mask for the cached training sample."""
    gold = load_ground_truth(cfg["paths"]["data_dir"])
    fd = os.path.join(cfg["paths"]["artefacts_dir"], "train", "feats")
    feats = pd.concat([pd.read_parquet(os.path.join(fd, p)) for p in sorted(os.listdir(fd))], ignore_index=True)
    y = np.array([int(p in gold.get(s, ())) for s, p in zip(feats["s1_id"], feats["pool_id"])], dtype=np.int8)
    s1 = pd.read_parquet(os.path.join(cfg["paths"]["artefacts_dir"], "train", "s1.parquet"),
                         columns=["entity_id", "country_norm"])
    country = dict(zip(s1["entity_id"], s1["country_norm"]))
    uniq = pd.unique(feats["s1_id"])
    f0 = {s for s in uniq if fold_of(s) == 0}
    val = feats["s1_id"].isin(f0).to_numpy()
    row_country = feats["s1_id"].map(country).to_numpy()
    return feats, y, gold, country, val, row_country


def fit(X, y, Xva, yva, params):
    """Stage-1 LightGBM with early stopping on the given validation rows."""
    dtr = lgb.Dataset(X, y, params=params)
    dva = lgb.Dataset(Xva, yva, reference=dtr, params=params)
    m = lgb.train(params, dtr, 5000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m, m.predict(Xva, num_iteration=m.best_iteration)


def scores(df, gold, s1_ids, params):
    """{s1_id: f05} under fixed decision params."""
    pred = decide(df, s1_ids, params)
    return {s: f05(pred[s], gold.get(s, ())) for s in s1_ids}, pred


def decompose(df, gold, s1_ids, pred, country):
    """Where the lost F0.5 goes: per country, share of total loss by error type."""
    cand = df.groupby("s1_id")["pool_id"].agg(set).to_dict()
    rows = []
    for s in s1_ids:
        g, p = set(gold.get(s, ())), set(pred[s])
        loss = 1.0 - f05(p, g)
        missed_blocking = len(g - cand.get(s, set()))
        if not g:
            kind = "singleton_fp" if p else "ok"
        elif not p:
            kind = "empty_pred_blocking" if missed_blocking == len(g) else "empty_pred"
        elif not (p & g):
            kind = "all_wrong"
        elif p - g and g - p:
            kind = "fp_and_fn"
        elif p - g:
            kind = "extra_fp"
        elif g - p:
            kind = "missed_blocking" if missed_blocking == len(g - p) else "missed_ranked"
        else:
            kind = "ok"
        rows.append((s, country.get(s), kind, loss, len(g), missed_blocking))
    e = pd.DataFrame(rows, columns=["s1_id", "country", "kind", "loss", "n_gold", "miss_block"])
    n = len(e)
    print(f"  gold-size distribution: {e['n_gold'].value_counts().sort_index().head(8).to_dict()}", flush=True)
    t = e[e["kind"] != "ok"].groupby(["country", "kind"]).agg(n=("loss", "size"), loss=("loss", "sum"))
    t["F05_pts_lost"] = t["loss"] / n
    print("  loss decomposition (F05_pts_lost sums to 1 - macro F0.5):", flush=True)
    print(t.sort_values("loss", ascending=False).to_string(), flush=True)
    return e


def oracle(df, gold, s1_ids):
    """Macro F0.5 of a perfect decision restricted to the candidates (the blocking ceiling)."""
    cand = df.groupby("s1_id")["pool_id"].agg(set).to_dict()
    return float(np.mean([f05(cand.get(s, set()) & set(gold.get(s, ())), gold.get(s, ())) for s in s1_ids]))


def log(cfg, tag, mode, val_f05, loco_f05, notes):
    """Append one row to logs/experiments.csv."""
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        h = ""
    path = os.path.join(cfg["paths"]["logs_dir"], "experiments.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([datetime.datetime.now().strftime("%Y-%m-%dT%H:%M"), h, f"{tag} ({mode})", "",
                                "" if val_f05 is None else f"{val_f05:.4f}",
                                "" if loco_f05 is None else f"{loco_f05:.4f}", notes])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", default="fold0", choices=["fold0", "loco"])
    ap.add_argument("--drop", default="")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--leaves", type=int)
    ap.add_argument("--min_leaf", type=int)
    args = ap.parse_args()
    if os.name == "nt":
        from entity_matcher.no_throttle import disable_throttling
        disable_throttling()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    params = lgb_params(cfg)
    for k, v in (("learning_rate", args.lr), ("num_leaves", args.leaves), ("min_data_in_leaf", args.min_leaf)):
        if v is not None:
            params[k] = v
    feats, y, gold, country, val, row_country = load(cfg)
    drop = {c for c in args.drop.split(",") if c}
    cols = [c for c in feats.columns if c not in ID_COLS and c not in drop]
    X = feats[cols].to_numpy(np.float32)
    note = f"{len(cols)} feats; lr {params['learning_rate']} leaves {params['num_leaves']}"
    print(f"[{args.tag}] {feats['s1_id'].nunique():,} S1, {len(feats):,} rows, {note}", flush=True)

    if args.mode == "fold0":
        m, p = fit(X[~val], y[~val], X[val], y[val], params)
        df = feats.loc[val, ["s1_id", "pool_id"]].reset_index(drop=True)
        df["prob"], df["label"] = p, y[val]
        ids = list(pd.unique(df["s1_id"]))
        best, total = tune(df, gold, ids, verbose=False)
        sc, pred = scores(df, gold, ids, best)
        by_c = pd.Series(sc).groupby(pd.Series({s: country.get(s) for s in ids})).mean()
        print(f"  trees {m.best_iteration} | logloss {m.best_score['valid_0']['binary_logloss']:.5f} | "
              f"decision {best}", flush=True)
        print(f"  fold-0 macro F0.5 {total:.4f} | " + " | ".join(f"{c} {v:.4f}" for c, v in by_c.items())
              + f" | blocking ceiling {oracle(df, gold, ids):.4f}", flush=True)
        decompose(df, gold, ids, pred, country)
        os.makedirs(os.path.join(cfg["paths"]["artefacts_dir"], "exp"), exist_ok=True)
        df.to_parquet(os.path.join(cfg["paths"]["artefacts_dir"], "exp", f"preds_{args.tag}.parquet"))
        imp = sorted(zip(cols, m.feature_importance("gain")), key=lambda x: -x[1])
        print("  top gain: " + ", ".join(c for c, _ in imp[:15]), flush=True)
        log(cfg, args.tag, "fold0", total, None, note + "; " + " ".join(f"{c}={v:.4f}" for c, v in by_c.items()))
    else:
        res = []
        for a, b in (("us", "india"), ("india", "us")):
            tr = ~val & (row_country == a)
            va_a, va_b = val & (row_country == a), val & (row_country == b)
            m, pa = fit(X[tr], y[tr], X[va_a], y[va_a], params)
            pb = m.predict(X[va_b], num_iteration=m.best_iteration)
            da = feats.loc[va_a, ["s1_id", "pool_id"]].reset_index(drop=True)
            da["prob"] = pa
            db = feats.loc[va_b, ["s1_id", "pool_id"]].reset_index(drop=True)
            db["prob"] = pb
            ids_a, ids_b = list(pd.unique(da["s1_id"])), list(pd.unique(db["s1_id"]))
            pa_best, own = tune(da, gold, ids_a, verbose=False)
            sc, _ = scores(db, gold, ids_b, pa_best)
            transfer = float(np.mean(list(sc.values())))
            _, target = tune(db, gold, ids_b, verbose=False)
            res.append(transfer)
            print(f"  train {a} -> {b}: own-country {own:.4f} | transfer {transfer:.4f} | "
                  f"tuned on target {target:.4f} | decision {pa_best}", flush=True)
        loco = float(np.mean(res))
        print(f"  LOCO mean (transfer) {loco:.4f}", flush=True)
        log(cfg, args.tag, "loco", None, loco, note + f"; us->india {res[0]:.4f} india->us {res[1]:.4f}")


if __name__ == "__main__":
    main()
