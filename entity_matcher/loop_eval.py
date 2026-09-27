"""One loop experiment = champion (v11_qwen) + one change, on the fixed fit / dev / lockbox split (entity_matcher.splits).

  1. stacker trained on FIT S1s (labels of fit only), predicting dev, lockbox and the test set
  2. fit-split cross-fitted OOF (4 quarters) -> decision tuning / calibration are fitted HERE ONLY
  3. dev scored by entity_matcher.local_score (the only reader of dev labels); paired bootstrap vs the champion
  4. test predictions: India/US (seen countries) from this experiment, unseen countries (France) copied from
     v11 unchanged; France churn is reported both for the composed file (0 by construction) and "if applied"
  5. --final (submission gate only): stacker retrained on ALL fold-0 S1s, frozen decision, write output/<out>/

  python -m entity_matcher.loop_eval --tag E000_champion
  python -m entity_matcher.loop_eval --tag E001_exact --reuse E000_champion --method expf_exact
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from entity_matcher.blend_eval import logit
from entity_matcher.decide import decide, tune
from entity_matcher.evaluate import f05, fold_of
from entity_matcher.io_utils import write_tsv
from entity_matcher.local_score import load_gold, paired_bootstrap, read_id_lists, score_file, split_ids
from entity_matcher.stack_eval import STACK_PARAMS, apply_fill, build_X, context, load_ce

# qwen = Qwen3-4B scores as they existed when v11 was built (precomputed/ce_qwen_v11)
CHAMP_EXTRA = ["ce_large=precomputed/ce_large_out", "ce_full=precomputed/ce_full_out", "qwen=precomputed/ce_qwen_v11"]
# test-time coverage of the contested-only CEs: own score only where the lgbm/ce_full logit blend is in
# [0.02, 0.98] (the XL / Qwen contested set), ce_full elsewhere -> dev / lockbox rows get the same
EVAL_FILL = ["qwen=ce_full:0.02:0.98"]
PAIRS = "precomputed/full_out/oof_train_full_part*.parquet"
LGBM_TEST = "precomputed/full_out/probs_model_full_part*.parquet"
SWAP = "ce_large=precomputed/ce_france_out"
SHIFT = "france:-1"
V11 = "output/v11_qwen/matching_results.tsv.gz"
EXP = "artefacts/exp"


def read_pairs(pattern, cols=None, min_prob=0.001):
    files = sorted(glob.glob(pattern))
    col = "lgbm_prob" if "lgbm_prob" in pq.read_schema(files[0]).names else "prob"
    flt = [(col, ">=", min_prob)] if min_prob else None
    df = pd.concat([pd.read_parquet(f, columns=cols, filters=flt) for f in files], ignore_index=True)
    return df.rename(columns={"prob": "lgbm_prob"})


def dev_candidates(path, ids):
    """Full blocking candidate lists (before any prob filter) of the given S1s, cached as a TSV."""
    if not os.path.exists(path):
        c = read_pairs(PAIRS, cols=["s1_id", "pool_id"], min_prob=None)
        c = c[c["s1_id"].isin(set(ids))]
        write_tsv(ids, c.groupby("s1_id")["pool_id"].agg(list).to_dict(), path, "candidate_entity_ids")
    return path


def attach_ces(df, kind, extra):
    df = df.merge(load_ce("precomputed/ce_out", kind), on=["s1_id", "pool_id"], how="left")
    for name, d in extra.items():
        df = df.merge(load_ce(d, kind).rename(columns={"ce_prob": f"{name}_prob"}), on=["s1_id", "pool_id"], how="left")
    return df.reset_index(drop=True)


def ce_avg_cols(df, avg):
    """--avg name=a+b: replace <a>_prob by the logit mean of <a>_prob and <b>_prob (where both present)."""
    for spec in avg:
        name, members = spec.split("=")
        cols = [f"{m}_prob" for m in members.split("+")]
        z = np.nanmean(np.stack([logit(df[c].to_numpy(dtype=np.float64)) for c in cols]), axis=0)
        any_na = df[cols].isna().any(axis=1).to_numpy()
        df[f"{name}_prob"] = np.where(any_na, df[f"{cols[0]}"], 1 / (1 + np.exp(-z)))
        df.drop(columns=[c for c in cols if c != f"{name}_prob"], inplace=True)
    return df


def build_fold0(extra, fills, eval_fills, raw, avg, part_of):
    all_pairs = read_pairs(PAIRS)
    df = attach_ces(all_pairs[all_pairs["fold"] == 0].drop(columns=["fold"]), "oof_fold0", extra)
    part = df["s1_id"].map(part_of)
    is_fit = (part == "fit").to_numpy()
    names = [e for e in extra]
    filled = {f.split("=")[0] + "_prob" for f in fills + eval_fills}
    for c in ["ce_prob"] + [f"{e}_prob" for e in names]:
        print(f"{c}: missing on {int(df[c].isna().sum()):,} of {len(df):,} fold-0 pairs", flush=True)
        if c not in filled:
            df[c] = df[c].fillna(0.0)  # stack_submit --ce_fill zero on the training side
    # fit rows: the champion's training-side coverage; dev / lockbox rows: the test-side coverage
    fit_side, eval_side = apply_fill(df.copy(), fills), apply_fill(df.copy(), eval_fills)
    for c in filled:
        df[c] = np.where(is_fit, fit_side[c].to_numpy(), eval_side[c].to_numpy())
        df[c] = df[c].fillna(0.0)
    del fit_side, eval_side
    df = ce_avg_cols(df, avg)
    ex = tuple(n for n in names if f"{n}_prob" in df.columns)
    X = build_X(df, all_pairs[["s1_id", "pool_id", "lgbm_prob"]], "train", ex)
    if raw:
        from entity_matcher.raw_feats import raw_features
        X = X.join(raw_features(df, "train").astype(np.float32))
    return df[["s1_id", "pool_id"]], X, part.to_numpy(), ex


def build_test(extra, fills, raw, avg):
    """Test pairs and stacker matrix exactly as entity_matcher.stack_submit builds them (fill from lgbm_prob, --fill, --swap)."""
    lg = read_pairs(LGBM_TEST)
    te = attach_ces(lg, "test", extra)
    for c in ["ce_prob"] + [f"{e}_prob" for e in extra]:
        if not any(f.startswith(f"{c[:-5]}=") for f in fills):
            te[c] = te[c].fillna(te["lgbm_prob"])
    te = apply_fill(te, fills)
    col, d = SWAP.split("=")
    if f"{col}_prob" in te.columns:
        files = sorted(glob.glob(os.path.join(d, "ce_test_unseen_part*.parquet")))
        files += sorted(glob.glob(os.path.join(d, "ce_extra_test_part*.parquet")))
        sw = pd.concat([pd.read_parquet(p) for p in files]).drop_duplicates(["s1_id", "pool_id"])
        sw = sw[sw["s1_id"].isin(unseen_s1())].set_index(["s1_id", "pool_id"])["ce_prob"]
        new = pd.Series(pd.MultiIndex.from_frame(te[["s1_id", "pool_id"]]).map(sw), index=te.index)
        print(f"swap {col}: {new.notna().sum():,} test pairs replaced from {d}", flush=True)
        te[f"{col}_prob"] = new.fillna(te[f"{col}_prob"])
    te = ce_avg_cols(te, avg)
    ex = tuple(n for n in extra if f"{n}_prob" in te.columns)
    X = build_X(te, lg, "test", ex)
    del lg
    if raw:
        from entity_matcher.raw_feats import raw_features
        X = X.join(raw_features(te, "test").astype(np.float32))
    return te[["s1_id", "pool_id"]], X


_UNSEEN = None


def test_country():
    s1 = pd.read_parquet("artefacts/test/s1.parquet", columns=["entity_id", "country_norm"])
    return dict(zip(s1["entity_id"], s1["country_norm"]))


def unseen_s1():
    """Test S1s of countries absent from training (open set; France today)."""
    global _UNSEEN
    if _UNSEEN is None:
        tr_c = set(pd.read_parquet("artefacts/train/s1.parquet", columns=["country_norm"])["country_norm"])
        _UNSEEN = {s for s, c in test_country().items() if c not in tr_c}
    return _UNSEEN


def train_stacker(X, y, s1, params, seeds):
    """stack_submit recipe: early stopping on the crc32 % 20 == 15 slice, refit on all rows at best_iteration;
    one model per seed (seed bagging when len(seeds) > 1)."""
    va = pd.Series(s1).map(lambda s: fold_of(s, 20) == 15).to_numpy()
    models = []
    for sd in seeds:
        p = dict(params, seed=sd)
        m = lgb.train(p, lgb.Dataset(X[~va], y[~va]), 3000, valid_sets=[lgb.Dataset(X[va], y[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        models.append(lgb.train(p, lgb.Dataset(X, y), m.best_iteration))
    return models


def predict(models, X):
    return np.mean([m.predict(X) for m in models], axis=0)


def stage2_X(X, keys, p1):
    """Stage-1 stacker probability, its logit and per-S1 context (rank, gap, max, sum, n>0.5, second best)."""
    tmp = pd.DataFrame({"s1_id": keys["s1_id"].to_numpy(), "p1": p1}, index=X.index)
    ctx = context(tmp, "p1", "s2")
    return X.assign(p1=p1, p1_logit=logit(p1)).join(ctx).astype(np.float32)


def labels(s1, pool, gold):
    return np.array([int(p in gold.get(s, ())) for s, p in zip(s1, pool)])


def shift_unseen(te):
    """Champion's France logit shift (applies only to the named country, as in stack_submit)."""
    c, delta = SHIFT.split(":")
    cc = test_country()
    in_c = te["s1_id"].map(cc).eq(c).to_numpy()
    p = te["prob"].to_numpy()
    te["prob"] = np.where(in_c, 1 / (1 + np.exp(-(logit(p) + float(delta)))), p)
    return te


def churn(new, ref, countries):
    """Per country: share of S1 whose predicted set differs, and match counts (ref -> new)."""
    rows = []
    for s, c in countries.items():
        a, b = set(ref.get(s, ())), set(new.get(s, ()))
        rows.append((c, a != b, len(a), len(b)))
    d = pd.DataFrame(rows, columns=["country", "changed", "m_ref", "m_new"])
    g = d.groupby("country").agg(S1=("changed", "size"), changed=("changed", "mean"), m_ref=("m_ref", "sum"),
                                 m_new=("m_new", "sum"))
    g["m_delta"] = g["m_new"] - g["m_ref"]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--hypothesis", default="")
    ap.add_argument("--extra", action="append", default=None, help="name=folder (default: champion's three)")
    ap.add_argument("--fill", action="append", default=None, help="fit / test rows; default: qwen=ce_full")
    ap.add_argument("--eval_fill", action="append", default=None,
                    help="dev / lockbox rows, mimicking test coverage; default: qwen=ce_full:0.02:0.98")
    ap.add_argument("--test_fill", action="append", default=None, help="test rows; default: same as --fill")
    ap.add_argument("--avg", action="append", default=[], help="name=a+b: logit-average CE columns a and b into a")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--params", default="", help="JSON overrides of STACK_PARAMS")
    ap.add_argument("--stage2", action="store_true", help="second-pass stacker on stage-1 per-S1 context")
    ap.add_argument("--seeds", default="42", help="comma list; >1 = seed-bagged stacker")
    ap.add_argument("--method", default="expf", choices=["expf", "expf_exact"])
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--tune", default="", help="grid tuned on fit OOF, e.g. 'expf:1,1.5,2;expf_exact:1,1.5,2'")
    ap.add_argument("--tune_by_country", action="store_true",
                    help="with --tune: separate decision parameters per seen country (fit OOF); unseen use the pooled ones")
    ap.add_argument("--calib", default="", choices=["", "isotonic"], help="calibrator fitted on fit OOF")
    ap.add_argument("--reuse", default="", help="tag whose saved probabilities to reuse (decision-only change)")
    ap.add_argument("--baseline", default="E000_champion")
    ap.add_argument("--no_test", action="store_true")
    ap.add_argument("--final", default="", help="submission gate: output dir; retrains on ALL fold-0 S1s")
    args = ap.parse_args()
    if os.name == "nt":
        from entity_matcher.no_throttle import disable_throttling
        disable_throttling()
    t0 = time.time()
    extra = dict(e.split("=") for e in (CHAMP_EXTRA if args.extra is None else args.extra))
    fills = [f for f in (["qwen=ce_full"] if args.fill is None else args.fill) if f.split("=")[0] in extra]
    eval_fills = [f for f in (EVAL_FILL if args.eval_fill is None else args.eval_fill) if f.split("=")[0] in extra]
    test_fills = fills if args.test_fill is None else [f for f in args.test_fill if f.split("=")[0] in extra]
    params = dict(STACK_PARAMS, **json.loads(args.params or "{}"))
    seeds = [int(s) for s in args.seeds.split(",")]
    out = os.path.join(EXP, args.tag)
    os.makedirs(out, exist_ok=True)
    fit, dev, lock = (split_ids(k)["s1_id"].tolist() for k in ("fit", "dev", "lockbox"))
    part_of = {**{s: "fit" for s in fit}, **{s: "dev" for s in dev}, **{s: "lockbox" for s in lock}}

    if args.final:
        return final_gate(args, extra, fills, test_fills, params, seeds, part_of, out)

    src_dir = os.path.join(EXP, args.reuse) if args.reuse else out
    if not args.reuse:
        keys, X, part, ex = build_fold0(extra, fills, eval_fills, args.raw, args.avg, part_of)
        is_fit = part == "fit"
        gold_fit = load_gold(fit)  # FIT labels only
        kf = keys[is_fit]
        y = labels(kf["s1_id"], kf["pool_id"], gold_fit)
        Xf = X[is_fit]
        # fit-split cross-fitted OOF: 4 quarters of fit S1s, each predicted by a stacker trained on the other 3
        q = kf["s1_id"].map(lambda s: fold_of(s[::-1], 4)).to_numpy()  # independent of the % 20 early-stop slice
        oof = np.zeros(len(kf))
        for k in range(4):
            oof[q == k] = predict(train_stacker(Xf[q != k], y[q != k], kf["s1_id"].to_numpy()[q != k], params, seeds[:1]),
                                  Xf[q == k])
        models = train_stacker(Xf, y, kf["s1_id"].to_numpy(), params, seeds)
        cols = list(Xf.columns)
        if args.stage2:  # stage 2 trained on fit rows with stage-1 OOF context; its own OOF by the same quarters
            X2f = stage2_X(Xf, kf, oof)
            oof2 = np.zeros(len(kf))
            for k in range(4):
                oof2[q == k] = predict(train_stacker(X2f[q != k], y[q != k], kf["s1_id"].to_numpy()[q != k], params,
                                                     seeds[:1]), X2f[q == k])
            models2 = train_stacker(X2f, y, kf["s1_id"].to_numpy(), params, seeds)
            cols2 = list(X2f.columns)
            print(f"stage 2: {len(cols2)} features, trees {[m.num_trees() for m in models2]}", flush=True)
            del X2f
            oof = oof2

        def final_prob(Xp, kp):
            p1 = predict(models, Xp[cols])
            return predict(models2, stage2_X(Xp[cols], kp, p1)[cols2]) if args.stage2 else p1

        kf.assign(prob=oof).to_parquet(os.path.join(out, "fit_oof_probs.parquet"), index=False)
        print(f"stacker: {is_fit.sum():,} fit pairs, {len(cols)} features, trees {[m.num_trees() for m in models]}",
              flush=True)
        for name in ("dev", "lockbox"):
            sel = part == name
            assert "label" not in X.columns, "label leaked into inference frame"
            keys[sel].assign(prob=final_prob(X[sel], keys[sel])).to_parquet(os.path.join(out, f"{name}_probs.parquet"),
                                                                            index=False)
        del X, Xf
        if not args.no_test:
            tk, Xt = build_test(extra, test_fills, args.raw, args.avg)
            tk.assign(prob=final_prob(Xt, tk)).to_parquet(os.path.join(out, "test_probs.parquet"), index=False)
            del Xt
    fit_time = time.time() - t0

    # decision layer: calibration + parameters fitted on the fit OOF only
    gold_fit = load_gold(fit)
    oof = pd.read_parquet(os.path.join(src_dir, "fit_oof_probs.parquet"))
    calib = None
    if args.calib == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        calib = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        calib.fit(oof["prob"].to_numpy(), labels(oof["s1_id"], oof["pool_id"], gold_fit))
        oof["prob"] = calib.predict(oof["prob"].to_numpy())
    d_params = {"method": args.method, "alpha": args.alpha, "one_to_one": True}
    if args.tune:
        if args.tune == "full":  # decide.tune's full grid (thresholds x relative-to-best ratios + expf) + expf_exact
            grid = [{"method": "thresh", "t": float(t), "r": r} for t in np.round(np.arange(0.10, 0.96, 0.05), 2)
                    for r in (0.0, 0.2, 0.4, 0.6, 0.8)]
            grid += [{"method": m, "alpha": a} for m in ("expf", "expf_exact") for a in (1.0, 1.25, 1.5, 1.75, 2.0)]
        else:
            grid = [{"method": m, "alpha": float(a)} for part_ in args.tune.split(";") for m, al in [part_.split(":")]
                    for a in al.split(",")]
        d_params, fit_score = tune(oof, gold_fit, fit, verbose=False, grid=grid, o2o_opts=(True,))
        print(f"tuned on fit OOF: {d_params} -> fit-OOF macro F0.5 {fit_score:.5f}", flush=True)
    by_c = {}  # country -> decision params (only with --tune_by_country)
    tr_country = dict(zip(*[pd.read_parquet("artefacts/train/s1.parquet", columns=["entity_id", "country_norm"])[c]
                           for c in ("entity_id", "country_norm")])) if args.tune_by_country else {}
    if args.tune and args.tune_by_country:
        for c in sorted(set(tr_country[s] for s in fit)):
            ids_c = [s for s in fit if tr_country[s] == c]
            sub = oof[oof["s1_id"].isin(set(ids_c))]
            by_c[c], sc = tune(sub, gold_fit, ids_c, verbose=False, grid=grid, o2o_opts=(True,))
            print(f"tuned on fit OOF [{c}]: {by_c[c]} -> {sc:.5f}", flush=True)

    def decide_c(pr, ids, cmap):
        """decide() with per-country parameters when tuned by country (countries never share pool records)."""
        if not by_c:
            return decide(pr, ids, d_params)
        out_ = {}
        c_of = {s: cmap.get(s) for s in ids}
        for c in set(c_of.values()):
            ids_c = [s for s in ids if c_of[s] == c]
            out_.update(decide(pr[pr["s1_id"].isin(set(ids_c))], ids_c, by_c.get(c, d_params)))
        return out_

    fit_oof_score = float(np.mean([f05(v, gold_fit.get(s, ())) for s, v in decide_c(oof, fit, tr_country).items()]))

    def decided(name, ids):
        pr = pd.read_parquet(os.path.join(src_dir, f"{name}_probs.parquet"))
        if calib is not None:
            pr["prob"] = calib.predict(pr["prob"].to_numpy())
        return pr, decide_c(pr, ids, tr_country)

    for name, ids in (("dev", dev), ("lockbox", lock)):
        write_tsv(ids, decided(name, ids)[1], os.path.join(out, f"{name}_matching.tsv"), "matched_entity_ids")
    cands = dev_candidates("artefacts/splits/dev_candidates.tsv", dev)
    rep = score_file(os.path.join(out, "dev_matching.tsv"), "dev", cands, os.path.join(out, "dev_per_entity.parquet"))
    base = os.path.join(EXP, args.baseline, "dev_per_entity.parquet")
    if args.tag != args.baseline and os.path.exists(base):
        a = pd.read_parquet(base).set_index("s1_id")["f05"]
        b = pd.read_parquet(os.path.join(out, "dev_per_entity.parquet")).set_index("s1_id")["f05"].reindex(a.index)
        rep["delta_vs_baseline"] = float((b - a).mean())
        rep["delta_ci95"] = list(paired_bootstrap(a.to_numpy(), b.to_numpy()))

    # test: seen countries from this experiment, unseen (France) copied from v11
    if os.path.exists(os.path.join(src_dir, "test_probs.parquet")):
        cc = test_country()
        ids = list(cc)
        tp = pd.read_parquet(os.path.join(src_dir, "test_probs.parquet"))
        if calib is not None:
            tp["prob"] = calib.predict(tp["prob"].to_numpy())
        new = decide_c(shift_unseen(tp), ids, cc)
        v11 = read_id_lists(V11)
        uns = unseen_s1()
        composed = {s: (v11.get(s, []) if s in uns else new.get(s, [])) for s in ids}
        write_tsv(ids, composed, os.path.join(out, "test_matching.tsv"), "matched_entity_ids")
        ch_comp, ch_raw = churn(composed, v11, cc), churn(new, v11, cc)
        print("test churn vs v11, submitted composition (France = v11):\n" + ch_comp.round(4).to_string(), flush=True)
        print("test churn vs v11 IF applied to France too (not submitted):\n" + ch_raw.round(4).to_string(), flush=True)
        rep["churn_vs_v11"] = ch_comp.round(5).reset_index().to_dict("records")
        rep["churn_if_france_applied"] = ch_raw.round(5).reset_index().to_dict("records")
    rep.update({"tag": args.tag, "hypothesis": args.hypothesis, "reuse": args.reuse, "extra": extra, "fill": fills,
                "eval_fill": eval_fills, "test_fill": test_fills, "avg": args.avg, "raw": args.raw, "params": params, "seeds": seeds,
                "calib": args.calib, "decision": d_params, "decision_by_country": by_c, "fit_oof_f05": fit_oof_score,
                "runtime_min": round((time.time() - t0) / 60, 1), "fit_predict_min": round(fit_time / 60, 1)})
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    if calib is not None:
        import pickle
        with open(os.path.join(out, "calib.pkl"), "wb") as f:
            pickle.dump(calib, f)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("params", "churn_vs_v11", "churn_if_france_applied")},
                     indent=1), flush=True)

    fr = [r for r in rep.get("churn_if_france_applied", []) if r["country"] == "france"]
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    os.makedirs("logs", exist_ok=True)
    with open("logs/experiments.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([time.strftime("%Y-%m-%dT%H:%M"), f"{head}+wt", f"{args.tag}: {args.hypothesis}",
                                f"{rep['blocking_recall']:.4f}", f"{rep['macro_f05']:.5f}", "",
                                f"dev ci95 {rep['ci95'][0]:.5f}-{rep['ci95'][1]:.5f}; P {rep['pair_precision']:.4f} "
                                f"R {rep['pair_recall']:.4f}; " + " ".join(f"{k} {v:.5f}" for k, v in rep["by_country"].items())
                                + (f"; delta {rep['delta_vs_baseline']:+.5f} ci95 {rep['delta_ci95'][0]:+.5f}.."
                                   f"{rep['delta_ci95'][1]:+.5f}" if "delta_vs_baseline" in rep else "")
                                + f"; decision {d_params}; fitOOF {fit_oof_score:.5f}"
                                + (f"; France churn 0 (v11 kept), if applied {fr[0]['changed']:.4f} S1 "
                                   f"{int(fr[0]['m_delta']):+d} matches" if fr else "")
                                + f"; {rep['runtime_min']} min"])


def final_gate(args, extra, fills, test_fills, params, seeds, part_of, out):
    """Submission gate step 3-5: retrain the frozen recipe on ALL fold-0 S1s, predict test, apply the decision
    frozen in artefacts/exp/<tag>/report.json (+ calib.pkl), France from v11, write + validate."""
    import pickle
    import yaml
    from entity_matcher.run import write_submission
    with open(os.path.join(out, "report.json")) as f:
        rep = json.load(f)
    d_params = rep["decision"]
    calib = None
    if os.path.exists(os.path.join(out, "calib.pkl")):
        with open(os.path.join(out, "calib.pkl"), "rb") as f:
            calib = pickle.load(f)
    # the final model is trained like the champion: training-side CE coverage on every fold-0 row
    keys, X2, _, _ = build_fold0(extra, fills, fills, args.raw, args.avg, part_of)
    gold = load_gold(list(part_of))  # gate only: fit + dev + lockbox labels for the final retrain
    y = labels(keys["s1_id"], keys["pool_id"], gold)
    models = train_stacker(X2, y, keys["s1_id"].to_numpy(), params, seeds)
    cols = list(X2.columns)
    del X2
    tk, Xt = build_test(extra, test_fills, args.raw, args.avg)
    tk = tk.assign(prob=predict(models, Xt[cols]))
    del Xt
    tk.to_parquet(os.path.join(out, "final_test_probs.parquet"), index=False)
    if calib is not None:
        tk["prob"] = calib.predict(tk["prob"].to_numpy())
    cc = test_country()
    ids = list(cc)
    new = decide(shift_unseen(tk), ids, d_params)
    v11 = read_id_lists(V11)
    uns = unseen_s1()
    composed = {s: (v11.get(s, []) if s in uns else new.get(s, [])) for s in ids}
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    os.makedirs(args.final, exist_ok=True)
    write_submission(cfg, ids, composed, args.final, extra_cands=tk[["s1_id", "pool_id"]])
    written = read_id_lists(os.path.join(args.final, "matching_results.tsv"))
    print("final churn vs v11:\n" + churn(written, v11, cc).round(4).to_string(), flush=True)
    print("final churn vs v11 IF France applied (not submitted):\n" + churn(new, v11, cc).round(4).to_string(), flush=True)
    with open(os.path.join(args.final, "blend.json"), "w") as f:
        json.dump({"recipe": "entity_matcher.loop_eval --final", "tag": args.tag, "extra": extra, "fill": fills, "test_fill": test_fills, "avg": args.avg,
                   "raw": args.raw, "params": params, "seeds": seeds, "calib": rep.get("calib", ""),
                   "decision": d_params, "shift": SHIFT, "swap": SWAP, "unseen_countries": "copied from " + V11,
                   "train_s1": "all fold-0 (fit + dev + lockbox)", "pairs": PAIRS, "lgbm_test": LGBM_TEST}, f, indent=1)


if __name__ == "__main__":
    main()
