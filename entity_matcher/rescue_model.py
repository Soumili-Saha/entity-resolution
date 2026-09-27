"""Step 3d-g: RESCUE model for the extra blocking keys (entity_matcher.rescue_keys, K1|K2 only) on top of the champion.

Pairs: new K1|K2 pairs not in the step-3 candidate set. Features: string similarity (rapidfuzz), token / char-3gram
Jaccard, house-number / postcode agree-conflict, key hits, plus context: the S1's current champion decision
(max stacker prob, number of predicted matches) and how strongly some S1 already claims the pool record
(max LightGBM prob over all S1s of the pair table).
Model: LightGBM on FIT pairs only (labels of fit), cross-fitted OOF on 4 fit quarters -> thresholds tuned on fit
(macro F0.5 with the champion's fit-OOF decisions as base): add the best rescued pair of an S1 when p >= t
(t_empty for S1s the champion leaves empty), one-to-one (pool records already predicted, or rescued with a higher
prob for another S1, are skipped). Dev: base = E000 dev decisions; scored by entity_matcher.local_score vs E000.
Test (India/US only): base = the ORIGINAL v11 file; France rows untouched.

  python -m entity_matcher.rescue_model --tag R001
"""
import argparse
import glob
import json
import os
import re
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from entity_matcher.cpus import n_cpus
from entity_matcher.decide import decide
from entity_matcher.evaluate import f05, fold_of
from entity_matcher.io_utils import write_tsv
from entity_matcher.local_score import load_gold, paired_bootstrap, read_id_lists, score_file, split_ids

OUT = "artefacts/rescue"
EXP = "artefacts/exp"
COLS = ["entity_id", "business_name", "name_core", "name_skel", "addr_clean", "postal", "nums"]
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50,
          "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": 42}
_DIG = re.compile(r"\d+")


def load_rows(split, k, ids):
    return ds.dataset(f"artefacts/{split}/s{k}.parquet").to_table(
        columns=COLS, filter=pc.field("entity_id").isin(pa.array(sorted(ids), type=pa.string()))).to_pandas()


def jac(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def grams(s, n=3):
    s = s.replace(" ", "")
    return {s[i:i + n] for i in range(max(len(s) - n + 1, 0))}


def pair_features(p, split):
    a = load_rows(split, 1, set(p["s1_id"])).set_index("entity_id").reindex(p["s1_id"]).fillna("").reset_index(drop=True)
    b = pd.concat([load_rows(split, k, set(p["pool_id"])) for k in (2, 3)]).set_index("entity_id")
    b = b.reindex(p["pool_id"]).fillna("").reset_index(drop=True)
    w = n_cpus()
    X = pd.DataFrame(index=p.index)
    for c in ("name_core", "name_skel", "addr_clean"):
        x, y = a[c].tolist(), b[c].tolist()
        X[f"{c}_tset"] = cpdist(x, y, scorer=fuzz.token_set_ratio, workers=w, dtype=np.float32) / 100
        X[f"{c}_ratio"] = cpdist(x, y, scorer=fuzz.ratio, workers=w, dtype=np.float32) / 100
    X["name_jw"] = cpdist(a["name_core"].tolist(), b["name_core"].tolist(), scorer=JaroWinkler.normalized_similarity,
                          workers=w, dtype=np.float32)
    X["name_lev"] = cpdist(a["name_core"].tolist(), b["name_core"].tolist(), scorer=Levenshtein.normalized_similarity,
                           workers=w, dtype=np.float32)
    X["name_tok_jac"] = [jac(set(x.split()), set(y.split())) for x, y in zip(a["name_core"], b["name_core"])]
    X["name_3g_jac"] = [jac(grams(x), grams(y)) for x, y in zip(a["name_core"], b["name_core"])]
    sa = [_DIG.sub(" ", x) for x in a["addr_clean"]]
    sb = [_DIG.sub(" ", x) for x in b["addr_clean"]]
    X["street_tset"] = cpdist(sa, sb, scorer=fuzz.token_set_ratio, workers=w, dtype=np.float32) / 100
    X["addr_tok_jac"] = [jac(set(x.split()), set(y.split())) for x, y in zip(a["addr_clean"], b["addr_clean"])]
    X["addr_3g_jac"] = [jac(grams(x), grams(y)) for x, y in zip(a["addr_clean"], b["addr_clean"])]
    na, nb = [set(x.split()) for x in a["nums"]], [set(y.split()) for y in b["nums"]]
    X["num_shared"] = [len(x & y) for x, y in zip(na, nb)]
    X["num_conflict"] = [float(bool(x) and bool(y) and not (x & y)) for x, y in zip(na, nb)]
    fa = [x.split()[0] if x else "" for x in a["nums"]]
    fb = [y.split()[0] if y else "" for y in b["nums"]]
    X["first_num_eq"] = [float(x != "" and x == y) for x, y in zip(fa, fb)]
    X["postal_eq"] = ((a["postal"] == b["postal"]) & (a["postal"] != "")).astype(np.float32).to_numpy()
    X["postal_conflict"] = ((a["postal"] != b["postal"]) & (a["postal"] != "") & (b["postal"] != "")).astype(np.float32).to_numpy()
    X["b_addr_empty"] = (b["addr_clean"].str.len() == 0).astype(np.float32).to_numpy()
    X["b_nonlatin"] = [float(not re.search(r"[A-Za-z]", x)) for x in b["business_name"]]
    X["len_name_a"], X["len_name_b"] = a["name_core"].str.len().to_numpy(), b["name_core"].str.len().to_numpy()
    X["b_source"] = (p["pool_id"].str[:2] == "S3").astype(np.float32).to_numpy()
    for c in ("k1", "k2", "k3", "k4", "hits", "min_kfreq"):
        X[c] = p[c].to_numpy()
    return X.astype(np.float32)


def pool_claim(split):
    """max LightGBM prob with which any S1 of the step-3 table claims each pool record (prob >= 0.01)."""
    pat = "precomputed/full_out/oof_train_full_part*.parquet" if split == "train" else "precomputed/full_out/probs_model_full_part*.parquet"
    df = pd.concat([pd.read_parquet(f, columns=["pool_id", "prob"], filters=[("prob", ">=", 0.01)])
                    for f in sorted(glob.glob(pat))])
    return df.groupby("pool_id")["prob"].max()


def context(p, probs, decided):
    """S1-side: max champion stacker prob, number of champion matches; pool-side claim added by caller."""
    mx = probs.groupby("s1_id")["prob"].max()
    X = pd.DataFrame(index=p.index)
    X["s1_max_prob"] = p["s1_id"].map(mx).fillna(0).to_numpy()
    X["s1_n_pred"] = p["s1_id"].map({s: len(v) for s, v in decided.items()}).fillna(0).to_numpy()
    X["s1_empty"] = (X["s1_n_pred"] == 0).astype(np.float32)
    return X.astype(np.float32)


def apply_rescue(p, prob, base, t, t_empty, taken):
    """Add, per S1, its best rescued pair when prob clears the threshold; one-to-one against taken ids and
    between rescues. Returns {s1: [ids]} (a copy of base with additions) and the list of added pairs."""
    d = p[["s1_id", "pool_id"]].assign(prob=prob)
    d = d[~d["pool_id"].isin(taken)].sort_values("prob", ascending=False)
    d = d[~d.duplicated("pool_id")]  # a pool record goes to the S1 that rescues it most confidently
    d = d[~d.duplicated("s1_id")]    # at most one rescued pair per S1
    empty = d["s1_id"].map(lambda s: len(base.get(s, ())) == 0).to_numpy()
    d = d[np.where(empty, d["prob"].to_numpy() >= t_empty, d["prob"].to_numpy() >= t)]
    out = {s: list(v) for s, v in base.items()}
    for s, q in zip(d["s1_id"], d["pool_id"]):
        out.setdefault(s, []).append(q)
    return out, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="R001")
    ap.add_argument("--base", default="E000_champion", help="experiment whose fit OOF / dev decisions are the base")
    ap.add_argument("--keys", default="k1,k2", help="keys whose pairs are scored (and would enter the candidate file)")
    ap.add_argument("--no_test", action="store_true")
    ap.add_argument("--prefilter", default="", help="pandas eval rule on the features, e.g. 'addr_clean_tset >= 0.6'")
    args = ap.parse_args()
    t0 = time.time()
    out = os.path.join(EXP, args.tag)
    os.makedirs(out, exist_ok=True)
    keys = args.keys.split(",")
    fit, dev, lock = (split_ids(k)["s1_id"].tolist() for k in ("fit", "dev", "lockbox"))
    part_of = {**{s: "fit" for s in fit}, **{s: "dev" for s in dev}, **{s: "lockbox" for s in lock}}
    p = pd.read_parquet(os.path.join(OUT, "new_pairs_train.parquet"))
    p = p[(p[keys] > 0).any(axis=1)].reset_index(drop=True)
    part = p["s1_id"].map(part_of).to_numpy()
    print(f"fold-0 rescue pairs ({'|'.join(keys)}): {len(p):,} | fit {np.sum(part == 'fit'):,} dev {np.sum(part == 'dev'):,} "
          f"lockbox {np.sum(part == 'lockbox'):,}", flush=True)
    X = pair_features(p, "train")
    X["pool_claim"] = p["pool_id"].map(pool_claim("train")).fillna(0).to_numpy().astype(np.float32)
    # context from the champion: fit rows from its fit OOF, dev / lockbox rows from its fit-trained predictions
    d_par = {"method": "expf", "alpha": 1.5, "one_to_one": True}
    oof = pd.read_parquet(os.path.join(EXP, args.base, "fit_oof_probs.parquet"))
    base_fit = decide(oof, fit, d_par)
    ctx = []
    for name, ids, pr, dec in (("fit", fit, oof, base_fit),
                               ("dev", dev, None, read_id_lists(os.path.join(EXP, args.base, "dev_matching.tsv"))),
                               ("lockbox", lock, None, read_id_lists(os.path.join(EXP, args.base, "lockbox_matching.tsv")))):
        pr = pr if pr is not None else pd.read_parquet(os.path.join(EXP, args.base, f"{name}_probs.parquet"))
        sel = part == name
        ctx.append(context(p[sel], pr, dec))
        if name == "dev":
            base_dev = dec
        if name == "lockbox":
            base_lock = dec
    X = X.join(pd.concat(ctx))
    gold_fit = load_gold(fit)
    if args.prefilter:  # rule-based stage before the model: only survivors are scored (and enter the candidate file)
        keep = X.eval(args.prefilter).to_numpy()
        print(f"prefilter {args.prefilter!r}: keeps {keep.sum():,} of {len(keep):,} fold-0 pairs", flush=True)
        p, X, part = p[keep].reset_index(drop=True), X[keep].reset_index(drop=True), part[keep]
    X.assign(s1_id=p["s1_id"].to_numpy(), pool_id=p["pool_id"].to_numpy(), part=part).to_parquet(
        os.path.join(OUT, "feat_train.parquet"), index=False)
    is_fit = part == "fit"
    pf = p[is_fit]
    y = np.array([int(b in gold_fit.get(a, ())) for a, b in zip(pf["s1_id"], pf["pool_id"])])
    Xf = X[is_fit]
    print(f"fit rescue pairs {len(pf):,}, gold {y.sum():,} ({y.mean():.3%})", flush=True)

    def train(Xt, yt, s1):
        va = pd.Series(s1).map(lambda s: fold_of(s, 20) == 15).to_numpy()
        m = lgb.train(PARAMS, lgb.Dataset(Xt[~va], yt[~va]), 2000, valid_sets=[lgb.Dataset(Xt[va], yt[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        return lgb.train(PARAMS, lgb.Dataset(Xt, yt), max(m.best_iteration, 20))

    q = pf["s1_id"].map(lambda s: fold_of(s[::-1], 4)).to_numpy()
    oof_r = np.zeros(len(pf))
    for k in range(4):
        m = train(Xf[q != k], y[q != k], pf["s1_id"].to_numpy()[q != k])
        oof_r[q == k] = m.predict(Xf[q == k])
    from sklearn.metrics import roc_auc_score, average_precision_score
    print(f"fit OOF rescue AUC {roc_auc_score(y, oof_r):.4f}, AP {average_precision_score(y, oof_r):.4f}", flush=True)
    model = train(Xf, y, pf["s1_id"].to_numpy())
    imp = pd.Series(model.feature_importance("gain"), index=Xf.columns).sort_values(ascending=False)
    print("top features: " + ", ".join(f"{k} {v:.0f}" for k, v in imp.head(10).items()), flush=True)

    # thresholds tuned on FIT (cross-fitted OOF rescue probs + champion fit-OOF decisions as base)
    b_fit = np.array([f05(base_fit.get(s, ()), gold_fit.get(s, ())) for s in fit])
    taken_fit = {x for v in base_fit.values() for x in v}
    best = (0.0, 1.01, 1.01)
    grid_t = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    for t in grid_t:
        for te in [x for x in grid_t + [1.01] if x >= t]:
            new, added = apply_rescue(pf, oof_r, base_fit, t, te, taken_fit)
            ch = set(added["s1_id"])
            gain = sum(f05(new[s], gold_fit.get(s, ())) - f05(base_fit.get(s, ()), gold_fit.get(s, ())) for s in ch)
            gain /= len(fit)
            if gain > best[0]:
                best = (gain, t, te)
    gain, t, te = best
    print(f"tuned on fit OOF: t={t} t_empty={te} -> fit macro F0.5 gain {gain:+.6f} (base {b_fit.mean():.5f})", flush=True)

    # dev / lockbox: full-fit rescue model on top of the champion's decisions
    res = {"tag": args.tag, "keys": keys, "prefilter": args.prefilter, "fold0_pairs_scored": len(p), "t": t, "t_empty": te, "fit_gain": gain,
           "fit_oof_auc": float(roc_auc_score(y, oof_r))}
    for name, ids, base in (("dev", dev, base_dev), ("lockbox", lock, base_lock)):
        sel = part == name
        pr = model.predict(X[sel])
        taken = {x for v in base.values() for x in v}
        new, added = apply_rescue(p[sel], pr, base, t, te, taken)
        write_tsv(ids, new, os.path.join(out, f"{name}_matching.tsv"), "matched_entity_ids")
        added.to_parquet(os.path.join(out, f"{name}_added.parquet"), index=False)
        res[f"{name}_added"] = len(added)
    base_dir = os.path.join(EXP, args.base)
    rep = score_file(os.path.join(out, "dev_matching.tsv"), "dev", save_per=os.path.join(out, "dev_per_entity.parquet"))
    a = pd.read_parquet(os.path.join(base_dir, "dev_per_entity.parquet")).set_index("s1_id")["f05"]
    bb = pd.read_parquet(os.path.join(out, "dev_per_entity.parquet")).set_index("s1_id")["f05"].reindex(a.index)
    res.update(dev_f05=rep["macro_f05"], dev_ci95=rep["ci95"], dev_by_country=rep["by_country"],
               dev_delta=float((bb - a).mean()), dev_delta_ci95=list(paired_bootstrap(a.to_numpy(), bb.to_numpy())),
               dev_singleton_f05=rep["singleton_f05"])
    base_rep = json.load(open(os.path.join(base_dir, "report.json")))
    res["dev_singleton_f05_base"] = base_rep.get("singleton_f05")
    print(json.dumps(res, indent=1), flush=True)
    if not args.no_test:
        res.update(test_rescue(model, X.columns, t, te, keys, out, args.prefilter))
    res["runtime_min"] = round((time.time() - t0) / 60, 1)
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(res, f, indent=1)
    model.save_model(os.path.join(out, "rescue_model.txt"))


def test_rescue(model, cols, t, te, keys, out, prefilter=""):
    """India/US test: rescued pairs appended to the ORIGINAL v11 rows; France byte-identical."""
    from entity_matcher.loop_eval import V11, churn, test_country, unseen_s1
    p = pd.read_parquet(os.path.join(OUT, "new_pairs_test.parquet"))
    p = p[(p[keys] > 0).any(axis=1)].reset_index(drop=True)
    v11 = read_id_lists(V11)
    X = pair_features(p, "test")
    X["pool_claim"] = p["pool_id"].map(pool_claim("test")).fillna(0).to_numpy().astype(np.float32)
    tp = pd.read_parquet(os.path.join(EXP, "E000_champion", "test_probs.parquet"))
    X = X.join(context(p, tp, v11))
    if prefilter:
        keep = X.eval(prefilter).to_numpy()
        p, X = p[keep].reset_index(drop=True), X[keep].reset_index(drop=True)
    prob = model.predict(X[list(cols)])
    taken = {x for v in v11.values() for x in v}
    uns = unseen_s1()
    assert not p["s1_id"].isin(uns).any(), "rescue pairs must be India/US only"
    new, added = apply_rescue(p, prob, v11, t, te, taken)
    cc = test_country()
    ids = list(cc)
    composed = {s: (v11.get(s, []) if s in uns else new.get(s, [])) for s in ids}
    write_tsv(ids, composed, os.path.join(out, "test_matching.tsv"), "matched_entity_ids")
    p[["s1_id", "pool_id"]].assign(prob=prob).to_parquet(os.path.join(out, "test_rescue_probs.parquet"), index=False)
    added.to_parquet(os.path.join(out, "test_added.parquet"), index=False)
    ch = churn(composed, v11, cc)
    print("test churn vs v11 (France untouched):\n" + ch.round(5).to_string(), flush=True)
    return {"test_pairs_scored": len(p), "test_added": len(added), "test_cands_added_per_s1": len(p) / len(ids),
            "test_churn": ch.round(6).reset_index().to_dict("records")}


if __name__ == "__main__":
    main()
