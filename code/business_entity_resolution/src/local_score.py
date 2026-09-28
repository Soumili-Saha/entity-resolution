"""Official metric, implemented exactly as in the problem statement (Evaluation Criteria):

  per Source 1 entity:  P = |pred & gold| / |pred|,  R = |pred & gold| / |gold|
                        F0.5 = (1.25 * P * R) / (0.25 * P + R)
  singleton (gold empty): 1.0 if pred is empty, else 0.0
  gold non-empty, pred empty or no overlap: 0.0
  final score: plain mean over ALL Source 1 entities of the evaluation set (macro average).

Each S1 is scored independently against its own ID list: there is no transitive closure and no
cluster-level (pairwise) scoring. With TP = |pred & gold| the formula simplifies to
1.25 * TP / (0.25 * |gold| + |pred|), which is what f05 computes.

This module is the ONLY place that reads holdout (dev / lockbox) labels. Pipeline code writes
predictions to disk first; this scores them:

  python -m src.local_score --pred artefacts/exp/E000/dev_matching.tsv --split dev
"""
import argparse
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

BETA2 = 0.25  # beta = 0.5
SPLIT_DIR = "artefacts/splits"
GT_PATH = "dataset/train/train_ground_truth.tsv"


def f05(pred, gold):
    """Official per-entity F0.5 (pred/gold: iterables of S2/S3 ids)."""
    pred, gold = set(pred), set(gold)
    if not gold:
        return 1.0 if not pred else 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def read_id_lists(path):
    """{s1_id: [ids]} from a matching_results / candidate_pairs TSV (optionally .gz). Rejects duplicates
    the same way the portal does (duplicate S1 rows or duplicate ids inside one list raise)."""
    op = gzip.open if path.endswith(".gz") else open
    out = {}
    with op(path, "rt", encoding="utf-8", newline="") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        assert header[0] == "source1_entity_id" and len(header) == 2, f"bad header {header}"
        for line in f:
            s1, _, ids = line.rstrip("\r\n").partition("\t")
            lst = [x for x in ids.split(",") if x]
            assert s1 not in out, f"duplicate S1 row {s1}"
            assert len(lst) == len(set(lst)), f"duplicate id in list of {s1}"
            out[s1] = lst
    return out


def load_gold(ids=None):
    """{s1_id: set(ids)} from the training ground truth, restricted to ids when given."""
    gt = pd.read_csv(GT_PATH, sep="\t", dtype=str, keep_default_na=False, engine="pyarrow")
    keep = None if ids is None else set(ids)
    return {s: {x for x in m.split(",") if x} for s, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"])
            if keep is None or s in keep}


def split_ids(name):
    """S1 ids of a saved split (fit / dev / lockbox) with their country."""
    return pd.read_parquet(os.path.join(SPLIT_DIR, f"{name}.parquet"))


def bootstrap_ci(scores, n=1000, seed=0):
    """95% percentile CI of the mean by resampling entities."""
    rng = np.random.default_rng(seed)
    s = np.asarray(scores, dtype=np.float64)
    means = np.array([s[rng.integers(0, len(s), len(s))].mean() for _ in range(n)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap(a, b, n=1000, seed=0):
    """95% CI of mean(b - a) over the same entities (a, b aligned per-entity score arrays)."""
    return bootstrap_ci(np.asarray(b) - np.asarray(a), n, seed)


def score(pred, gold, ids, country=None, candidates=None):
    """Full report for predictions {s1: ids} on the S1 list ids. Missing S1 rows count as empty
    predictions (the portal would reject the file; the caller is warned via n_missing_rows)."""
    per = np.array([f05(pred.get(s, ()), gold.get(s, ())) for s in ids])
    tp = sum(len(set(pred.get(s, ())) & gold.get(s, set())) for s in ids)
    n_pred = sum(len(pred.get(s, ())) for s in ids)
    n_gold = sum(len(gold.get(s, ())) for s in ids)
    single = np.array([not gold.get(s) for s in ids])
    lo, hi = bootstrap_ci(per)
    rep = {"macro_f05": float(per.mean()), "ci95": [lo, hi], "n_s1": len(ids),
           "n_missing_rows": int(sum(s not in pred for s in ids)),
           "pair_precision": tp / max(n_pred, 1), "pair_recall": tp / max(n_gold, 1),
           "singleton_f05": float(per[single].mean()) if single.any() else None,
           "multi_f05": float(per[~single].mean()) if (~single).any() else None}
    if country is not None:
        c = pd.Series([country.get(s) for s in ids])
        rep["by_country"] = pd.Series(per).groupby(c).mean().round(6).to_dict()
    if candidates is not None:
        hit = sum(len(set(candidates.get(s, ())) & gold.get(s, set())) for s in ids)
        rep["blocking_recall"] = hit / max(n_gold, 1)
        rep["cands_per_s1"] = sum(len(candidates.get(s, ())) for s in ids) / max(len(ids), 1)
    return rep, per


def score_file(pred_path, split, cand_path=None, save_per=None):
    """Score a prediction file on a saved split; optionally save per-entity scores for paired tests."""
    sp = split_ids(split)
    ids = sp["s1_id"].tolist()
    gold = load_gold(ids)
    pred = read_id_lists(pred_path)
    cands = read_id_lists(cand_path) if cand_path else None
    rep, per = score(pred, gold, ids, dict(zip(sp["s1_id"], sp["country"])), cands)
    if save_per:
        pd.DataFrame({"s1_id": ids, "f05": per}).to_parquet(save_per, index=False)
    return rep


def error_breakdown(exp_dir, split="dev"):
    """Where the lost score goes on a split: per-entity loss (1 - F0.5) summed by cause, from an experiment
    folder with <split>_matching.tsv and <split>_probs.parquet (pairs scored by the stacker)."""
    sp = split_ids(split)
    ids = sp["s1_id"].tolist()
    gold = load_gold(ids)
    pred = read_id_lists(os.path.join(exp_dir, f"{split}_matching.tsv"))
    cands = read_id_lists(os.path.join("artefacts/splits", f"{split}_candidates.tsv"))
    scored = pd.read_parquet(os.path.join(exp_dir, f"{split}_probs.parquet"))
    scored_set = scored.groupby("s1_id")["pool_id"].agg(set).to_dict()
    rows = []
    for s in ids:
        g, p = gold.get(s, set()), set(pred.get(s, ()))
        loss = 1 - f05(p, g)
        if loss == 0:
            continue
        fp, fn = p - g, g - p
        fn_block = fn - set(cands.get(s, ()))
        fn_filter = (fn - fn_block) - scored_set.get(s, set())
        cause = ("fp_on_singleton" if not g else "empty_pred_on_match" if not p else
                 "fp_and_fn" if fp and fn else "fp_only" if fp else "fn_only")
        rows.append((s, loss, cause, len(g), len(p), len(fp), len(fn), len(fn_block), len(fn_filter)))
    d = pd.DataFrame(rows, columns=["s1_id", "loss", "cause", "n_gold", "n_pred", "fp", "fn", "fn_blocking",
                                    "fn_prob_filter"])
    tot = len(ids)
    print(f"{split}: {tot:,} S1, total loss {d['loss'].sum():.1f} entity-points = {d['loss'].sum() / tot:.5f} of F0.5")
    g = d.groupby("cause").agg(S1=("loss", "size"), loss=("loss", "sum"), fp=("fp", "sum"), fn=("fn", "sum"),
                               fn_blocking=("fn_blocking", "sum"), fn_prob_filter=("fn_prob_filter", "sum"))
    g["share_of_loss"] = g["loss"] / d["loss"].sum()
    print(g.sort_values("loss", ascending=False).round(3).to_string())
    blk = d[d["fn_blocking"] > 0]
    print(f"entities with a blocking miss: {len(blk):,}, their loss {blk['loss'].sum():.1f}; "
          f"with a prob<0.001 filter miss: {int((d['fn_prob_filter'] > 0).sum()):,}")
    d.to_parquet(os.path.join(exp_dir, f"{split}_errors.parquet"), index=False)
    return d


def main():
    ap = argparse.ArgumentParser()
    if len(sys.argv) > 1 and sys.argv[1] == "errors":  # python -m src.local_score errors <exp_dir>
        error_breakdown(sys.argv[2])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "compare":  # python -m src.local_score compare <expA> <expB> <split>
        a_dir, b_dir, split = sys.argv[2:5]
        if split == "lockbox":
            print("LOCKBOX touch (submission gate)")
        reps = []
        for d in (a_dir, b_dir):
            reps.append(score_file(os.path.join(d, f"{split}_matching.tsv"), split,
                                   save_per=os.path.join(d, f"{split}_per_entity.parquet")))
        a = pd.read_parquet(os.path.join(a_dir, f"{split}_per_entity.parquet"))["f05"].to_numpy()
        b = pd.read_parquet(os.path.join(b_dir, f"{split}_per_entity.parquet"))["f05"].to_numpy()
        lo, hi = paired_bootstrap(a, b)
        print(json.dumps({"split": split, "A": a_dir, "B": b_dir, "A_f05": reps[0]["macro_f05"], "A_ci95": reps[0]["ci95"],
                          "A_by_country": reps[0]["by_country"], "B_f05": reps[1]["macro_f05"], "B_ci95": reps[1]["ci95"],
                          "B_by_country": reps[1]["by_country"], "delta": float((b - a).mean()),
                          "delta_ci95": [lo, hi]}, indent=1))
        return
    ap.add_argument("--pred", required=True, help="matching TSV (.tsv or .tsv.gz), one row per S1")
    ap.add_argument("--split", required=True, choices=["fit", "dev", "lockbox"])
    ap.add_argument("--cands", default=None, help="optional candidate TSV for blocking recall")
    ap.add_argument("--save_per", default=None, help="parquet of per-entity scores (for paired bootstrap)")
    ap.add_argument("--lockbox_ok", action="store_true", help="required to score the lockbox (submission gate only)")
    args = ap.parse_args()
    if args.split == "lockbox" and not args.lockbox_ok:
        raise SystemExit("lockbox is scored only at the submission gate: pass --lockbox_ok")
    print(json.dumps(score_file(args.pred, args.split, args.cands, args.save_per), indent=1))


if __name__ == "__main__":
    main()


def fn_prob_profile(exp_dir, split="dev"):
    """Stacker probability of the scored false negatives and false positives (label-side view for error analysis)."""
    sp = split_ids(split)
    gold = load_gold(sp["s1_id"].tolist())
    pred = read_id_lists(os.path.join(exp_dir, f"{split}_matching.tsv"))
    pr = pd.read_parquet(os.path.join(exp_dir, f"{split}_probs.parquet"))
    lab = np.array([p in gold.get(s, ()) for s, p in zip(pr["s1_id"], pr["pool_id"])])
    chosen = np.array([p in set(pred.get(s, ())) for s, p in zip(pr["s1_id"], pr["pool_id"])])
    top = pr.groupby("pool_id")["prob"].transform("max").to_numpy()
    bins = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0001]
    for name, m in (("FN (gold, not chosen)", lab & ~chosen), ("FP (chosen, not gold)", ~lab & chosen)):
        c = pd.cut(pr.loc[m, "prob"], bins).value_counts().sort_index()
        print(f"{name}: {m.sum():,} pairs\n" + c.to_string())
    fn = lab & ~chosen
    lost_o2o = fn & (pr["prob"].to_numpy() < top) & (pr["prob"].to_numpy() >= 0.3)
    print(f"FN with prob >= 0.3 whose pool record went to another dev S1 with higher prob: {lost_o2o.sum():,}")


def rescue_ceiling(new_pairs_path, oof_path, split="fit", alpha=1.5):
    """Oracle gain on a split if every gold pair among the new (rescue) pairs were added to the champion's
    decisions with perfect precision, per key subset. Uses FIT labels only when split == 'fit'."""
    from src.decide import decide
    ids = split_ids(split)["s1_id"].tolist()
    gold = load_gold(ids)
    oof = pd.read_parquet(oof_path)
    base = decide(oof, ids, {"method": "expf", "alpha": alpha, "one_to_one": True})
    b = np.array([f05(base.get(s, ()), gold.get(s, ())) for s in ids])
    p = pd.read_parquet(new_pairs_path)
    p = p[p["s1_id"].isin(set(ids))]
    p = p[[q in gold.get(s, ()) for s, q in zip(p["s1_id"], p["pool_id"])]]
    subsets = {"all keys": p, "k1|k2": p[(p["k1"] > 0) | (p["k2"] > 0)], "k1": p[p["k1"] > 0],
               "hits>=2": p[p["hits"] >= 2], "hits>=3": p[p["hits"] >= 3]}
    print(f"{split}: base macro F0.5 {b.mean():.5f} on {len(ids):,} S1")
    for name, sub in subsets.items():
        add = sub.groupby("s1_id")["pool_id"].agg(list).to_dict()
        new = np.array([f05(list(base.get(s, ())) + add.get(s, []), gold.get(s, ())) for s in ids])
        print(f"  oracle +{name:9s}: {len(sub):,} gold pairs -> {new.mean():.5f} (gain {new.mean() - b.mean():+.5f})")


def added_stats(exp_dir, split="dev"):
    """TP / FP among pairs a rescue experiment added on a split, and how many S1 went from empty to non-empty."""
    ids = split_ids(split)
    country = dict(zip(ids["s1_id"], ids["country"]))
    gold = load_gold(ids["s1_id"].tolist())
    a = pd.read_parquet(os.path.join(exp_dir, f"{split}_added.parquet"))
    a["tp"] = [q in gold.get(s, ()) for s, q in zip(a["s1_id"], a["pool_id"])]
    a["singleton"] = [not gold.get(s) for s in a["s1_id"]]
    a["country"] = a["s1_id"].map(country)
    print(f"{split}: {len(a)} added, TP {int(a['tp'].sum())}, FP {int((~a['tp']).sum())} "
          f"(on singletons {int((~a['tp'] & a['singleton']).sum())})")
    print(a.groupby("country")["tp"].agg(["size", "sum"]).to_string())
    return a
