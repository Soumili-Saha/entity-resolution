"""CLI entry point: python -m src.run --stage all --split test

Every stage caches its output under artefacts/<split>/ so a single stage
(e.g. --stage decide) can be rerun without redoing the ones before it.
"""
import argparse
import os
import time

import pandas as pd
import yaml

from src.io_utils import read_tsv
from src.normalize import normalize_frame

STAGES = ["normalize", "embed", "block", "features", "cross_encoder", "rank", "decide"]

KEEP_COLS = ["entity_id", "business_name", "business_address", "country", "source",
             "name_clean", "name_core", "name_skel", "name_ns", "name_alt", "name_alt_skel",
             "addr_clean", "postal", "nums", "landmarks", "country_norm"]


def parse_args():
    """Parse --stage, --split and --config command-line options."""
    parser = argparse.ArgumentParser(description="Business Entity Resolution pipeline")
    parser.add_argument("--stage", default="all", choices=["all", *STAGES])
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def model_dir(cfg):
    """Folder holding the trained matcher (paths.model_dir, default artefacts/model)."""
    return cfg["paths"].get("model_dir") or os.path.join(cfg["paths"]["artefacts_dir"], "model")


def probs_path(cfg):
    """Test probabilities file, tagged by model folder so several model versions can coexist."""
    return os.path.join(art_dir(cfg, "test"), f"probs_{os.path.basename(os.path.normpath(model_dir(cfg)))}.parquet")


def art_dir(cfg, split):
    """artefacts/<split>/, created on demand."""
    d = os.path.join(cfg["paths"]["artefacts_dir"], split)
    os.makedirs(d, exist_ok=True)
    return d


def stage_normalize(cfg, split, chunk_rows=1_000_000):
    """Clean each source TSV in row chunks (bounded memory) into artefacts/<split>/s{k}.parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    t0 = time.time()
    d = art_dir(cfg, split)
    for k in (1, 2, 3):
        src = os.path.join(cfg["paths"]["data_dir"], split, f"{split}_source{k}.tsv")
        raw = read_tsv(src)
        raw["source"] = k
        writer = None
        for start in range(0, len(raw), chunk_rows):
            out = normalize_frame(raw.iloc[start:start + chunk_rows])[KEEP_COLS]
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(os.path.join(d, f"s{k}.parquet"), table.schema)
            writer.write_table(table)
        writer.close()
        print(f"[normalize:{split}] source{k}: {len(raw):,} rows [{time.time() - t0:.0f}s]", flush=True)
        del raw


def load_normalized(cfg, split, columns=None):
    """Load cached normalised frames: (s1, pool) where pool = Source 2 + Source 3."""
    d = art_dir(cfg, split)
    s1 = pd.read_parquet(os.path.join(d, "s1.parquet"), columns=columns)
    pool = pd.concat([pd.read_parquet(os.path.join(d, f"s{k}.parquet"), columns=columns) for k in (2, 3)],
                     ignore_index=True)
    return s1, pool


def stage_embed(cfg, split):
    """Encode S1 and pool records with the blocking encoder; cache emb_s1.npy / emb_pool.npy."""
    import numpy as np
    from src.embed import encode, record_text
    d = art_dir(cfg, split)
    model = cfg["blocking"].get("embed_model_path") or cfg["blocking"]["embed_model"]
    s1, pool = load_normalized(cfg, split, columns=["name_clean", "addr_clean"])
    for name, df in (("s1", s1), ("pool", pool)):
        path = os.path.join(d, f"emb_{name}.npy")
        if os.path.exists(path):
            print(f"[embed:{split}] {name}: cached, skipping", flush=True)
            continue
        t0 = time.time()
        tmp = path.replace(".npy", ".tmp.npy")
        emb = encode(record_text(df), model_name=model, out_path=tmp)
        shape = emb.shape
        del emb
        os.replace(tmp, path)
        print(f"[embed:{split}] {name} {shape} [{time.time() - t0:.0f}s]", flush=True)


NORM_COLS = ["entity_id", "source", "name_clean", "name_core", "name_skel", "name_ns", "name_alt",
             "name_alt_skel", "addr_clean", "postal", "nums", "landmarks", "country_norm"]


def query_index(cfg, split, s1):
    """S1 rows the matcher is trained / scored on: all of test, a fixed random sample of train.
    (Blocking always runs on ALL S1 so candidate-competition context matches test.)"""
    import numpy as np
    if split == "test":
        return np.arange(len(s1))
    n = min(cfg["train"]["sample_s1"], len(s1))
    return np.sort(np.random.RandomState(cfg["seed"]).choice(len(s1), n, replace=False))


def stage_block(cfg, split):
    """Token + embedding blocking for the query S1s against the full pool -> cands.parquet."""
    import numpy as np
    from src.blocking import block_all
    t0 = time.time()
    d = art_dir(cfg, split)
    b = cfg["blocking"]
    s1, pool = load_normalized(cfg, split, columns=["entity_id", "name_skel", "name_alt_skel", "name_ns",
                                                    "addr_clean", "country_norm"])
    qi = np.arange(len(s1))  # block every S1; the train sample is selected in stage_features
    e_s1 = e_pool = None
    emb_path = os.path.join(d, "emb_pool.npy")
    if os.path.exists(emb_path):
        e_s1 = np.load(os.path.join(d, "emb_s1.npy"), mmap_mode="r")
        e_pool = np.load(emb_path, mmap_mode="r")
    else:
        print(f"[block:{split}] no embeddings cached - token pass only", flush=True)
    cands = block_all(s1, pool, e_s1, e_pool, token_k=b["token_top_k"], token_max_df=b["token_max_df"],
                      embed_k=b["embed_top_k"], cap=b["max_candidates"])
    cands.to_parquet(os.path.join(d, "cands.parquet"), index=False)  # integer indices only
    sample = query_index(cfg, split, s1)
    pd.DataFrame({"s1_idx": sample}).to_parquet(os.path.join(d, "query_idx.parquet"), index=False)
    print(f"[block:{split}] {len(cands):,} candidates, {len(cands) / len(qi):.1f} per S1 [{time.time() - t0:.0f}s]")
    if split == "train":
        from src.evaluate import blocking_recall, oracle_f05
        from src.io_utils import load_ground_truth
        gold = load_ground_truth(cfg["paths"]["data_dir"])
        sub = cands[np.isin(cands["s1_idx"].to_numpy(), sample)]
        s1_ids, pool_ids = s1["entity_id"].to_numpy(), pool["entity_id"].to_numpy()
        cs = {}
        for a, p in zip(s1_ids[sub["s1_idx"].to_numpy()], pool_ids[sub["pool_idx"].to_numpy()]):
            cs.setdefault(a, []).append(p)
        ids = s1_ids[sample].tolist()
        for col in ("tok_rank", "emb_rank"):
            if col in sub:
                cs_p = {}
                m = sub[col].to_numpy() < 999
                for a, p in zip(s1_ids[sub["s1_idx"].to_numpy()[m]], pool_ids[sub["pool_idx"].to_numpy()[m]]):
                    cs_p.setdefault(a, []).append(p)
                print(f"[block:train] {col[:3]} pass alone: recall {blocking_recall(cs_p, gold, ids):.4f}")
        print(f"[block:train] union: recall {blocking_recall(cs, gold, ids):.4f} | oracle F0.5 {oracle_f05(cs, gold, ids):.4f}")


def stage_features(cfg, split):
    """Pair features for all candidates, written in S1 chunks to artefacts/<split>/feats/part_*.parquet."""
    import numpy as np
    from src.features import build_pair_features, global_context
    t0 = time.time()
    d = art_dir(cfg, split)
    fd = os.path.join(d, "feats")
    os.makedirs(fd, exist_ok=True)
    for old in os.listdir(fd):
        os.remove(os.path.join(fd, old))
    cands = pd.read_parquet(os.path.join(d, "cands.parquet"))
    ctx = global_context(cands)  # competition context over ALL S1s, as at test time
    if split == "train":  # training sample size comes from the config (train.sample_s1)
        ids = pd.read_parquet(os.path.join(d, "s1.parquet"), columns=["entity_id"])
        sample = query_index(cfg, split, ids)
        pd.DataFrame({"s1_idx": sample}).to_parquet(os.path.join(d, "query_idx.parquet"), index=False)
    else:
        sample = pd.read_parquet(os.path.join(d, "query_idx.parquet"))["s1_idx"].to_numpy()
    if len(sample) < cands["s1_idx"].nunique():
        keep = np.isin(cands["s1_idx"].to_numpy(), sample)
        cands, ctx = cands.loc[keep].reset_index(drop=True), ctx.loc[keep].reset_index(drop=True)
    s1, pool = load_normalized(cfg, split, columns=NORM_COLS)
    uniq = np.unique(cands["s1_idx"].to_numpy())
    step = cfg["train"]["feature_chunk_s1"]
    s1_idx = cands["s1_idx"].to_numpy()
    for n, start in enumerate(range(0, len(uniq), step)):
        lo, hi = uniq[start], uniq[min(start + step, len(uniq)) - 1]
        m = (s1_idx >= lo) & (s1_idx <= hi)
        part = cands.loc[m]
        f = build_pair_features(part, s1, pool).reset_index(drop=True)
        f = pd.concat([f, ctx.loc[m].reset_index(drop=True)], axis=1).astype(np.float32)
        f.insert(0, "s1_id", s1["entity_id"].to_numpy()[part["s1_idx"].to_numpy()])
        f.insert(1, "pool_id", pool["entity_id"].to_numpy()[part["pool_idx"].to_numpy()])
        f.to_parquet(os.path.join(fd, f"part_{n:03d}.parquet"), index=False)
        print(f"[features:{split}] part {n}: {len(f):,} rows, {f.shape[1] - 2} features [{time.time() - t0:.0f}s]", flush=True)


def load_features(cfg, split):
    """Concatenate cached feature parts."""
    fd = os.path.join(art_dir(cfg, split), "feats")
    return pd.concat([pd.read_parquet(os.path.join(fd, p)) for p in sorted(os.listdir(fd))], ignore_index=True)


def load_feature_matrix(cfg, split, s1_index=None, pool_index=None):
    """Cached feature parts -> (float32 matrix, id frame, feature names), filling a preallocated array
    part by part so peak memory stays near one copy of the features. With s1_index / pool_index
    (pd.Index of entity ids) the id frame holds integer s1_idx / pool_idx instead of strings."""
    import numpy as np
    import pyarrow.parquet as pq
    from src.ranker import ID_COLS
    fd = os.path.join(art_dir(cfg, split), "feats")
    parts = [os.path.join(fd, p) for p in sorted(os.listdir(fd))]
    n = sum(pq.ParquetFile(p).metadata.num_rows for p in parts)
    feat_cols = [c for c in pq.ParquetFile(parts[0]).schema_arrow.names if c not in ID_COLS]
    X = np.empty((n, len(feat_cols)), dtype=np.float32)
    ids, off = [], 0
    for p in parts:
        t = pd.read_parquet(p)
        X[off:off + len(t)] = t[feat_cols].to_numpy(np.float32)
        if s1_index is not None:
            ids.append(pd.DataFrame({"s1_id": s1_index.get_indexer(t["s1_id"]).astype(np.int32),
                                     "pool_id": pool_index.get_indexer(t["pool_id"]).astype(np.int32)}))
        else:
            ids.append(t[ID_COLS])
        off += len(t)
        del t
    return X, pd.concat(ids, ignore_index=True), feat_cols


def gold_index_pairs(cfg, s1_index, pool_index):
    """Ground-truth pairs as integer arrays (s1_idx, pool_idx) - compact, no Python sets."""
    import numpy as np
    from src.io_utils import read_tsv
    gt = read_tsv(os.path.join(cfg["paths"]["data_dir"], "train", "train_ground_truth.tsv"))
    ex = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")
    ex = ex[ex["m"].fillna("") != ""]
    a = s1_index.get_indexer(ex["source1_entity_id"]).astype(np.int64)
    b = pool_index.get_indexer(ex["m"]).astype(np.int64)
    ok = (a >= 0) & (b >= 0)
    return a[ok], b[ok]


def stage_rank(cfg, split):
    """train: grouped OOF LightGBM + decision tuning + final model. test: predict probabilities."""
    import json
    from src.ranker import predict_lgbm, train_lgbm
    d = art_dir(cfg, split)
    md = model_dir(cfg)
    os.makedirs(md, exist_ok=True)
    if split == "train":
        import numpy as np
        from src.decide import tune
        from src.ranker import stage2_matrix
        # integer ids everywhere: string ids / Python sets for 10M+ rows do not fit in 16 GB
        s1_ent = pd.read_parquet(os.path.join(d, "s1.parquet"), columns=["entity_id"])["entity_id"]
        pool_ent = pd.concat([pd.read_parquet(os.path.join(d, f"s{k}.parquet"), columns=["entity_id"])["entity_id"]
                              for k in (2, 3)], ignore_index=True)
        s1_index, pool_index = pd.Index(s1_ent), pd.Index(pool_ent)
        n_pool = len(pool_index)
        X, df, feat_cols = load_feature_matrix(cfg, "train", s1_index, pool_index)
        ga, gb = gold_index_pairs(cfg, s1_index, pool_index)
        q = pd.read_parquet(os.path.join(d, "query_idx.parquet"))["s1_idx"].to_numpy()
        s1_ids = q.tolist()
        in_q = np.isin(ga, q)
        gold = {}
        for a_, b_ in zip(ga[in_q], gb[in_q]):
            gold.setdefault(int(a_), set()).add(int(b_))
        key = df["s1_id"].to_numpy(np.int64) * n_pool + df["pool_id"].to_numpy(np.int64)
        y = np.isin(key, ga * n_pool + gb).astype(np.int8)
        del key, ga, gb, pool_index, pool_ent
        from src.evaluate import fold_of
        n_folds = cfg["validation"]["n_folds"]
        fold_by_s1 = np.zeros(len(s1_ent), dtype=np.int8)
        fold_by_s1[q] = [fold_of(x, n_folds) for x in s1_ent.to_numpy()[q]]
        fold = fold_by_s1[df["s1_id"].to_numpy()]
        print(f"[rank:train] {len(df):,} pairs, {int(y.sum()):,} positives, {len(q):,} S1", flush=True)
        # decision tuning is a pure-Python grid: on very large samples tune on a random S1 subset
        tune_max = cfg.get("decide", {}).get("tune_max_s1") or len(s1_ids)
        tune_ids = s1_ids if len(s1_ids) <= tune_max else             np.random.RandomState(cfg["seed"]).choice(np.array(s1_ids), tune_max, replace=False).tolist()
        models, oof, feat_cols = train_lgbm(X, y, fold, feat_cols, cfg)
        df["prob"] = oof
        df.to_parquet(os.path.join(d, "oof.parquet"), index=False)
        tdf = df if len(tune_ids) == len(s1_ids) else df[df["s1_id"].isin(set(tune_ids))]
        params, score = tune(tdf, gold, tune_ids)
        print(f"[rank:train] stage1 OOF macro F0.5 = {score:.4f} on {len(tune_ids):,} S1", flush=True)
        for i, m in enumerate(models):
            m.save_model(os.path.join(md, f"lgbm_{i}.txt"))
        mc = {"features": feat_cols, "decision": params, "oof_f05": score, "n_models": len(models), "stage2": False}
        with open(os.path.join(md, "config.json"), "w") as f:
            json.dump(mc, f, indent=2)
        del models
        # stage 2: add per-S1 probability context from the OOF stage-1 predictions
        X2, ctx_cols = stage2_matrix(X, df, oof)
        del X
        feat_cols2 = feat_cols + ctx_cols
        models2, oof2, _ = train_lgbm(X2, y, fold, feat_cols2, cfg)
        del X2
        df["prob"] = oof2
        tdf = df if len(tune_ids) == len(s1_ids) else df[df["s1_id"].isin(set(tune_ids))]
        params2, score2 = tune(tdf, gold, tune_ids)
        print(f"[rank:train] stage2 OOF macro F0.5 = {score2:.4f}", flush=True)
        if score2 > score + 0.0005:
            for i, m in enumerate(models2):
                m.save_model(os.path.join(md, f"lgbm2_{i}.txt"))
            mc.update({"stage2": True, "features2": feat_cols2, "decision": params2, "oof_f05": score2})
            df.to_parquet(os.path.join(d, "oof.parquet"), index=False)
        with open(os.path.join(md, "config.json"), "w") as f:
            json.dump(mc, f, indent=2)
    else:
        with open(os.path.join(md, "config.json")) as f:
            mc = json.load(f)
        from src.ranker import prob_context
        fd = os.path.join(d, "feats")
        out = []
        for p in sorted(os.listdir(fd)):
            feats = pd.read_parquet(os.path.join(fd, p))  # parts hold whole S1 groups
            df = feats[["s1_id", "pool_id"]].copy()
            df["prob"] = predict_lgbm(feats, md, mc)
            if mc.get("stage2"):
                f2 = pd.concat([feats.reset_index(drop=True), prob_context(feats, df["prob"].to_numpy())], axis=1)
                df["prob"] = predict_lgbm(f2, md, mc, stage=2)
            out.append(df)
            print(f"[rank:test] {p} scored", flush=True)
        pd.concat(out, ignore_index=True).to_parquet(probs_path(cfg), index=False)


def stage_decide(cfg, split):
    """test: probabilities -> matches; write matching_results.tsv + candidate_pairs.tsv and validate."""
    import json

    from src.decide import decide
    if split != "test":
        print("[decide] train decisions are tuned and scored inside --stage rank")
        return
    d = art_dir(cfg, "test")
    md = model_dir(cfg)
    with open(os.path.join(md, "config.json")) as f:
        params = json.load(f)["decision"]
    s1_ids = pd.read_parquet(os.path.join(d, "s1.parquet"), columns=["entity_id"])["entity_id"].tolist()
    probs = pd.read_parquet(probs_path(cfg))
    write_submission(cfg, s1_ids, decide(probs, s1_ids, params), cfg["paths"]["output_dir"])


def write_submission(cfg, s1_ids, matches, out, extra_cands=None):
    """Write matching_results.tsv + candidate_pairs.tsv to `out` (matches restricted to the test
    candidates) and run the official validator. extra_cands: optional DataFrame (s1_id, pool_id) of
    additional candidates, e.g. the scored pairs of a run that re-blocked the test set elsewhere."""
    import subprocess
    import sys

    import numpy as np
    from src.io_utils import write_tsv
    d = art_dir(cfg, "test")
    cands = pd.read_parquet(os.path.join(d, "cands.parquet"), columns=["s1_idx", "pool_idx"])
    pool_ids = np.concatenate([pd.read_parquet(os.path.join(d, f"s{k}.parquet"), columns=["entity_id"])["entity_id"].to_numpy()
                               for k in (2, 3)])
    s1_arr = np.asarray(s1_ids)
    cand_lists = {}
    for a, p in zip(s1_arr[cands["s1_idx"].to_numpy()], pool_ids[cands["pool_idx"].to_numpy()]):
        cand_lists.setdefault(a, []).append(p)
    if extra_cands is not None:
        for a, p in zip(extra_cands["s1_id"].to_numpy(), extra_cands["pool_id"].to_numpy()):
            cand_lists.setdefault(a, []).append(p)  # write_tsv de-duplicates
    # the matcher can only choose candidates: enforce matches subset of candidates
    for s_id, ms in matches.items():
        if ms:
            allowed = set(cand_lists.get(s_id, ()))
            matches[s_id] = [m for m in ms if m in allowed]
    write_tsv(s1_ids, matches, os.path.join(out, "matching_results.tsv"), "matched_entity_ids")
    write_tsv(s1_ids, cand_lists, os.path.join(out, "candidate_pairs.tsv"), "candidate_entity_ids")
    n_empty = sum(1 for s in s1_ids if not matches.get(s))
    print(f"[export] {len(s1_ids):,} S1 rows, {n_empty / len(s1_ids):.1%} empty, "
          f"{sum(len(v) for v in matches.values()) / len(s1_ids):.2f} matches per S1", flush=True)
    subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", os.path.join(out, "matching_results.tsv"),
                    "--candidate", os.path.join(out, "candidate_pairs.tsv"), "--test-dir",
                    os.path.join(cfg["paths"]["data_dir"], "test")])


def main():
    """Run the requested pipeline stage(s) for the requested split."""
    args = parse_args()
    if os.name == "nt":  # keep all cores when the session is locked (Windows EcoQoS throttling)
        from src.no_throttle import disable_throttling
        disable_throttling()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    stages = STAGES if args.stage == "all" else [args.stage]
    os.makedirs(model_dir(cfg), exist_ok=True)
    for st in stages:
        if st == "normalize":
            stage_normalize(cfg, args.split)
        elif st == "embed":
            stage_embed(cfg, args.split)
        elif st == "block":
            stage_block(cfg, args.split)
        elif st == "features":
            stage_features(cfg, args.split)
        elif st == "rank":
            stage_rank(cfg, args.split)
        elif st == "decide":
            stage_decide(cfg, args.split)
        elif st == "cross_encoder":
            print("[cross_encoder] not used yet - skipped")
        else:
            raise ValueError(f"unknown stage {st}")


if __name__ == "__main__":
    main()
