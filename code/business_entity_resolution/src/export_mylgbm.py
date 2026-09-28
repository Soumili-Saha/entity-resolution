"""Export the er_pipeline v4 LightGBM (stage 1 + expansion + stage 2) as an extra score folder for the v11 stacker.

Rebuilds the exact candidate order of the Colab runs from their caches (no feature computation, no training):
  train: er_validation_v4 cell 4  -> C[realistic_mask] + E, p = v4/p_final.npy   (3-fold OOF, leakage-free)
  test : test_v4 cells 6-9        -> C + E, p = pred1/p_*.npy ++ p1E.npy, stage-2 rows from pred2_v3.npz
then looks up every stacker pair (handoff/lgbm_ranker_v1, lgbm >= 0.001) and writes
  handoff/lgbm_ranker_v2/ce_oof_fold0_part00.parquet  (fold-0 train S1s covered by the 15% selection)
  handoff/lgbm_ranker_v2/ce_test_part00.parquet       (all test S1s)
with columns s1_id, pool_id, ce_prob. A covered S1's pair that the v4 candidates never contained gets MISS_P;
train S1s outside the selection are left out (fill them with --fill mylgbm=lgbm).

  python -m src.export_mylgbm
"""
import glob
import json
import os
import sys
import zlib

import numpy as np
import pandas as pd

MP = "src/pipeline_v2"
WORK = os.environ.get("PIPELINE_V2_WORK", "handoff/pipeline_v2_work")  # work folder written by the pipeline-v2 notebooks
RUN = f"{WORK}/run_frac0.15_seed42"
TEST = f"{WORK}/test_run_v4"
DATA = "dataset"
OUT = "handoff/lgbm_ranker_v2"
MISS_P = 1e-4
sys.path.insert(0, MP)
import er_pipeline as er  # noqa: E402


def blocks(run_dir):
    """run_blocking() with every job cached: concat, sort, dedupe."""
    bdir = os.path.join(run_dir, "blocks")
    C = pd.concat([pd.read_parquet(os.path.join(bdir, f)) for f in sorted(os.listdir(bdir))], ignore_index=True)
    return C.sort_values(["i1", "i2", "pool"]).drop_duplicates(["i1", "i2"]).reset_index(drop=True)


def check_jobs(run_dir):
    parts = json.load(open(f"{run_dir}/partitions.json"))
    n_jobs = len(parts) + len({p.split("|")[0] for p in parts})
    n_files = len(os.listdir(f"{run_dir}/blocks"))
    assert n_files == n_jobs, f"{run_dir}/blocks has {n_files} files, expected {n_jobs} - download incomplete"


def train_scores():
    check_jobs(RUN)
    S = pd.read_parquet(f"{RUN}/selected.parquet", columns=["entity_id"])
    C = blocks(RUN)
    gt = er.load_ground_truth(DATA)
    mask = er.realistic_mask(C, S, gt)
    CE = er.concat_candidates(C[mask].reset_index(drop=True), pd.read_parquet(f"{RUN}/v4/E.parquet"))
    p = np.load(f"{RUN}/v4/p_final.npy")
    assert len(p) == len(CE), f"p_final {len(p):,} vs candidates {len(CE):,}"
    return S.entity_id.values, CE.i1.values, CE.i2.values, p


def test_scores():
    check_jobs(TEST)
    S = pd.read_parquet(f"{TEST}/selected.parquet", columns=["entity_id"])
    C = blocks(TEST)
    n_parts = -(-len(C) // 500_000)
    files = [f"{TEST}/pred1/p_{k:04d}.npy" for k in range(n_parts)]
    missing = [f for f in files if not os.path.exists(f)]
    assert not missing, f"{len(missing)} pred1 files missing, e.g. {missing[0]}"
    p1 = np.concatenate([np.load(f) for f in files])
    assert len(p1) == len(C), f"pred1 {len(p1):,} vs candidates {len(C):,}"
    CE = er.concat_candidates(C, pd.read_parquet(f"{TEST}/E.parquet"))
    p = np.r_[p1, np.load(f"{TEST}/p1E.npy")].astype(np.float32)
    assert len(p) == len(CE), f"p1+p1E {len(p):,} vs candidates {len(CE):,}"
    z = np.load(f"{TEST}/pred2_v3.npz")
    p[z["rows"]] = z["p2"]
    return S.entity_id.values, CE.i1.values, CE.i2.values, p


def stacker_pairs(pattern, fold0):
    files = sorted(glob.glob(pattern))
    names = __import__("pyarrow.parquet", fromlist=["x"]).read_schema(files[0]).names
    col = "lgbm_prob" if "lgbm_prob" in names else "prob"
    cols = ["s1_id", "pool_id", col] + (["fold"] if fold0 and "fold" in names else [])
    df = pd.concat([pd.read_parquet(f, columns=cols, filters=[(col, ">=", 0.001)]) for f in files], ignore_index=True)
    if fold0:
        f0 = df["fold"] == 0 if "fold" in df else df["s1_id"].map(lambda s: zlib.crc32(s.encode()) % 5 == 0)
        df = df[f0]
    return df[["s1_id", "pool_id"]].reset_index(drop=True)


def lookup(pairs, eid, i1, i2, p):
    idx = pd.Index(eid)
    a = idx.get_indexer(pairs["s1_id"].to_numpy())
    b = idx.get_indexer(pairs["pool_id"].to_numpy())
    code = i1.astype(np.int64) * (1 << 26) + i2.astype(np.int64)
    order = np.argsort(code, kind="stable")
    sc = code[order]
    out = np.full(len(pairs), np.nan, np.float32)
    covered = a >= 0
    out[covered] = MISS_P
    q = np.flatnonzero(covered & (b >= 0))
    ct = a[q].astype(np.int64) * (1 << 26) + b[q]
    pos = np.clip(np.searchsorted(sc, ct), 0, len(sc) - 1)
    hit = sc[pos] == ct
    out[q[hit]] = p[order[pos[hit]]]
    res = pairs.assign(ce_prob=out)[covered].reset_index(drop=True)
    print(f"  stacker pairs {len(pairs):,} | S1 covered {int(covered.sum()):,} | found in v4 candidates "
          f"{int(hit.sum()):,} | not found -> {MISS_P} {int(covered.sum() - hit.sum()):,}", flush=True)
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    print("train (OOF) ...", flush=True)
    tr = lookup(stacker_pairs("handoff/lgbm_ranker_v1/oof_train_full_part*.parquet", True), *train_scores())
    tr.to_parquet(f"{OUT}/ce_oof_fold0_part00.parquet", index=False)
    print("test ...", flush=True)
    te = lookup(stacker_pairs("handoff/lgbm_ranker_v1/probs_model_full_part*.parquet", False), *test_scores())
    te.to_parquet(f"{OUT}/ce_test_part00.parquet", index=False)
    print(f"wrote {OUT}: fold-0 {len(tr):,} rows, test {len(te):,} rows")


if __name__ == "__main__":
    main()
