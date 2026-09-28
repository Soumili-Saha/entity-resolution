"""xlmr_large_ce_v2: continue xlmr_large_ce_v1 on the full-data (all 2.2M train S1) LightGBM candidates.

Recipe (handoff/xlmr_large_ce_v2/REPORT.md):
  init   artefacts/ce_large/final (xlmr_large_ce_v1), lr 5e-6, bs 128, 1 epoch, bf16, max_len 128
  train  lgbm_ranker_v1 OOF rows with fold != 0: all positives + negatives with prob >= 0.001 + 5 % of the rest,
         capped at 4M
  fold0  lgbm_ranker_v1 fold-0 rows (prob >= 0.001) UNION ce_training_pairs fold-0 rows (lgbm_prob >= 0.001)
  test   lgbm_ranker_v1 test pairs (prob >= 0.001) UNION ce_training_pairs test pairs

  python -m src.ce_large_v2                       # full run (~2 h training on the AWS g5.2xlarge)
  python -m src.ce_large_v2 --smoke 20000         # quick end-to-end check on 20k pairs per stage
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.ce_rescore import attach_text, load_texts, log, metrics, predict, read_pairs, train, write_parts
from src.io_utils import read_tsv

KEY = ["s1_id", "pool_id"]


def with_labels(df, data_dir):
    """Add label (0/1) from train_ground_truth.tsv when the pair file has none."""
    if "label" in df.columns:
        return df
    gt = read_tsv(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
    gt = gt.assign(pool_id=gt["matched_entity_ids"].str.split(",")).explode("pool_id")
    gt = gt[gt["pool_id"].fillna("") != ""].rename(columns={"source1_entity_id": "s1_id"})[KEY]
    gt["label"] = np.int8(1)
    return df.merge(gt, on=KEY, how="left").fillna({"label": 0}).astype({"label": "int8"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="artefacts/ce_large/final", help="xlmr_large_ce_v1 checkpoint")
    ap.add_argument("--ckpt", default="artefacts/ce_large_v2")
    ap.add_argument("--out", default="handoff/xlmr_large_ce_v2")
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--infer_bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--thr", type=float, default=0.001)
    ap.add_argument("--rest_frac", type=float, default=0.05)
    ap.add_argument("--cap", type=int, default=4_000_000)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", type=int, default=0, help="cap every stage at N pairs for a quick check (0 = full run)")
    args = ap.parse_args()
    args.model = args.init
    if args.smoke:
        args.ckpt += "_smoke"; args.out = os.path.join("artefacts", "smoke_" + os.path.basename(args.out))
    os.makedirs(args.ckpt, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    cap = lambda df: df.iloc[:args.smoke] if args.smoke else df

    oof = read_pairs("handoff/lgbm_ranker_v1/oof_train_full_part*.parquet")
    nz = oof[oof["fold"] != 0]
    nz = with_labels(nz[KEY + ["prob"]], args.data_dir)
    core = (nz["label"].values == 1) | (nz["prob"].values >= args.thr)
    fill = ~core & (rng.rand(len(nz)) < args.rest_frac)
    tr = nz[core | fill]
    if len(tr) > args.cap:
        tr = tr.iloc[rng.choice(len(tr), args.cap, replace=False)]
    log(f"train sample: core {int(core.sum()):,}, random fill {int(fill.sum()):,} -> using {len(tr):,}")
    tr = cap(tr[KEY + ["label"]].sample(frac=1.0, random_state=args.seed))

    texts = load_texts(args.data_dir, "train")
    tok = AutoTokenizer.from_pretrained(args.init)
    final = os.path.join(args.ckpt, "final")
    if os.path.exists(os.path.join(final, "config.json")):
        model = AutoModelForSequenceClassification.from_pretrained(final).cuda()
    else:
        model = train(args, tok, texts, tr=tr)
    del tr, nz

    h = read_pairs("handoff/ce_training_pairs/train_pairs_part*.parquet")
    h0 = h[(h["fold"] == 0) & (h["lgbm_prob"] >= args.thr)][KEY + ["label"]]
    s0 = with_labels(oof[(oof["fold"] == 0) & (oof["prob"] >= args.thr)][KEY + ["prob"]], args.data_dir)
    log(f"fold-0 list: step-3 {len(s0):,} + handoff {len(h0):,}")
    f0 = cap(pd.concat([s0[KEY + ["label"]], h0]).drop_duplicates(KEY).reset_index(drop=True))
    f0["ce_prob"] = predict(model, tok, *attach_text(f0, texts), args)
    s0 = s0.merge(f0[KEY + ["ce_prob"]], on=KEY)
    m = {"n_union": int(len(f0)), "n_step3_rows": int(len(s0)), "pos_step3_rows": int(s0["label"].sum()),
         "ce_full_on_step3_rows": metrics(s0["label"].values, s0["ce_prob"].values),
         "step3_lgbm_on_step3_rows": metrics(s0["label"].values, s0["prob"].values)}
    json.dump(m, open(os.path.join(args.out, "fold0_metrics.json"), "w"), indent=1)
    log(f"fold-0 metrics {m}")
    write_parts(f0[KEY + ["ce_prob"]], args.out, "ce_oof_fold0")
    del texts, h, oof

    texts = load_texts(args.data_dir, "test")
    s3 = read_pairs("handoff/lgbm_ranker_v1/probs_model_full_part*.parquet", columns=KEY + ["prob"])
    ht = read_pairs("handoff/ce_training_pairs/test_pairs_part*.parquet", columns=KEY)
    te = pd.concat([s3[s3["prob"] >= args.thr][KEY], ht]).drop_duplicates().reset_index(drop=True)
    log(f"test list: step-3 {int((s3['prob'] >= args.thr).sum()):,} + handoff {len(ht):,} -> union {len(te):,}")
    te = cap(te)
    te["ce_prob"] = predict(model, tok, *attach_text(te, texts), args)
    write_parts(te, args.out, "ce_test")
    json.dump(json.load(open(os.path.join(args.ckpt, "train_info.json"))), open(os.path.join(args.out, "train_info.json"), "w"), indent=1)
    log("done")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
