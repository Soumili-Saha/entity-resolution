"""xlmr_large_ce_v3_unseen: transductive self-training of xlmr_large_ce_v1 on test S1s of countries absent
from training. Pseudo-labels come only from our own models on the provided test candidates (no external data,
no test labels).

Recipe (handoff/xlmr_large_ce_v3_unseen/REPORT.md, pseudo_stats.json):
  pairs      ce_training_pairs test pairs of unseen-country S1s (2,102,556 in the submitted data)
  teachers   LightGBM (lgbm_prob), xlmr_base_ce_v1, xlmr_large_ce_v1
  positive   all three teachers >= --pos_thr, at most --max_pos (600k) sampled
  decoy neg  one-to-one decoys: the pool record is a pseudo-positive of ANOTHER S1 and this S1 scores it
             >= --decoy_min on at least one teacher
  easy neg   all remaining pairs where all three teachers are <= --neg_thr
  Default thresholds reproduce the reported counts on the submitted data within 0.2 %: positive candidates
  746,848 (report 746,968), decoys 56,686 (56,551), easy negatives 745,172 (745,038).
  replay     the same number of labelled fold != 0 train pairs (ce_training_pairs), so India / US are kept
  training   continue artefacts/ce_large/final, lr 5e-6, bs 128, 1 epoch, bf16, max_len 128
  outputs    ce_oof_fold0 (labelled fold 0, no-harm check) and ce_test_unseen (all unseen-country test pairs)

  python -m src.ce_large_v3_unseen                  # full run on the AWS g5.2xlarge
  python -m src.ce_large_v3_unseen --smoke 20000    # quick end-to-end check on 20k pairs per stage
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.ce_rescore import attach_text, load_texts, log, metrics, predict, read_pairs, train, write_parts
from src.unseen_pairs import unseen_pairs, unseen_s1

KEY = ["s1_id", "pool_id"]


def pseudo_labels(uns, args, rng):
    """(pseudo-labelled unseen-country pairs, stats)."""
    h = read_pairs("handoff/ce_training_pairs/test_pairs_part*.parquet", columns=KEY + ["lgbm_prob"])
    fr = h[h["s1_id"].isin(uns)]
    for name, d in (("base", "handoff/xlmr_base_ce_v1"), ("large", "handoff/xlmr_large_ce_v1")):
        ce = read_pairs(os.path.join(d, "ce_test_part*.parquet")).drop_duplicates(KEY).rename(columns={"ce_prob": name})
        fr = fr.merge(ce, on=KEY, how="left")
    fr = fr.fillna({"base": fr["lgbm_prob"], "large": fr["lgbm_prob"]})
    s = fr[["lgbm_prob", "base", "large"]]
    lo, hi = s.min(axis=1), s.max(axis=1)

    pos_c = fr[lo >= args.pos_thr]
    owner = pos_c.drop_duplicates("pool_id").set_index("pool_id")["s1_id"]
    other = fr["pool_id"].isin(owner.index) & (fr["s1_id"] != fr["pool_id"].map(owner))
    decoy = fr[other & (hi >= args.decoy_min)]
    easy = fr[(hi <= args.neg_thr) & (lo < args.pos_thr) & ~other]
    pos = pos_c.sample(min(len(pos_c), args.max_pos), random_state=args.seed)
    if args.max_easy:
        easy = easy.sample(min(len(easy), args.max_easy), random_state=args.seed)
    ps = pd.concat([pos[KEY].assign(label=1), decoy[KEY].assign(label=0), easy[KEY].assign(label=0)])
    stats = {"unseen_s1": len(uns), "unseen_pairs": int(len(fr)), "pseudo_pos": int(len(pos)),
             "pseudo_decoy_neg": int(len(decoy)), "pseudo_rand_neg": int(len(easy)),
             "pos_candidates": int(len(pos_c)), "decoy_candidates": int(len(decoy)),
             "pos_thr": args.pos_thr, "decoy_min": args.decoy_min, "neg_thr": args.neg_thr}
    return ps.astype({"label": "int8"}), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="artefacts/ce_large/final", help="xlmr_large_ce_v1 checkpoint")
    ap.add_argument("--ckpt", default="artefacts/ce_large_v3_unseen")
    ap.add_argument("--out", default="handoff/xlmr_large_ce_v3_unseen")
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--pos_thr", type=float, default=0.93)
    ap.add_argument("--decoy_min", type=float, default=0.065)
    ap.add_argument("--neg_thr", type=float, default=0.22)
    ap.add_argument("--max_pos", type=int, default=600_000)
    ap.add_argument("--max_easy", type=int, default=0, help="cap easy negatives (0 = all)")
    ap.add_argument("--thr", type=float, default=0.001, help="replay: hard train pairs are lgbm_prob >= thr")
    ap.add_argument("--rest_frac", type=float, default=0.10)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--infer_bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=128)
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

    uns = unseen_s1(args.data_dir)
    ps, stats = pseudo_labels(uns, args, rng)
    log(f"pseudo-labels {stats}")
    tr = read_pairs("handoff/ce_training_pairs/train_pairs_part*.parquet")
    tr = tr[tr["fold"] != 0]
    tr = tr[(tr["lgbm_prob"] >= args.thr) | (rng.rand(len(tr)) < args.rest_frac)]
    tr = tr.iloc[rng.choice(len(tr), min(len(tr), len(ps)), replace=False)][KEY + ["label"]]
    stats.update({"replay_pairs": int(len(tr)), "train_pairs": int(len(ps) + len(tr))})
    mix = pd.concat([ps.assign(s1_id="te:" + ps["s1_id"], pool_id="te:" + ps["pool_id"]),
                     tr.assign(s1_id="tr:" + tr["s1_id"], pool_id="tr:" + tr["pool_id"])], ignore_index=True)
    mix = cap(mix.sample(frac=1.0, random_state=args.seed))
    json.dump(stats, open(os.path.join(args.out, "pseudo_stats.json"), "w"), indent=1)

    t_tr, t_te = load_texts(args.data_dir, "train"), load_texts(args.data_dir, "test")
    texts = pd.concat([pd.Series(t_tr.values, index="tr:" + t_tr.index), pd.Series(t_te.values, index="te:" + t_te.index)])
    tok = AutoTokenizer.from_pretrained(args.init)
    final = os.path.join(args.ckpt, "final")
    if os.path.exists(os.path.join(final, "config.json")):
        model = AutoModelForSequenceClassification.from_pretrained(final).cuda()
    else:
        model = train(args, tok, texts, tr=mix)
    del texts, mix

    f0 = read_pairs("handoff/ce_training_pairs/train_pairs_part*.parquet")
    f0 = cap(f0[(f0["fold"] == 0) & (f0["lgbm_prob"] >= args.thr)].reset_index(drop=True))
    f0["ce_prob"] = predict(model, tok, *attach_text(f0, t_tr), args)
    m = {"n": int(len(f0)), "ce_v3_unseen": metrics(f0["label"].values, f0["ce_prob"].values)}
    json.dump(m, open(os.path.join(args.out, "fold0_metrics.json"), "w"), indent=1)
    log(f"fold-0 no-harm check (should not degrade vs xlmr_large_ce_v1) {m}")
    write_parts(f0[KEY + ["ce_prob"]], args.out, "ce_oof_fold0")

    te = cap(unseen_pairs(args.data_dir))
    te["ce_prob"] = predict(model, tok, *attach_text(te, t_te), args)
    write_parts(te, args.out, "ce_test_unseen")
    log("done")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
