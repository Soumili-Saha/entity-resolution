"""Cross-encoder re-scorer (xlm-roberta, MIT): xlmr_base_ce_v1 / xlmr_large_ce_v1.

Trains on fold != 0 train pairs (all rows with lgbm_prob >= thr + a random share of the rest), then scores
fold-0 rows with lgbm_prob >= thr (ce_oof_fold0) and all test pairs (ce_test). Pairs: handoff/ce_training_pairs (500k-S1 sample).

Usage: python -m src.ce_rescore --model FacebookAI/xlm-roberta-base --ckpt artefacts/ce_base --out handoff/xlmr_base_ce_v1
       python -m src.ce_rescore --model FacebookAI/xlm-roberta-large --lr 1e-5 --ckpt artefacts/ce_large --out handoff/xlmr_large_ce_v1
"""
import argparse
import glob
import json
import os
import queue
import threading
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sklearn.metrics import log_loss, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.io_utils import read_tsv

PART_MAX_BYTES = 90 * 1024 * 1024


def log(msg):
    print(f"[ce {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_texts(data_dir, split):
    """id -> 'business_name | business_address' for Source 1/2/3 of a split."""
    parts = []
    for k in (1, 2, 3):
        df = read_tsv(os.path.join(data_dir, split, f"{split}_source{k}.tsv"),
                      usecols=["entity_id", "business_name", "business_address"])
        parts.append(pd.Series((df["business_name"] + " | " + df["business_address"]).values, index=df["entity_id"].values))
    return pd.concat(parts)


def attach_text(pairs, texts):
    a = texts.reindex(pairs["s1_id"].values).values
    b = texts.reindex(pairs["pool_id"].values).values
    missing = int(pd.isna(a).sum() + pd.isna(b).sum())
    if missing:
        log(f"WARNING {missing} ids without text -> empty string")
        a = np.where(pd.isna(a), "", a)
        b = np.where(pd.isna(b), "", b)
    return a.astype(object), b.astype(object)


def read_pairs(pattern, columns=None):
    return pd.concat([pd.read_parquet(f, columns=columns) for f in sorted(glob.glob(pattern))], ignore_index=True)


def batches(t1, t2, order, bs, tok, max_len, skip=0, labels=None, depth=16):
    """Tokenize batches in a background thread (the Rust tokenizer releases the GIL)."""
    q = queue.Queue(depth)

    def work():
        for bi, st in enumerate(range(0, len(order), bs)):
            if bi < skip:
                continue
            idx = order[st:st + bs]
            enc = tok(list(t1[idx]), list(t2[idx]), truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt")
            item = {k: v.pin_memory() for k, v in enc.items()}
            if labels is not None:
                item["labels"] = torch.from_numpy(labels[idx].astype(np.float32)).pin_memory()
            q.put((idx, item))
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (x := q.get()) is not None:
        yield x


def train(args, tok, texts, tr=None):
    """Fine-tune args.model (a hub id or a checkpoint dir to continue from). tr: optional ready training set
    (s1_id, pool_id, label) keyed like `texts`; default = fold != 0 pairs of handoff/ce_training_pairs."""
    rng = np.random.RandomState(args.seed)
    if tr is None:
        tr = read_pairs("handoff/ce_training_pairs/train_pairs_part*.parquet")
        tr = tr[tr["fold"] != 0]
        keep = (tr["lgbm_prob"].values >= args.thr) | (rng.rand(len(tr)) < args.rest_frac)
        tr = tr[keep].reset_index(drop=True)
    else:
        tr = tr.reset_index(drop=True)
    y = tr["label"].values.astype(np.float32)
    log(f"train pairs {len(tr):,} (pos {int(y.sum()):,})")
    t1, t2 = attach_text(tr, texts)
    del tr
    order = rng.permutation(len(y))

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, fused=True)
    steps_per_epoch = (len(order) + args.bs - 1) // args.bs
    total = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * total), total)
    state_path = os.path.join(args.ckpt, "state.pt")
    step = 0
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cuda", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step = st["step"]
        log(f"resumed from step {step}")
    start_step = step
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, run_loss, n_loss = time.time(), 0.0, 0
    log(f"steps {total:,} ({steps_per_epoch:,}/epoch), bs {args.bs}, lr {args.lr}")
    for ep in range(args.epochs):
        ep_order = order if ep == 0 else np.random.RandomState(args.seed + ep).permutation(len(order))
        skip = max(0, step - ep * steps_per_epoch)
        if skip >= steps_per_epoch:
            continue
        for _, b in batches(t1, t2, ep_order, args.bs, tok, args.max_len, skip=skip, labels=y):
            b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
            lab = b.pop("labels")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**b).logits.squeeze(-1)
            loss = lossf(logits.float(), lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            run_loss += loss.item() if step % 50 == 0 else 0.0
            n_loss += step % 50 == 0
            if step % 1000 == 0:
                el = time.time() - t0
                log(f"step {step:,}/{total:,} loss {run_loss / max(n_loss, 1):.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e} {el / 60:.1f} min, eta {el / max(step - start_step, 1) * (total - step) / 60:.0f} min")
                run_loss, n_loss = 0.0, 0
            if step % args.save_every == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "step": step}, state_path + ".tmp")
                os.replace(state_path + ".tmp", state_path)
                log(f"checkpoint at step {step:,}")
    final = os.path.join(args.ckpt, "final")
    model.save_pretrained(final)
    tok.save_pretrained(final)
    info = {"train_pairs": int(len(y)), "train_pos": int(y.sum()), "steps": total,
            "train_minutes": round((time.time() - t0) / 60, 1)}
    json.dump(info, open(os.path.join(args.ckpt, "train_info.json"), "w"), indent=1)
    log(f"saved {final}")
    return model


@torch.no_grad()
def predict(model, tok, t1, t2, args):
    """Scores in input order; batches are length-sorted to cut padding."""
    model.eval()
    lens = np.fromiter((len(a) + len(b) for a, b in zip(t1, t2)), dtype=np.int32, count=len(t1))
    order = np.argsort(lens, kind="stable")
    out = np.empty(len(t1), np.float32)
    t0, done = time.time(), 0
    for idx, b in batches(t1, t2, order, args.infer_bs, tok, args.max_len):
        b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**b).logits.squeeze(-1).float()
        out[idx] = torch.sigmoid(logits).cpu().numpy()
        done += len(idx)
        if (done // args.infer_bs) % 2000 == 0:
            log(f"  scored {done:,}/{len(t1):,} ({done / (time.time() - t0):,.0f} pairs/s)")
    log(f"  scored {len(t1):,} in {(time.time() - t0) / 60:.1f} min")
    return out


def write_parts(df, out_dir, name):
    """zstd parquet parts, each < 90 MB; rows sorted by s1_id."""
    df = df.sort_values(["s1_id", "pool_id"], kind="stable").reset_index(drop=True)
    for f in glob.glob(os.path.join(out_dir, f"{name}_part*.parquet")):
        os.remove(f)
    rows = 3_000_000
    while True:
        paths = []
        for i, st in enumerate(range(0, len(df), rows)):
            p = os.path.join(out_dir, f"{name}_part{i:02d}.parquet")
            pq.write_table(pa.Table.from_pandas(df.iloc[st:st + rows], preserve_index=False), p, compression="zstd")
            paths.append(p)
        if max(os.path.getsize(p) for p in paths) < PART_MAX_BYTES:
            break
        for p in paths:
            os.remove(p)
        rows //= 2
    log(f"wrote {len(paths)} parts for {name}: " + ", ".join(f"{os.path.getsize(p) / 2**20:.0f}MB" for p in paths))


def metrics(y, p):
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return {"auc": float(roc_auc_score(y, p)), "logloss": float(log_loss(y, p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    ap.add_argument("--ckpt", default="artefacts/ce_base")
    ap.add_argument("--out", default="handoff/xlmr_base_ce_v1")
    ap.add_argument("--phase", default="all", choices=["all", "train", "oof", "test"])
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--infer_bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--thr", type=float, default=0.001)
    ap.add_argument("--rest_frac", type=float, default=0.10)
    ap.add_argument("--save_every", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.ckpt, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    log(f"{torch.cuda.get_device_name(0)} | {vars(args)}")
    tok = AutoTokenizer.from_pretrained(args.model)
    final = os.path.join(args.ckpt, "final")

    if args.phase in ("all", "train", "oof"):
        texts = load_texts(args.data_dir, "train")
        log(f"train texts {len(texts):,}")
        if args.phase != "oof" and not os.path.exists(os.path.join(final, "config.json")):
            model = train(args, tok, texts)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(final).cuda()
        if args.phase != "train":
            oof = read_pairs("handoff/ce_training_pairs/train_pairs_part*.parquet")
            oof = oof[(oof["fold"] == 0) & (oof["lgbm_prob"] >= args.thr)].reset_index(drop=True)
            log(f"fold-0 pairs {len(oof):,}")
            t1, t2 = attach_text(oof, texts)
            oof["ce_prob"] = predict(model, tok, t1, t2, args)
            m = {"n": int(len(oof)), "pos": int(oof["label"].sum()),
                 "ce": metrics(oof["label"].values, oof["ce_prob"].values),
                 "lgbm": metrics(oof["label"].values, oof["lgbm_prob"].values)}
            json.dump(m, open(os.path.join(args.out, "fold0_metrics.json"), "w"), indent=1)
            log(f"fold-0 metrics {m}")
            write_parts(oof[["s1_id", "pool_id", "ce_prob"]], args.out, "ce_oof_fold0")
            open(os.path.join(args.out, "ce_oof_fold0.DONE"), "w").close()
        del texts
    else:
        model = AutoModelForSequenceClassification.from_pretrained(final).cuda()

    if args.phase in ("all", "test"):
        texts = load_texts(args.data_dir, "test")
        te = read_pairs("handoff/ce_training_pairs/test_pairs_part*.parquet", columns=["s1_id", "pool_id"])
        log(f"test pairs {len(te):,}")
        t1, t2 = attach_text(te, texts)
        del texts
        te["ce_prob"] = predict(model, tok, t1, t2, args)
        write_parts(te, args.out, "ce_test")
        open(os.path.join(args.out, "ce_test.DONE"), "w").close()
    log("done")


if __name__ == "__main__":
    main()
