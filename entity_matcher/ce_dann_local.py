"""DANN proxy on a local GPU: does domain-adversarial alignment help a cross-encoder on an UNSEEN country?

Source (labelled): India fold!=0 pairs of the 500k-S1 precomputed sample. Target (unlabelled): US fold!=0 pairs.
The model is xlm-roberta-base with a match head on the [CLS] vector; a domain head (India vs US) sits behind a
gradient-reversal layer (Ganin & Lempitsky 2015; DADER, SIGMOD 2022). --lam 0 is the control: the domain head still
trains (its accuracy is logged) but no reversed gradient reaches the encoder, so DANN - control isolates alignment.

Evaluation: US fold-0 and India fold-0 pairs of the same sample -> pair AUC / logloss per country, written to
<out>/scores_fold0.parquet (s1_id, pool_id, ce_prob) for entity-level checks.

Usage: python -m entity_matcher.ce_dann_local --lam 0.0 --out artefacts/dann/control
       python -m entity_matcher.ce_dann_local --lam 0.2 --out artefacts/dann/dann
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class DannCE(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        self.match = nn.Sequential(nn.Dropout(0.1), nn.Linear(h, 1))
        self.domain = nn.Sequential(nn.Linear(h, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1))

    def forward(self, ids, mask, lam=0.0):
        cls = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        # lam == 0 (control): detach so the domain head learns but the encoder gets no domain gradient
        dom_in = GradReverse.apply(cls, lam) if lam > 0 else cls.detach()
        return self.match(cls).squeeze(-1), self.domain(dom_in).squeeze(-1)


def texts(pairs, s1, pool):
    a = s1.reindex(pairs["s1_id"]).fillna("")
    b = pool.reindex(pairs["pool_id"]).fillna("")
    return ((a["business_name"] + " | " + a["business_address"]).tolist(),
            (b["business_name"] + " | " + b["business_address"]).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=0.0, help="max GRL weight (0 = control)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--n_src", type=int, default=300_000)
    ap.add_argument("--bs", type=int, default=32, help="source pairs per step (the same number of target pairs)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--eval_n", type=int, default=0, help="score only a sample of the fold-0 pairs (smoke tests)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda"

    s1 = pd.read_parquet("artefacts/train/s1.parquet", columns=["entity_id", "business_name", "business_address",
                                                                "country_norm"]).set_index("entity_id")
    pool = pd.concat([pd.read_parquet(f"artefacts/train/s{k}.parquet", columns=["entity_id", "business_name",
                                                                                "business_address"])
                      for k in (2, 3)]).set_index("entity_id")
    for t in (s1, pool):
        t["business_name"] = t["business_name"].fillna("")
        t["business_address"] = t["business_address"].fillna("")
    pr = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("precomputed/ce/train_pairs_part*.parquet"))],
                   ignore_index=True)
    pr["c"] = pr["s1_id"].map(s1["country_norm"])
    hard = pr["lgbm_prob"] >= 0.001
    src = pr[(pr["c"] == "india") & (pr["fold"] != 0) & (hard | (pr["label"] == 1))]
    src = entity_matcher.sample(min(args.n_src, len(src)), random_state=args.seed)
    tgt = pr[(pr["c"] == "us") & (pr["fold"] != 0) & hard].sample(len(src), random_state=args.seed, replace=True)
    ev = pr[(pr["fold"] == 0) & hard & pr["c"].isin(["india", "us"])]
    if args.eval_n:
        ev = ev.sample(min(args.eval_n, len(ev)), random_state=args.seed)
    print(f"source India {len(src):,} (pos {entity_matcher.label.mean():.3f}) | target US {len(tgt):,} (labels unused) | "
          f"eval fold-0 {len(ev):,}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = DannCE(args.model).to(dev)

    def enc(a, b):
        t = tok(a, b, truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
        return t["input_ids"].to(dev), t["attention_mask"].to(dev)

    log = []
    if not args.eval_only:
        sa, sb = texts(src, s1, pool)
        ta, tb = texts(tgt, s1, pool)
        ys = torch.tensor(src["label"].to_numpy(), dtype=torch.float32)
        steps = len(src) // args.bs
        params = [{"params": model.enc.parameters(), "lr": args.lr}, {"params": model.match.parameters(), "lr": args.lr * 5},
                  {"params": model.domain.parameters(), "lr": args.lr * 5}]
        opt = torch.optim.AdamW(params, weight_decay=0.01)
        sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
        scaler = torch.amp.GradScaler()
        bce = nn.BCEWithLogitsLoss()
        order = rng.permutation(len(src))
        t0, acc_d, n_d, ml = time.time(), 0.0, 0, 0.0
        model.train()
        for st in range(steps):
            idx = order[st * args.bs:(st + 1) * args.bs]
            p = st / steps
            lam = args.lam * (2 / (1 + math.exp(-10 * p)) - 1)
            i1, m1 = enc([sa[i] for i in idx], [sb[i] for i in idx])
            i2, m2 = enc([ta[i] for i in idx], [tb[i] for i in idx])
            with torch.autocast("cuda", dtype=torch.float16):
                lm, ds = model(i1, m1, lam)
                _, dt = model(i2, m2, lam)
                loss_m = bce(lm.float(), ys[idx].to(dev))
                dl = torch.cat([ds, dt]).float()
                dy = torch.cat([torch.zeros_like(ds), torch.ones_like(dt)]).float()
                loss = loss_m + bce(dl, dy)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            acc_d += ((dl > 0).float() == dy).float().mean().item()
            n_d += 1
            ml += loss_m.item()
            if (st + 1) % 200 == 0:
                rec = {"step": st + 1, "of": steps, "lam": round(lam, 4), "match_loss": round(ml / n_d, 4),
                       "domain_acc": round(acc_d / n_d, 4), "min": round((time.time() - t0) / 60, 1)}
                log.append(rec)
                print(rec, flush=True)
                acc_d, n_d, ml = 0.0, 0, 0.0
        torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    else:
        model.load_state_dict(torch.load(os.path.join(args.out, "model.pt")))

    model.eval()
    ea, eb = texts(ev, s1, pool)
    probs = np.zeros(len(ev), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(ev), 256):
            ii, mm = enc(ea[i:i + 256], eb[i:i + 256])
            with torch.autocast("cuda", dtype=torch.float16):
                lm, _ = model(ii, mm)
            probs[i:i + 256] = torch.sigmoid(lm.float()).cpu().numpy()
    out = ev[["s1_id", "pool_id"]].assign(ce_prob=probs)
    out.to_parquet(os.path.join(args.out, "scores_fold0.parquet"))
    res = {}
    for c in ("india", "us"):
        m = (ev["c"] == c).to_numpy()
        y = ev["label"].to_numpy()[m]
        res[c] = {"auc": round(float(roc_auc_score(y, probs[m])), 5),
                  "logloss": round(float(log_loss(y, np.clip(probs[m], 1e-6, 1 - 1e-6))), 5), "pairs": int(m.sum())}
    print(json.dumps(res), flush=True)
    json.dump({"args": vars(args), "eval": res, "log": log}, open(os.path.join(args.out, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
