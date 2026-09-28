"""Fine-tune the blocking encoder (multilingual-e5-small, MIT) on training match pairs.

Contrastive InfoNCE with in-batch negatives: for a batch of (S1 text, matched S2/S3 text)
pairs, each S1 must pick its own match among all candidates in the batch. Batches are built
within one country and hold at most one pair per S1, so there are no false negatives from
the same entity. Only non-validation S1 entities are used (see evaluate.is_val).

Usage: python -m src.finetune_embed --pairs 600000 --out artefacts/embed_ft
"""
import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModel, AutoTokenizer

from src.embed import record_text
from src.evaluate import is_val
from src.io_utils import load_ground_truth
from src.run import load_normalized


def mean_pool(out, mask):
    """Mean of token embeddings over the attention mask (e5 pooling)."""
    m = mask.unsqueeze(-1).to(out.dtype)
    return (out * m).sum(1) / m.sum(1).clamp(min=1e-6)


def build_pairs(cfg, n_pairs, seed=42):
    """Sample (s1_text, pool_text, country) positive pairs from training S1s outside validation."""
    rng = random.Random(seed)
    gold = load_ground_truth(cfg["paths"]["data_dir"])
    cols = ["entity_id", "name_clean", "addr_clean", "country_norm"]
    s1, pool = load_normalized(cfg, "train", columns=cols)
    s1_text = dict(zip(s1["entity_id"], record_text(s1)))
    s1_cty = dict(zip(s1["entity_id"], s1["country_norm"]))
    pool_text = dict(zip(pool["entity_id"], record_text(pool)))
    del s1, pool
    ids = [i for i, m in gold.items() if m and not is_val(i)]
    rng.shuffle(ids)
    pairs, rounds = [], 0
    while len(pairs) < n_pairs and rounds < 3:
        for i in ids:
            m = rng.choice(sorted(gold[i]))
            pairs.append((i, s1_text[i], pool_text[m], s1_cty[i]))
            if len(pairs) >= n_pairs:
                break
        rounds += 1
    return pairs


def make_batches(pairs, bs, seed=42):
    """Country-homogeneous batches with unique S1 ids inside each batch."""
    rng = random.Random(seed)
    by_c = {}
    for p in pairs:
        by_c.setdefault(p[3], []).append(p)
    batches = []
    for plist in by_c.values():
        rng.shuffle(plist)
        # sort by the S1's last two address tokens (city / state) so in-batch negatives are
        # businesses from the same area - much harder than random negatives
        plist.sort(key=lambda p: " ".join(p[1].rsplit(" ", 2)[-2:]))
        cur, seen = [], set()
        for p in plist:
            if p[0] in seen:
                continue
            cur.append(p)
            seen.add(p[0])
            if len(cur) == bs:
                batches.append(cur)
                cur, seen = [], set()
    rng.shuffle(batches)
    return batches


def main():
    """Train and save a SentenceTransformer-compatible checkpoint."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pairs", type=int, default=600000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--max_steps", type=int, default=2500)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--max_len", type=int, default=48)
    ap.add_argument("--out", default="artefacts/embed_ft")
    args = ap.parse_args()
    if os.name == "nt":  # keep all cores when the session is locked (Windows EcoQoS throttling)
        from src.no_throttle import disable_throttling
        disable_throttling()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    torch.manual_seed(42)
    base = cfg["blocking"]["embed_model"]

    t0 = time.time()
    pairs = build_pairs(cfg, args.pairs)
    batches = make_batches(pairs, args.bs)[:args.max_steps]
    print(f"{len(pairs):,} pairs -> {len(batches):,} batches [{time.time() - t0:.0f}s]", flush=True)

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModel.from_pretrained(base).cuda()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = max(1, len(batches) // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (len(batches) - s) / max(1, len(batches) - warm)))
    scaler = torch.amp.GradScaler()
    labels = torch.arange(args.bs, device="cuda")

    def enc(texts):
        """Tokenise + encode + normalise a list of texts."""
        b = tok(texts, padding="max_length", truncation=True, max_length=args.max_len, return_tensors="pt").to("cuda")
        return F.normalize(mean_pool(model(**b).last_hidden_state, b["attention_mask"]), dim=-1)

    def save(path):
        """Save a SentenceTransformer-format checkpoint with the current weights."""
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(base, device="cpu")
        st[0].auto_model.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})
        os.makedirs(path, exist_ok=True)
        st.save(path)
        print(f"  saved {path} [{time.time() - t0:.0f}s]", flush=True)

    run_loss = 0.0
    for step, batch in enumerate(batches):
        a = [p[1] for p in batch]
        b = [p[2] for p in batch]
        with torch.autocast("cuda", dtype=torch.float16):
            ea, eb = enc(a), enc(b)
            logits = (ea @ eb.T).float() / args.temp
            y = labels[:len(batch)]
            loss = (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)) / 2
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        run_loss = 0.98 * run_loss + 0.02 * loss.item() if step else loss.item()
        if step % 100 == 0:
            print(f"  step {step}/{len(batches)} loss {run_loss:.4f} [{time.time() - t0:.0f}s]", flush=True)
        if step and step % args.save_every == 0:
            save(args.out)

    save(args.out)

if __name__ == "__main__":
    main()
