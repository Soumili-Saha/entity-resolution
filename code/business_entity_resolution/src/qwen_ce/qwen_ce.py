"""Qwen3-4B-Base cross-encoder (Apache-2.0, 4.02B) for Business Entity Resolution.

LoRA fine-tune of a sequence-classification head on (S1 record, candidate record) pairs from
handoff/ce_training_pairs/train_pairs (fold != 0 only), then score:
  1. fold-0 held-out rows with lgbm_prob >= 0.001 (same set as the ce_large REPORT)  -> qwen_oof_fold0.parquet
  2. test pairs whose ce_large prob is in [0.02, 0.98] (contested)                 -> qwen_test_contested.parquet
Outputs: s1_id, pool_id, ce_prob (float32), sorted by s1_id.
"""
import argparse, glob, math, os, time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from prompts import encode_pairs, init_yes_no_head
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-4B-Base")
p.add_argument("--inputs", default="artefacts/qwen_ce/inputs")
p.add_argument("--data", default="dataset")
p.add_argument("--out", default="artefacts/qwen_ce/out")
p.add_argument("--n_train", type=int, default=0, help="cap train rows (0 = all); for smoke runs")
p.add_argument("--train_file", default="train.parquet", help="training pairs parquet under --inputs (s1_id, pool_id, label)")
p.add_argument("--n_eval", type=int, default=0, help="cap fold-0 rows (0 = all); for smoke runs")
p.add_argument("--n_test", type=int, default=0, help="cap contested test rows (0 = all); for smoke runs")
p.add_argument("--bs", type=int, default=64)
p.add_argument("--eval_bs", type=int, default=256)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--max_len", type=int, default=128)
p.add_argument("--lora_r", type=int, default=32)
p.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
               help="LoRA targets; Qwen3.5 adds its DeltaNet projections in_proj_qkv,in_proj_z,out_proj")
p.add_argument("--reserve_gb", type=float, default=0)
p.add_argument("--torch_deltanet", action="store_true", help="Qwen3.5: use the pure-torch gated-delta-rule instead of fla kernels")
p.add_argument("--prompt", default="plain", choices=["plain", "pair", "reranker_short", "reranker"], help="see prompts.py")
p.add_argument("--init_yesno", action="store_true", help="Qwen3-Reranker: init the score head as W[yes] - W[no]")
p.add_argument("--full_ft", action="store_true", help="train ALL weights (fp32 master weights, bf16 autocast) instead of LoRA; for small CEs such as bge-reranker-v2-m3")
p.add_argument("--accum", type=int, default=1, help="micro-batches per step (same effective batch, less GPU memory)")
p.add_argument("--seed", type=int, default=7)
p.add_argument("--save_every", type=int, default=1000, help="checkpoint LoRA/head + optimizer every N steps (resume on restart)")
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
t0 = time.time()
log = lambda *a: print(f"[{(time.time() - t0) / 60:7.1f} min]", *a, flush=True)
dev = torch.device("cuda")
log("gpu", torch.cuda.get_device_name(0), "| threads", torch.get_num_threads())
if args.reserve_gb:  # hold room on a shared GPU (the caching allocator keeps it after del)
    _reserve = torch.empty(int(args.reserve_gb * 2**30), dtype=torch.uint8, device=dev); del _reserve


def read(pattern):
    return pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(os.path.join(args.inputs, pattern)))]).to_pandas()


def texts(split):
    """id -> 'name | address' for all three sources of a split."""
    out = {}
    for s in (1, 2, 3):
        df = pd.read_csv(f"{args.data}/{split}/{split}_source{s}.tsv", sep="\t", dtype=str,
                         keep_default_na=False, quoting=3)
        out.update(zip(df.entity_id, df.business_name + " | " + df.business_address))
    return out


# ---------------- data ----------------
# built by prep.py: train (fold != 0 only), fold0 (XL's 890,838-pair set), test (contested superset)
train = pd.read_parquet(os.path.join(args.inputs, args.train_file))
if args.n_train: train = train.iloc[: args.n_train]
fold0 = pd.read_parquet(os.path.join(args.inputs, "fold0.parquet"))
if args.n_eval: fold0 = fold0.sample(args.n_eval, random_state=args.seed)
test = pd.read_parquet(os.path.join(args.inputs, "test.parquet"))
if args.n_test: test = test.iloc[: args.n_test]
log(f"train {len(train):,} (pos {train.label.mean():.3f}) | fold0 {len(fold0):,} | test contested {len(test):,}")

tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = "<|endoftext|>"


def encode(df, lut):
    a = [lut.get(x, "") for x in df.s1_id]; b = [lut.get(x, "") for x in df.pool_id]
    miss = sum(1 for x in b if not x)
    enc = encode_pairs(tok, a, b, args.prompt, args.max_len)
    return enc, miss


lut = texts("train")
train_ids, m1 = encode(train, lut); fold0_ids, m2 = encode(fold0, lut)
del lut
lut = texts("test")
test_ids, m3 = encode(test, lut)
del lut
log(f"tokenised | missing texts train {m1} fold0 {m2} test {m3} | mean len {np.mean([len(x) for x in train_ids]):.1f}")


def collate(ids):
    L = max(len(x) for x in ids)
    inp = torch.full((len(ids), L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(ids), L), dtype=torch.long)
    for i, x in enumerate(ids):
        inp[i, : len(x)] = torch.tensor(x); att[i, : len(x)] = 1
    return inp.to(dev, non_blocking=True), att.to(dev, non_blocking=True)


# ---------------- model ----------------
model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1,
                                                           dtype=torch.float32 if args.full_ft else torch.bfloat16,
                                                           attn_implementation="sdpa")
model.config.pad_token_id = tok.pad_token_id
if args.init_yesno:
    log("score head initialised from the reranker's yes/no LM rows:", init_yes_no_head(model, args.model, tok))
if args.torch_deltanet:  # fallback when the fla Triton kernels can't run (correct, slower)
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    for m in model.modules():
        if hasattr(m, "chunk_gated_delta_rule"): m.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
lcfg = LoraConfig(task_type="SEQ_CLS", r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                  target_modules=args.target_modules.replace("+", ",").split(","),
                  modules_to_save=["score"])
if args.full_ft:
    model = model.to(dev)
else:
    model = get_peft_model(model, lcfg).to(dev)
    model.print_trainable_parameters()
n_total = sum(p.numel() for p in model.parameters())
log(f"total params incl. LoRA: {n_total / 1e9:.3f}B")

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
steps = math.ceil(len(train_ids) / args.bs)
sch = get_linear_schedule_with_warmup(opt, int(0.03 * steps), steps)
y_all = torch.tensor(train.label.values, dtype=torch.float32)

# ---------------- score ----------------
@torch.no_grad()
def score(all_ids):
    model.eval()
    order = np.argsort([len(x) for x in all_ids])  # length-sorted batches for speed
    out = np.empty(len(all_ids), dtype=np.float32)
    for i in range(0, len(order), args.eval_bs):
        idx = order[i:i + args.eval_bs]
        inp, att = collate([all_ids[j] for j in idx])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[idx] = torch.sigmoid(model(input_ids=inp, attention_mask=att).logits[:, 0].float()).cpu().numpy()
        if (i // args.eval_bs) % 500 == 0:
            log(f"  scored {i:,}/{len(order):,}")
    return out


def write(df, prob, name):
    o = pd.DataFrame({"s1_id": df.s1_id.values, "pool_id": df.pool_id.values, "ce_prob": prob}).sort_values("s1_id")
    o.to_parquet(os.path.join(args.out, name), compression="zstd", index=False)
    log("wrote", name, len(o))



def evaluate(pf):
    from sklearn.metrics import log_loss, roc_auc_score
    ll = lambda y, p: log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])
    yl = fold0.label.values[: len(pf)]
    log(f"FOLD-0 qwen3-4b on {len(yl):,}: AUC {roc_auc_score(yl, pf):.5f} logloss {ll(yl, pf):.5f}")
    for name, path in [("ce_large", "ce_oof_fold0_part00.parquet"), ("ce_full", "ce_full/ce_oof_fold0_part00.parquet")]:
        try:
            cl = pd.read_parquet(os.path.join(args.inputs, path))
            m = fold0.iloc[: len(pf)][["s1_id", "pool_id", "label", "in_handoff"]].assign(q=pf).merge(cl, on=["s1_id", "pool_id"])
            h = m[m.in_handoff]
            log(f"FOLD-0 vs {name} on {len(m):,} common rows: AUC qwen {roc_auc_score(m.label, m.q):.5f} / {name} "
                f"{roc_auc_score(m.label, m.ce_prob):.5f} | logloss qwen {ll(m.label, m.q):.5f} / {name} {ll(m.label, m.ce_prob):.5f}"
                f" || handoff rows {len(h):,}: AUC qwen {roc_auc_score(h.label, h.q):.5f} / {name} {roc_auc_score(h.label, h.ce_prob):.5f}"
                f" logloss qwen {ll(h.label, h.q):.5f} / {name} {ll(h.label, h.ce_prob):.5f}")
        except Exception as e:
            log(f"{name} comparison skipped:", e)


# pre-flight: exercise scoring, metrics and writing on small slices before the long training run
_pf = score(fold0_ids[:2048]); evaluate(_pf)
write(test.iloc[:512], score(test_ids[:512]), "preflight_test.parquet")
model.train()
log("pre-flight OK")

# ---------------- train (1 epoch) ----------------
# resume: a finished adapter skips training; a mid-run checkpoint restarts at its step
ckpt_path, adapter_dir = os.path.join(args.out, "ckpt.pt"), os.path.join(args.out, "adapter")
trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
start = 0
if os.path.exists(os.path.join(adapter_dir, "DONE")):
    ck = torch.load(os.path.join(adapter_dir, "trainable.pt"), map_location=dev)
    for n, p in trainable.items(): p.data.copy_(ck[n])
    start = steps
    log("loaded finished adapter, skipping training")
elif os.path.exists(ckpt_path):
    ck = torch.load(ckpt_path, map_location=dev)
    for n, p in trainable.items(): p.data.copy_(ck["params"][n])
    opt.load_state_dict(ck["opt"]); sch.load_state_dict(ck["sch"]); start = ck["step"]
    log(f"resumed from checkpoint at step {start}")


def save_ckpt(step):
    tmp = ckpt_path + ".tmp"
    torch.save({"params": {n: p.detach() for n, p in trainable.items()}, "opt": opt.state_dict(),
                "sch": sch.state_dict(), "step": step}, tmp)
    os.replace(tmp, ckpt_path)


model.train()
run_loss, t_step = 0.0, time.time()
for s in range(start, steps):
    ids = train_ids[s * args.bs:(s + 1) * args.bs]
    y = y_all[s * args.bs:(s + 1) * args.bs].to(dev)
    mb, loss = math.ceil(len(ids) / args.accum), 0.0
    for k in range(0, len(ids), mb):
        inp, att = collate(ids[k:k + mb])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(input_ids=inp, attention_mask=att).logits[:, 0].float()
        l = F.binary_cross_entropy_with_logits(logit, y[k:k + mb], reduction="sum") / len(ids)
        l.backward(); loss += l.detach()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
    run_loss += loss.item()
    if (s + 1) % 500 == 0 or s == steps - 1:
        rate = 500 * args.bs / (time.time() - t_step); t_step = time.time()
        log(f"step {s + 1}/{steps} loss {run_loss / 500:.4f} | {rate:.0f} pairs/s | "
            f"ETA train {(steps - s - 1) * args.bs / max(rate, 1) / 60:.0f} min | "
            f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        run_loss = 0.0
    if (s + 1) % args.save_every == 0 and s + 1 < steps:
        save_ckpt(s + 1)
if start < steps:
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    torch.save({n: p.detach() for n, p in trainable.items()}, os.path.join(adapter_dir, "trainable.pt"))
    open(os.path.join(adapter_dir, "DONE"), "w").close()
    log("saved adapter")


if not os.path.exists(os.path.join(args.out, "qwen_oof_fold0.parquet")):
    pf = score(fold0_ids)
    write(fold0, pf, "qwen_oof_fold0.parquet")
    evaluate(pf)
else:
    log("fold-0 scores already written, skipping")

pt = score(test_ids)
write(test, pt, "qwen_test_contested.parquet")
log("DONE")
