"""Score a pair list with the trained Qwen3-4B CE (base + LoRA adapter saved by qwen_ce.py), no training.

--pairs: parquet with s1_id, pool_id (test split). Output: <out>/<stem>_part*.parquet (s1_id, pool_id, ce_prob float32,
sorted by s1_id, zstd, < 90 MB) + <stem>.DONE. Same prompt format and pooling as qwen_ce.py.
"""
import argparse, os, time

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from prompts import encode_pairs, init_yes_no_head

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-4B-Base")
p.add_argument("--adapter", default="artefacts/qwen_ce/out/adapter")
p.add_argument("--pairs", required=True)
p.add_argument("--split", default="test")
p.add_argument("--data", default="dataset")
p.add_argument("--out", default="handoff/qwen3_4b_lora_ce")
p.add_argument("--stem", default="ce_test_unseen_all")
p.add_argument("--eval_bs", type=int, default=256)
p.add_argument("--max_len", type=int, default=128)
p.add_argument("--params", default="", help="mid-epoch LoRA + head params.pt (saved from qwen_ce.py's ckpt.pt) instead of --adapter")
p.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", help="with --params: as in training")
p.add_argument("--lora_r", type=int, default=32, help="with --params: as in training")
p.add_argument("--lora_scale", type=float, default=1.0, help="WiSE-FT style: scale the LoRA delta by this (1 = as trained)")
p.add_argument("--init_yesno", action="store_true", help="reranker: with --lora_scale, also move the score head toward its yes/no init")
p.add_argument("--canon", default="", help="file.py:func applied to BOTH record texts before scoring (test-time canonicalisation)")
p.add_argument("--prompt", default="plain", choices=["plain", "pair", "reranker_short", "reranker"], help="see prompts.py")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
t0 = time.time()
log = lambda *a: print(f"[{(time.time() - t0) / 60:7.1f} min]", *a, flush=True)
dev = torch.device("cuda")
_reserve = torch.empty(int(14 * 2**30), dtype=torch.uint8, device=dev); del _reserve  # hold room on a shared GPU

pairs = pd.read_parquet(args.pairs)[["s1_id", "pool_id"]]
lut = {}
for s in (1, 2, 3):
    df = pd.read_csv(f"{args.data}/{args.split}/{args.split}_source{s}.tsv", sep="\t", dtype=str,
                     keep_default_na=False, quoting=3)
    lut.update(zip(df.entity_id, df.business_name + " | " + df.business_address))
del df
tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = "<|endoftext|>"
if args.canon:  # optional: canonicalise both sides at test time
    import importlib.util
    path, fn = args.canon.rsplit(":", 1)
    spec = importlib.util.spec_from_file_location("canon_mod", path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    canon = getattr(mod, fn)
    need = set(pairs.s1_id) | set(pairs.pool_id)
    lut = {k: canon(v) for k, v in lut.items() if k in need}
    log(f"canonicalised {len(lut):,} record texts with {args.canon}")
prompts = ([lut.get(a, "") for a in pairs.s1_id], [lut.get(b, "") for b in pairs.pool_id])
missing = sum(1 for b in pairs.pool_id if b not in lut)
del lut
ids = encode_pairs(tok, *prompts, args.prompt, args.max_len)
del prompts
log(f"{len(ids):,} pairs tokenised | missing texts {missing}")

model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1, dtype=torch.bfloat16,
                                                           attn_implementation="sdpa")
model.config.pad_token_id = tok.pad_token_id
head0 = None
if args.init_yesno:
    init_yes_no_head(model, args.model, tok); head0 = model.score.weight.detach().clone()
if args.params:  # a mid-epoch checkpoint: rebuild the training-time LoRA wrapper and load its trainable tensors by name
    lcfg = LoraConfig(task_type="SEQ_CLS", r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                      target_modules=args.target_modules.replace("+", ",").split(","), modules_to_save=["score"])
    model = get_peft_model(model, lcfg)
    ck = torch.load(args.params, map_location="cpu")
    named = dict(model.named_parameters())
    assert set(ck["params"]) <= set(named), f"unknown tensors: {sorted(set(ck['params']) - set(named))[:3]}"
    for n, t in ck["params"].items():
        named[n].data.copy_(t)
    log(f"loaded {len(ck['params'])} tensors from {args.params} (step {ck['step']})")
    model = model.to(dev).eval()
elif os.path.exists(os.path.join(args.adapter, "adapter_config.json")):
    model = PeftModel.from_pretrained(model, args.adapter).to(dev).eval()
else:  # a fully fine-tuned model (qwen_ce.py --full_ft): the "adapter" dir holds the whole model
    model = AutoModelForSequenceClassification.from_pretrained(args.adapter, num_labels=1, dtype=torch.bfloat16).to(dev).eval()
if args.lora_scale != 1.0:  # WiSE-FT for LoRA: W0 + a * dW; the head moves toward its init only if one is known
    n_l = 0
    for m in model.modules():
        if isinstance(getattr(m, "scaling", None), dict):
            for k in m.scaling: m.scaling[k] *= args.lora_scale
            n_l += 1
    for n, q in model.named_parameters():
        if "score" in n and "modules_to_save" in n and head0 is not None:
            q.data.copy_(args.lora_scale * q.data + (1 - args.lora_scale) * head0.to(q.device, q.dtype))
            log("score head interpolated toward the yes/no init")
    log(f"LoRA delta scaled by {args.lora_scale} in {n_l} layers")
log("model + adapter loaded")

order = np.argsort([len(x) for x in ids])
prob = np.empty(len(ids), dtype=np.float32)
with torch.no_grad():
    for i in range(0, len(order), args.eval_bs):
        idx = order[i:i + args.eval_bs]
        L = max(len(ids[j]) for j in idx)
        inp = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, j in enumerate(idx):
            inp[r, : len(ids[j])] = torch.tensor(ids[j]); att[r, : len(ids[j])] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(input_ids=inp.to(dev), attention_mask=att.to(dev)).logits[:, 0].float()
        prob[idx] = torch.sigmoid(logit).cpu().numpy()
        if (i // args.eval_bs) % 500 == 0:
            rate = (i + len(idx)) / max(time.time() - t0, 1)
            log(f"  scored {i + len(idx):,}/{len(order):,}")

out = pairs.assign(ce_prob=prob).sort_values("s1_id", kind="stable")
for k, st in enumerate(range(0, len(out), 3_000_000)):
    f = os.path.join(args.out, f"{args.stem}_part{k:02d}.parquet")
    out.iloc[st:st + 3_000_000].to_parquet(f, compression="zstd", index=False)
    assert os.path.getsize(f) < 90 * 2**20
open(os.path.join(args.out, f"{args.stem}.DONE"), "w").close()
log(f"DONE {len(out):,} rows -> {args.out}/{args.stem}_part*.parquet")
