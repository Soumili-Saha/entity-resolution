"""Pair -> token ids for the Qwen cross-encoders.

plain    : the v11 prompt "Record A: ...\nRecord B: ...\nSame business?" (Qwen3-4B-Base, Qwen3.5 adapters)
pair     : tokenizer(a, b) sentence-pair input with the model's own special tokens (bge-reranker-v2-m3 / xlm-r CEs)
reranker_short : the reranker template without the 39-token system block and with a short instruction (~70 tokens
           instead of ~130, so training runs ~2x faster)
reranker : Qwen3-Reranker's own chat template (system judge prompt, <Instruct>/<Query>/<Document>, empty think block),
           so the last token is where the reranker predicts "yes"/"no". Only the record texts are truncated; the
           template around them is always kept whole.
"""
RR_PRE = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct '
          'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
RR_SUF = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
RR_INS_SHORT = "Same real-world business?"
RR_INS = ("Given a business listing (name | address), judge whether the Document is a listing of the same "
          "real-world business at the same place")


def encode_pairs(tok, a, b, style, max_len):
    if style == "plain":
        prompts = [f"Record A: {x}\nRecord B: {y}\nSame business?" for x, y in zip(a, b)]
        return tok(prompts, truncation=True, max_length=max_len, add_special_tokens=False)["input_ids"]
    if style == "pair":
        return tok(list(a), list(b), truncation=True, max_length=max_len)["input_ids"]
    short = style == "reranker_short"
    pre = tok("<|im_start|>user\n" if short else RR_PRE, add_special_tokens=False)["input_ids"]
    suf = tok(RR_SUF, add_special_tokens=False)["input_ids"]
    ins = RR_INS_SHORT if short else RR_INS
    body = [f"<Instruct>: {ins}\n<Query>: {x}\n<Document>: {y}" for x, y in zip(a, b)]
    body = tok(body, truncation=True, max_length=max_len - len(pre) - len(suf), add_special_tokens=False)["input_ids"]
    return [pre + x + suf for x in body]


def init_yes_no_head(model, model_dir, tok):
    """score.weight = W[yes] - W[no] of the checkpoint's LM head, so the untrained classifier already outputs the
    reranker's own logit(yes) - logit(no)."""
    import json, os
    import torch
    from safetensors import safe_open
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
    else:  # single-file checkpoint (e.g. Qwen3-Reranker-0.6B)
        with safe_open(os.path.join(model_dir, "model.safetensors"), "pt") as f:
            wm = {k: "model.safetensors" for k in f.keys()}
    key = "lm_head.weight" if "lm_head.weight" in wm else "model.embed_tokens.weight"
    with safe_open(os.path.join(model_dir, wm[key]), "pt") as f:
        W = f.get_tensor(key)
    y, n = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    model.score.weight.data.copy_((W[y].float() - W[n].float())[None].to(model.score.weight.dtype))
    return key, y, n
