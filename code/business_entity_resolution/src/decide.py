"""Decision layer: pair probabilities -> final match set per S1 entity.

1. one_to_one : each S2/S3 record goes to at most one S1 (verified in training ground truth:
                no S2/S3 id ever appears under two S1s).
2. set choice : per S1, either
     'thresh' - keep candidates with p >= t and p >= r * best_p of that S1, or
     'expf'   - keep the sorted prefix (possibly empty) maximising approximate expected F0.5,
                with `alpha` scaling the value of predicting nothing.
Parameters are tuned on out-of-fold / validation probabilities, never on the leaderboard.
"""
import itertools

import numpy as np

from src.evaluate import f05


def one_to_one(df, prob_col="prob"):
    """Keep each S2/S3 record only under the S1 with its highest probability (ties: first)."""
    df = df.sort_values(prob_col, ascending=False)
    keep = ~df.duplicated(subset="pool_id", keep="first")
    return df[keep]


def group_sorted(df, prob_col="prob"):
    """{s1_id: (ids array, probs array)} with probabilities sorted high -> low."""
    df = df.sort_values(["s1_id", prob_col], ascending=[True, False])
    s1 = df["s1_id"].to_numpy()
    ids = df["pool_id"].to_numpy()
    p = df[prob_col].to_numpy()
    out = {}
    if len(df) == 0:
        return out
    cut = np.flatnonzero(s1[1:] != s1[:-1]) + 1
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(df)]):
        out[s1[a]] = (ids[a:b], p[a:b])
    return out


def choose_thresh(ids, probs, t, r):
    """Absolute threshold t plus relative-to-best ratio r."""
    if len(probs) == 0 or probs[0] < t:
        return []
    return list(ids[(probs >= t) & (probs >= r * probs[0])])


def choose_set(ids, probs, alpha=1.0):
    """Sorted prefix (possibly empty) maximising approximate expected F0.5."""
    if len(probs) == 0:
        return []
    best_k, best = 0, float(np.prod(1.0 - probs)) * alpha
    exp_gold, exp_tp = probs.sum(), 0.0
    for k, p in enumerate(probs, 1):
        exp_tp += p
        score = 1.25 * exp_tp / (0.25 * exp_gold + k)
        if score > best:
            best, best_k = score, k
    return list(ids[:best_k])


def choose_set_exact(ids, probs, alpha=1.0, max_n=15):
    """Sorted prefix maximising the EXACT expected F0.5 under independent Bernoulli(p_i) labels:
    E_k = sum_{a,b} P(TP_k = a) P(rest = b) * 1.25 a / (0.25 (a + b) + k), and E_0 = alpha * P(no match).
    Distributions are Poisson-binomial, built by convolution (candidates beyond max_n are ignored)."""
    p = np.asarray(probs[:max_n], dtype=np.float64)
    n = len(p)
    if n == 0:
        return []
    pre = [np.ones(1)]
    for x in p:
        pre.append(np.convolve(pre[-1], [1 - x, x]))
    suf = [np.ones(1)]
    for x in p[::-1]:
        suf.append(np.convolve(suf[-1], [1 - x, x]))
    suf = suf[::-1]  # suf[k] = distribution of the number of positives among p[k:]
    best_k, best = 0, alpha * float(np.prod(1 - p))
    for k in range(1, n + 1):
        a = np.arange(k + 1)[:, None]
        b = np.arange(n - k + 1)[None, :]
        val = np.where(a > 0, 1.25 * a / (0.25 * (a + b) + k), 0.0)
        score = float(pre[k] @ val @ suf[k])
        if score > best:
            best, best_k = score, k
    return list(ids[:best_k])


def apply_params(grouped, s1_ids, params):
    """{s1_id: [matched ids]} for every S1 under the given parameters."""
    out = {}
    for s in s1_ids:
        if s not in grouped:
            out[s] = []
            continue
        ids, probs = grouped[s]
        if params["method"] == "thresh":
            out[s] = choose_thresh(ids, probs, params["t"], params["r"])
        elif params["method"] == "expf_exact":
            out[s] = choose_set_exact(ids, probs, params["alpha"])
        else:
            out[s] = choose_set(ids, probs, params["alpha"])
    return out


FAST_GRID = [{"method": "expf", "alpha": a} for a in (1.0, 1.5, 2.0, 3.0, 5.0)]


def tune(df, gold, s1_ids, verbose=True, grid=None, o2o_opts=(True, False)):
    """Grid-search decision parameters (with and without one-to-one); returns (params, score).
    Default grid: thresholds x ratios + expected-F0.5 alphas; pass grid=FAST_GRID, o2o_opts=(True,)
    for the reduced grid (expected-F0.5 with one-to-one has won every full search so far)."""
    best, best_score = None, -1.0
    if grid is None:
        grid = [{"method": "thresh", "t": float(t), "r": r}
                for t, r in itertools.product(np.round(np.arange(0.10, 0.96, 0.05), 2), [0.0, 0.2, 0.4, 0.6])]
        grid += [{"method": "expf", "alpha": a} for a in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)]
    for o2o in o2o_opts:
        grouped = group_sorted(one_to_one(df) if o2o else df)
        for params in grid:
            preds = apply_params(grouped, s1_ids, params)
            score = float(np.mean([f05(preds[i], gold.get(i, ())) for i in s1_ids]))
            if score > best_score:
                best, best_score = dict(params, one_to_one=o2o), score
    if verbose:
        print(f"  decision params {best} -> macro F0.5 {best_score:.4f}", flush=True)
    return best, best_score


def decide(df, s1_ids, params):
    """Probabilities DataFrame(s1_id, pool_id, prob) -> {s1_id: [matched ids]}."""
    work = one_to_one(df) if params.get("one_to_one", True) else df
    return apply_params(group_sorted(work), s1_ids, params)
