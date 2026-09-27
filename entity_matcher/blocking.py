"""Candidate generation.

Every true pair in training shares the same country label, so blocking runs inside each
country value (an open set of strings - France is just another value; records whose
country appears only on one side still get blocked against the whole pool).

Passes (union):
  token_block : IDF-weighted overlap of rare tokens (name skeleton tokens + address tokens).
  embed_block : multilingual sentence-embedding kNN on 'name | address' (GPU, see embed.py).

Output: DataFrame(s1_idx, pool_idx, tok_score, tok_rank[, emb_score, emb_rank]).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _prefixed(col, prefix):
    """'ab cd' -> 'P:ab P:cd' for a whole string column (empty stays empty)."""
    col = col.fillna("")
    out = prefix + col.str.replace(" ", " " + prefix, regex=False)
    return out.where(col.str.len() > 0, "")


def token_docs(df):
    """One space-separated token document per record:
    'n:' name-skeleton tokens (+ alias), 's:' whole no-space name, 'a:' address tokens."""
    ns = df["name_ns"].fillna("")
    s_tok = ("s:" + ns).where(ns.str.len() >= 6, "")
    return (_prefixed(df["name_skel"], "n:") + " " + _prefixed(df["name_alt_skel"], "n:") + " "
            + s_tok + " " + _prefixed(df["addr_clean"], "a:")).tolist()


def _topk_rows(S, k):
    """Top-k columns per row of a CSR score matrix. Returns (rows, cols, scores)."""
    rows, cols, vals = [], [], []
    indptr, indices, data = S.indptr, S.indices, S.data
    for r in range(S.shape[0]):
        a, b = indptr[r], indptr[r + 1]
        if a == b:
            continue
        d = data[a:b]
        if b - a > k:
            sel = np.argpartition(-d, k - 1)[:k]
        else:
            sel = np.arange(b - a)
        rows.append(np.full(len(sel), r, dtype=np.int64))
        cols.append(indices[a:b][sel])
        vals.append(d[sel])
    if not rows:
        return np.array([], np.int64), np.array([], np.int64), np.array([], np.float32)
    return np.concatenate(rows), np.concatenate(cols).astype(np.int64), np.concatenate(vals)


def _row_ranks(r, v):
    """1-based rank of each score v within its row r (higher score = rank 1)."""
    order = np.lexsort((-v, r))
    rs = r[order]
    start = np.r_[0, np.flatnonzero(rs[1:] != rs[:-1]) + 1]
    pos = np.arange(len(r)) - np.repeat(start, np.diff(np.r_[start, len(r)]))
    ranks = np.empty(len(r), dtype=np.int16)
    ranks[order] = (pos + 1).astype(np.int16)
    return ranks


def token_country(q, pool_c, k=30, max_df=3000, chunk=20000):
    """Rare-token IDF cosine top-k for one country. q / pool_c are the country's frames.
    Returns local (row, col, score, rank) arrays.

    Tokens with document frequency above `max_df` in the country's pool are ignored - they make
    the sparse product dense and carry little identity signal."""
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(analyzer=str.split, binary=True, dtype=np.float32)
    B = vec.fit_transform(token_docs(pool_c)).tocsc()
    df_count = np.diff(B.indptr)
    keep = np.flatnonzero((df_count <= max_df) & (df_count > 0))
    idf = np.log1p(len(pool_c) / df_count[keep]).astype(np.float32)
    B = (B[:, keep] @ sp.diags(idf)).tocsr()
    A = vec.transform(token_docs(q)).tocsc()[:, keep]
    del vec
    A = (A @ sp.diags(idf)).tocsr()
    A = (sp.diags(1 / (np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel()) + 1e-6)) @ A).tocsr().astype(np.float32)
    B = sp.diags(1 / (np.sqrt(np.asarray(B.multiply(B).sum(1)).ravel()) + 1e-6)) @ B
    BT = B.T.tocsr().astype(np.float32)
    del B
    rows, cols, vals = [], [], []
    for start in range(0, A.shape[0], chunk):
        S = (A[start:start + chunk] @ BT).tocsr()
        r, c, v = _topk_rows(S, k)
        rows.append((r + start).astype(np.int32)); cols.append(c.astype(np.int32)); vals.append(v.astype(np.float32))
    r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    return r, c, v, _row_ranks(r, v)


def union_arrays(n_pool, passes, cap=None):
    """Union several passes {name: (row, col, score, rank)} on (row, col) without pandas merges.
    Missing score = 0, missing rank = 999; best_rank = min over passes; keep the `cap` best per row."""
    keys = np.concatenate([p[0].astype(np.int64) * n_pool + p[1] for p in passes.values()])
    uniq, inv = np.unique(keys, return_inverse=True)
    del keys
    out = {"row": (uniq // n_pool).astype(np.int32), "col": (uniq % n_pool).astype(np.int32)}
    best = np.full(len(uniq), 999, dtype=np.int16)
    off = 0
    for name, (r, c, v, rk) in passes.items():
        sl = inv[off:off + len(r)]
        off += len(r)
        sc = np.zeros(len(uniq), dtype=np.float32)
        sc[sl] = v
        rr = np.full(len(uniq), 999, dtype=np.int16)
        rr[sl] = rk
        out[f"{name}_score"], out[f"{name}_rank"] = sc, rr
        best = np.minimum(best, rr)
    out["best_rank"] = best
    if cap:
        # rows are already grouped (uniq is sorted by row); order by best_rank inside each row
        order = np.lexsort((best, out["row"]))
        rs = out["row"][order]
        start = np.r_[0, np.flatnonzero(rs[1:] != rs[:-1]) + 1]
        pos = np.arange(len(rs)) - np.repeat(start, np.diff(np.r_[start, len(rs)]))
        sel = np.sort(order[pos < cap])
        out = {k2: v2[sel] for k2, v2 in out.items()}
    return out


def block_all(s1, pool, e_s1=None, e_pool=None, token_k=20, token_max_df=3000, embed_k=30, cap=40, verbose=True):
    """Blocking per country value (open set): token pass (+ embedding kNN when embeddings are
    given), unioned and capped per S1. Returns a compact DataFrame with global s1_idx / pool_idx."""
    import time
    t0 = time.time()
    frames = []
    s1_c, pool_c = s1["country_norm"].to_numpy(), pool["country_norm"].to_numpy()
    for country in pd.unique(s1_c):
        si = np.flatnonzero(s1_c == country)
        pi = np.flatnonzero(pool_c == country)
        if len(pi) == 0:
            pi = np.arange(len(pool))  # label unseen on the pool side: block against everything
        passes = {"tok": token_country(s1.iloc[si], pool.iloc[pi], k=token_k, max_df=token_max_df)}
        if verbose:
            print(f"    [{country}] token pass: {len(passes['tok'][0]):,} pairs [{time.time() - t0:.0f}s]", flush=True)
        if e_s1 is not None:
            from entity_matcher.embed import knn
            idx, sim = knn(np.ascontiguousarray(e_s1[si]), e_pool, embed_k, base_rows=pi)
            kk = idx.shape[1]
            passes["emb"] = (np.repeat(np.arange(len(si), dtype=np.int32), kk), idx.ravel().astype(np.int32),
                             sim.ravel().astype(np.float32), np.tile(np.arange(1, kk + 1, dtype=np.int16), len(si)))
            del idx, sim
            if verbose:
                print(f"    [{country}] embedding pass done [{time.time() - t0:.0f}s]", flush=True)
        u = union_arrays(len(pi), passes, cap)
        del passes
        df = pd.DataFrame({k2: v2 for k2, v2 in u.items() if k2 not in ("row", "col")})
        df.insert(0, "s1_idx", si[u["row"]].astype(np.int32))
        df.insert(1, "pool_idx", pi[u["col"]].astype(np.int32))
        frames.append(df)
        if verbose:
            print(f"    [{country}] union: {len(df):,} pairs for {len(si):,} S1 [{time.time() - t0:.0f}s]", flush=True)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["s1_idx", "pool_idx"], kind="stable").reset_index(drop=True)


def token_block(s1, pool, k=30, max_df=3000):
    """Token pass only (all countries), as a DataFrame(s1_idx, pool_idx, tok_score, tok_rank)."""
    out = block_all(s1, pool, token_k=k, token_max_df=max_df, cap=None, verbose=False)
    return out.drop(columns=["best_rank"])
