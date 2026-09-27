"""Step 3c: targeted extra blocking keys for gold pairs the step-3 blocking missed (profile: entity_matcher.rescue_profile).

Every key is a co-occurrence of two tokens that are each moderately rare within the country's S2+S3 pool, so the
combination is highly selective. Keys whose pool frequency exceeds MAX_KEY_POOL are dropped (not selective).
  K1 num_street   house number x rare street token              (same address, different / random name)
  K2 street_pair  pair of the 3 rarest street tokens             (addresses without a house number)
  K3 name_addr    one of the 2 rarest name-skeleton tokens x one of the 3 rarest address tokens
                  (romanised native-script names, name variants and typos at the same locality)
  K4 name_exact   identical name_core, rare in the pool          (same name lost to the 25-per-S1 cap)
Pairs already in the step-3 candidate set are removed; per S1 at most CAP new pairs are kept (most key hits,
then rarest key). Processing is per country, tokens are integer ids from one joint Arrow dictionary.

  python -m entity_matcher.rescue_keys --split train   -> artefacts/rescue/new_pairs_train.parquet (fold-0 S1s) + fit report
  python -m entity_matcher.rescue_keys --split test    -> artefacts/rescue/new_pairs_test.parquet (seen countries only)
"""
import argparse
import glob
import os
import time
import zlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

OUT = "artefacts/rescue"
COLS = ["entity_id", "name_core", "name_skel", "addr_clean", "nums", "country_norm"]
MAX_TOK_DF_STREET = 2000
MAX_TOK_DF_NAME_ADDR = 20000
MAX_KEY_POOL = 20
MAX_NAME_POOL = 5
CAP = 5
LEGAL_SKEL = {"pvt", "prvt", "lmt", "ltd", "lmtd", "llc", "inc", "corp", "crp", "co", "llp", "lp", "plc", "pllc", "pc",
              "opc", "prv", "pr", "l", "lmtt", "lmttd"}
KEYS = ["k1", "k2", "k3", "k4"]


def load(split, k, country, ids=None):
    flt = pc.field("country_norm") == country
    if ids is not None:
        flt = flt & pc.field("entity_id").isin(pa.array(sorted(ids), type=pa.string()))
    return ds.dataset(f"artefacts/{split}/s{k}.parquet").to_table(columns=COLS, filter=flt)


def split_tokens(arr):
    lst = pc.utf8_split_whitespace(arr)
    return pc.list_parent_indices(lst).to_numpy(), pc.list_flatten(lst)


def _pos(p):
    """Position of each entry within its record, for a record-sorted array p."""
    if len(p) == 0:
        return np.zeros(0, dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(p[1:] != p[:-1]) + 1]
    return np.arange(len(p)) - np.repeat(starts, np.diff(np.r_[starts, len(p)]))


def rarest(parent, tok, df, n, keep):
    """Per record the n rarest distinct tokens among those with keep == True: (parent, tok) arrays."""
    p, t = parent[keep], tok[keep]
    order = np.lexsort((t, df[t], p))
    p, t = p[order], t[order]
    first = np.r_[True, (p[1:] != p[:-1]) | (t[1:] != t[:-1])] if len(p) else np.zeros(0, dtype=bool)
    p, t = p[first], t[first]
    sel = _pos(p) < n
    return p[sel], t[sel]


def record_keys(tab, enc, V):
    """(record index, key uint64, key type) for one table; enc = joint dictionary encoder results."""
    n_rec = tab.num_rows
    out = []
    # tokens
    ap, at = enc["addr"]
    np_, nt = enc["nums"]
    sp, st = enc["skel"]
    df_a, df_n = enc["df_addr"], enc["df_skel"]
    is_digit, is_legal, tlen = enc["is_digit"], enc["is_legal"], enc["tlen"]
    street_ok = (~is_digit[at]) & (tlen[at] >= 3)
    # K1: first house number x street tokens with df <= MAX_TOK_DF_STREET
    first_num = np.full(n_rec, -1, dtype=np.int64)
    if len(np_):
        firsts = np.r_[True, np_[1:] != np_[:-1]]
        first_num[np_[firsts]] = nt[firsts]
    m = street_ok & (df_a[at] <= MAX_TOK_DF_STREET) & (first_num[ap] >= 0)
    out.append((ap[m], first_num[ap[m]].astype(np.uint64) * V + at[m].astype(np.uint64), 1))
    # K2: pairs among the 3 rarest street tokens (df <= MAX_TOK_DF_STREET)
    p3, t3 = rarest(ap, at, df_a, 3, street_ok & (df_a[at] <= MAX_TOK_DF_STREET))
    for i in range(3):
        for j in range(i + 1, 3):
            a_idx, b_idx = _nth(p3, i), _nth(p3, j)
            common = np.intersect1d(p3[a_idx], p3[b_idx])
            ta = dict_lookup(p3[a_idx], t3[a_idx], common)
            tb = dict_lookup(p3[b_idx], t3[b_idx], common)
            lo, hi = np.minimum(ta, tb).astype(np.uint64), np.maximum(ta, tb).astype(np.uint64)
            out.append((common, lo * V + hi + np.uint64(V * V), 2))
    # K3: 2 rarest name-skeleton tokens x 3 rarest address tokens (df <= MAX_TOK_DF_NAME_ADDR, non-digit)
    name_ok = (~is_legal[st]) & (tlen[st] >= 2) & (df_n[st] <= MAX_TOK_DF_NAME_ADDR)
    pn, tn = rarest(sp, st, df_n, 2, name_ok)
    pa3, ta3 = rarest(ap, at, df_a, 3, street_ok & (df_a[at] <= MAX_TOK_DF_NAME_ADDR))
    j = pd.merge(pd.DataFrame({"r": pn, "n": tn}), pd.DataFrame({"r": pa3, "a": ta3}), on="r")
    out.append((j["r"].to_numpy(), j["n"].to_numpy().astype(np.uint64) * V + j["a"].to_numpy().astype(np.uint64)
                + np.uint64(2 * V * V), 3))
    # K4: exact name_core (encoded separately)
    nc = enc["name_id"]
    ok = nc >= 0
    out.append((np.flatnonzero(ok), nc[ok].astype(np.uint64) + np.uint64(3 * V * V), 4))
    rec = np.concatenate([o[0] for o in out]).astype(np.int32)
    key = np.concatenate([o[1] for o in out])
    kt = np.concatenate([np.full(len(o[0]), o[2], dtype=np.int8) for o in out])
    return rec, key, kt


def _nth(p, i):
    """Boolean mask of the i-th entry of each record in a record-sorted array."""
    return _pos(p) == i


def dict_lookup(p, t, keys):
    s = pd.Series(t, index=p)
    return s.reindex(keys).to_numpy()


def encode(s1, pool):
    """Joint integer ids for address / number / skeleton tokens and name_core across S1 and pool."""
    enc_s1, enc_pool = {}, {}
    n1 = s1.num_rows
    for field, col in (("addr", "addr_clean"), ("nums", "nums"), ("skel", "name_skel")):
        p1, t1 = split_tokens(s1[col].combine_chunks())
        p2, t2 = split_tokens(pool[col].combine_chunks())
        enc_s1[field], enc_pool[field] = (p1, t1), (p2, t2)
    # one dictionary for all token kinds (numbers and words may coincide: fine, keys are type-offset)
    allt = pa.chunked_array([enc_s1["addr"][1], enc_pool["addr"][1], enc_s1["nums"][1], enc_pool["nums"][1],
                             enc_s1["skel"][1], enc_pool["skel"][1]])
    d = pc.dictionary_encode(allt).combine_chunks()
    idx = d.indices.to_numpy()
    vocab = d.dictionary
    sizes = [len(x) for x in (enc_s1["addr"][1], enc_pool["addr"][1], enc_s1["nums"][1], enc_pool["nums"][1],
                              enc_s1["skel"][1], enc_pool["skel"][1])]
    parts = np.split(idx, np.cumsum(sizes)[:-1])
    V = len(vocab)
    enc_s1["addr"], enc_pool["addr"] = (enc_s1["addr"][0], parts[0]), (enc_pool["addr"][0], parts[1])
    enc_s1["nums"], enc_pool["nums"] = (enc_s1["nums"][0], parts[2]), (enc_pool["nums"][0], parts[3])
    enc_s1["skel"], enc_pool["skel"] = (enc_s1["skel"][0], parts[4]), (enc_pool["skel"][0], parts[5])
    vl = pc.utf8_length(vocab).to_numpy()
    is_digit = pc.utf8_is_digit(vocab).to_numpy(zero_copy_only=False)
    is_legal = pc.is_in(vocab, value_set=pa.array(sorted(LEGAL_SKEL))).to_numpy(zero_copy_only=False)
    # document frequency in the POOL (per record distinct)
    df_addr = np.bincount(np.unique(enc_pool["addr"][0].astype(np.int64) * V + parts[1]) % V, minlength=V)
    df_skel = np.bincount(np.unique(enc_pool["skel"][0].astype(np.int64) * V + parts[5]) % V, minlength=V)
    # name_core ids with pool frequency <= MAX_NAME_POOL
    names = pa.chunked_array([s1["name_core"].combine_chunks(), pool["name_core"].combine_chunks()])
    dn = pc.dictionary_encode(names).combine_chunks()
    nid = dn.indices.to_numpy().astype(np.int64)
    n_empty = pc.equal(dn.dictionary, "").to_numpy(zero_copy_only=False)
    cnt = np.bincount(nid[n1:], minlength=len(dn.dictionary))
    bad = n_empty | (cnt > MAX_NAME_POOL)
    nid = np.where(bad[nid], -1, nid)
    common = {"df_addr": df_addr, "df_skel": df_skel, "is_digit": is_digit, "is_legal": is_legal, "tlen": vl}
    enc_s1.update(common, name_id=nid[:n1])
    enc_pool.update(common, name_id=nid[n1:])
    return enc_s1, enc_pool, np.uint64(V)


def existing_pairs(split, s1_ids):
    """Step-3 candidate set of the given S1s as a set of (s1_id, pool_id)."""
    if split == "train":
        files = sorted(glob.glob("precomputed/full_out/oof_train_full_part*.parquet"))
        t = ds.dataset(files).to_table(columns=["s1_id", "pool_id"],
                                       filter=pc.field("s1_id").isin(pa.array(sorted(s1_ids), type=pa.string())))
        return t.to_pandas()
    c = pd.read_parquet("artefacts/test/cands.parquet")
    s1 = pd.read_parquet("artefacts/test/s1.parquet", columns=["entity_id"])["entity_id"].to_numpy()
    pool = np.concatenate([pd.read_parquet(f"artefacts/test/s{k}.parquet", columns=["entity_id"])["entity_id"].to_numpy()
                           for k in (2, 3)])
    df = pd.DataFrame({"s1_id": s1[c["s1_idx"].to_numpy()], "pool_id": pool[c["pool_idx"].to_numpy()]})
    return df[df["s1_id"].isin(s1_ids)]


def generate(split, country, s1_ids):
    t0 = time.time()
    s1 = load(split, 1, country, s1_ids)
    pool = pa.concat_tables([load(split, k, country) for k in (2, 3)])
    enc1, encp, V = encode(s1, pool)
    r1, key1, kt1 = record_keys(s1, enc1, V)
    rp, keyp, ktp = record_keys(pool, encp, V)
    del enc1, encp
    u1 = np.unique(key1)
    m = np.isin(keyp, u1)  # only pool keys that some S1 also has; their pool frequency is unchanged
    rp, keyp, ktp = rp[m], keyp[m], ktp[m]
    kp = pd.DataFrame({"rec_pool": rp, "key": keyp, "kt": ktp}).drop_duplicates()
    uk, cnt = np.unique(kp["key"].to_numpy(), return_counts=True)
    freq = cnt[np.searchsorted(uk, kp["key"].to_numpy())]
    kp = kp[freq <= MAX_KEY_POOL].assign(kfreq=freq[freq <= MAX_KEY_POOL].astype(np.int16))
    k1 = pd.DataFrame({"rec_s1": r1, "key": key1, "kt": kt1}).drop_duplicates()
    j = k1.merge(kp, on=["key", "kt"])
    del kp, k1
    hits = j.groupby(["rec_s1", "rec_pool", "kt"]).size().unstack(fill_value=0)
    hits.columns = [f"k{c}" for c in hits.columns]
    for c in KEYS:
        if c not in hits.columns:
            hits[c] = 0
    pairs = hits[KEYS].astype(np.int16)
    pairs["hits"] = pairs[KEYS].sum(axis=1)
    pairs["min_kfreq"] = j.groupby(["rec_s1", "rec_pool"])["kfreq"].min()
    pairs = pairs.reset_index()
    del j
    pairs["s1_id"] = np.asarray(s1["entity_id"])[pairs["rec_s1"].to_numpy()]
    pairs["pool_id"] = np.asarray(pool["entity_id"])[pairs["rec_pool"].to_numpy()]
    ex = existing_pairs(split, set(pairs["s1_id"]))
    ex = set(zip(ex["s1_id"], ex["pool_id"]))
    pairs = pairs[[(a, b) not in ex for a, b in zip(pairs["s1_id"], pairs["pool_id"])]]
    pairs = pairs.sort_values(["s1_id", "hits", "min_kfreq"], ascending=[True, False, True])
    pairs = pairs[pairs.groupby("s1_id").cumcount() < CAP]
    pairs = pairs.drop(columns=["rec_s1", "rec_pool"]).assign(country=country)
    print(f"[{split}/{country}] {len(s1_ids):,} S1, pool {pool.num_rows:,}: {len(pairs):,} new pairs "
          f"({len(pairs) / max(len(s1_ids), 1):.2f}/S1) [{time.time() - t0:.0f}s]", flush=True)
    return pairs.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.split == "train":  # fold-0 S1s (fit / dev / lockbox); fit only is used for any design choice
        gt = pd.read_parquet("artefacts/train/s1.parquet", columns=["entity_id", "country_norm"])
        gt = gt[[zlib.crc32(s.encode()) % 5 == 0 for s in gt["entity_id"]]]
    else:  # seen countries only (France stays byte-identical to v11)
        gt = pd.read_parquet("artefacts/test/s1.parquet", columns=["entity_id", "country_norm"])
        seen = set(pd.read_parquet("artefacts/train/s1.parquet", columns=["country_norm"])["country_norm"])
        gt = gt[gt["country_norm"].isin(seen)]
    parts = [generate(args.split, c, set(g["entity_id"])) for c, g in gt.groupby("country_norm")]
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(os.path.join(OUT, f"new_pairs_{args.split}.parquet"), index=False)
    if args.split == "train":
        report_fit(out)


def report_fit(pairs):
    """Recovered gold vs added volume per key, on FIT S1s only."""
    from entity_matcher.local_score import load_gold, split_ids
    fit = split_ids("fit")
    ids = set(fit["s1_id"])
    gold = load_gold(ids)
    p = pairs[pairs["s1_id"].isin(ids)].copy()
    p["gold"] = [b in gold.get(a, ()) for a, b in zip(p["s1_id"], p["pool_id"])]
    missed = pd.read_parquet(os.path.join(OUT, "fit_missed.parquet"))
    print(f"\nFIT: {len(ids):,} S1, blocking-missed gold {len(missed):,}; new pairs {len(p):,} "
          f"({len(p) / len(ids):.3f}/S1), gold among them {int(p['gold'].sum()):,} "
          f"({p['gold'].mean():.2%} precision, {p['gold'].sum() / len(missed):.1%} of missed recovered)")
    for k in KEYS:
        s = p[p[k] > 0]
        only = s[s[[x for x in KEYS if x != k]].sum(axis=1) == 0]
        print(f"  {k}: {len(s):,} pairs ({len(s) / len(ids):.3f}/S1), gold {int(s['gold'].sum()):,} "
              f"(prec {s['gold'].mean():.2%}); unique to {k}: {len(only):,} pairs, gold {int(only['gold'].sum()):,}")
    for h in (1, 2, 3):
        s = p[p["hits"] >= h]
        print(f"  hits >= {h}: {len(s):,} pairs ({len(s) / len(ids):.3f}/S1), gold {int(s['gold'].sum()):,} "
              f"(prec {s['gold'].mean():.2%})")
    m = missed.merge(p[["s1_id", "pool_id", "gold"]], on=["s1_id", "pool_id"], how="left")
    print("recovered by profile cause:\n" + m.assign(rec=m["gold"].fillna(False).astype(bool))
          .groupby(["cause", "country"])["rec"].agg(["size", "sum", "mean"]).round(3).to_string())


if __name__ == "__main__":
    main()
