"""Step 3b: profile the gold pairs that blocking never turned into candidates, on FIT S1s only.

A missed pair = (fit S1, true S2/S3 match) absent from the full step-3 candidate table
(precomputed/full_out/oof_train_full_part*, all candidates before any probability filter).
Each missed pair gets measurable flags and one primary cause (first rule that fires):
  translit      pool name is in a non-Latin script (Devanagari / Bengali ...)
  same_name     identical name_core or no-space name (lost to the per-S1 cap / competition)
  name_variant  name token Jaccard >= 0.5 (suffix / abbreviation / word order / extra word)
  name_typo     phonetic-skeleton token Jaccard >= 0.5 (spelling / typo / transliteration in Latin)
  same_address  names differ but house number shared and street token Jaccard >= 0.5 (DBA / trade name)
  other         name and address both differ
  python -m entity_matcher.rescue_profile  -> artefacts/rescue/fit_missed.parquet + printed breakdown with examples
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from entity_matcher.local_score import load_gold, split_ids

OUT = "artefacts/rescue"
COLS = ["entity_id", "business_name", "business_address", "name_core", "name_ns", "name_skel", "addr_clean",
        "postal", "nums", "country_norm"]
LEGAL = {"pvt", "private", "ltd", "limited", "llc", "inc", "incorporated", "corp", "corporation", "co", "company",
         "llp", "lp", "plc", "pllc", "pc", "opc", "the"}
_LATIN = re.compile(r"[A-Za-z]")


def jac(a, b):
    a, b = set(a.split()), set(b.split())
    return len(a & b) / len(a | b) if a and b else 0.0


def load_rows(path, ids):
    t = ds.dataset(path).to_table(columns=COLS, filter=pc.field("entity_id").isin(pa_array(ids)))
    return t.to_pandas().set_index("entity_id")


def pa_array(ids):
    import pyarrow as pa
    return pa.array(sorted(ids), type=pa.string())


def main():
    os.makedirs(OUT, exist_ok=True)
    fit = split_ids("fit")
    fit_ids = set(fit["s1_id"])
    gold = load_gold(fit_ids)
    files = sorted(glob.glob("precomputed/full_out/oof_train_full_part*.parquet"))
    cand = ds.dataset(files).to_table(columns=["s1_id", "pool_id"], filter=pc.field("fold") == 0).to_pandas()
    cand = cand[cand["s1_id"].isin(fit_ids)]
    have = set(zip(cand["s1_id"], cand["pool_id"]))
    n_cand = len(cand)
    del cand
    miss = pd.DataFrame([(s, p) for s, g in gold.items() for p in g if (s, p) not in have], columns=["s1_id", "pool_id"])
    n_gold = sum(len(g) for g in gold.values())
    print(f"fit: {len(fit_ids):,} S1, {n_gold:,} gold pairs, {n_cand:,} candidates ({n_cand / len(fit_ids):.1f}/S1), "
          f"missed {len(miss):,} ({len(miss) / n_gold:.2%}) on {miss['s1_id'].nunique():,} S1", flush=True)
    # is the missed pool record a candidate of ANY training S1 (all folds)?
    pool_set = pa_array(set(miss["pool_id"]))
    seen = set()
    for f in files:
        t = ds.dataset(f).to_table(columns=["pool_id"], filter=pc.field("pool_id").isin(pool_set))
        seen |= set(t.column("pool_id").to_pylist())
    miss["orphan"] = ~miss["pool_id"].isin(seen)
    a = load_rows("artefacts/train/s1.parquet", set(miss["s1_id"])).reindex(miss["s1_id"]).reset_index(drop=True)
    b = pd.concat([load_rows(f"artefacts/train/s{k}.parquet", set(miss["pool_id"])) for k in (2, 3)])
    b = b.reindex(miss["pool_id"]).reset_index(drop=True)
    a, b = a.fillna(""), b.fillna("")
    f = pd.DataFrame({"country": a["country_norm"], "s1_id": miss["s1_id"], "pool_id": miss["pool_id"],
                      "orphan": miss["orphan"]})
    f["translit"] = [not _LATIN.search(x) and bool(x.strip()) for x in b["business_name"]]
    f["name_eq"] = (a["name_core"] == b["name_core"]) | ((a["name_ns"] == b["name_ns"]) & (a["name_ns"] != ""))
    f["name_jac"] = [jac(x, y) for x, y in zip(a["name_core"], b["name_core"])]
    f["skel_jac"] = [jac(x, y) for x, y in zip(a["name_skel"], b["name_skel"])]
    nx = [set(x.split()) for x in a["nums"]]
    ny = [set(y.split()) for y in b["nums"]]
    f["num_shared"] = [bool(x & y) for x, y in zip(nx, ny)]
    f["num_conflict"] = [bool(x) and bool(y) and not (x & y) for x, y in zip(nx, ny)]
    strip = lambda s: " ".join(t for t in s.split() if not t.isdigit())
    f["street_jac"] = [jac(strip(x), strip(y)) for x, y in zip(a["addr_clean"], b["addr_clean"])]
    f["pool_addr_empty"] = b["addr_clean"].str.len() == 0
    f["postal_eq"] = (a["postal"] == b["postal"]) & (a["postal"] != "")
    f["legal_diff"] = [bool((set(x.lower().split()) ^ set(y.lower().split())) & LEGAL)
                       for x, y in zip(a["business_name"], b["business_name"])]
    cause = np.select(
        [f["translit"], f["name_eq"], f["name_jac"] >= 0.5, f["skel_jac"] >= 0.5,
         f["num_shared"] & (f["street_jac"] >= 0.5)],
        ["translit", "same_name", "name_variant", "name_typo", "same_address"], "other")
    f["cause"] = cause
    for c in ("business_name", "business_address"):
        f[f"a_{c}"], f[f"b_{c}"] = a[c].to_numpy(), b[c].to_numpy()
    f.to_parquet(os.path.join(OUT, "fit_missed.parquet"), index=False)

    print("\nprimary cause x country (missed gold pairs):")
    print(pd.crosstab(f["cause"], f["country"], margins=True).to_string())
    flags = ["orphan", "translit", "name_eq", "legal_diff", "num_shared", "num_conflict", "pool_addr_empty", "postal_eq"]
    print("\nflag rates by country:")
    print(f.groupby("country")[flags].mean().round(3).T.to_string())
    print("\nmean similarities by cause:")
    print(f.groupby("cause")[["name_jac", "skel_jac", "street_jac"]].mean().round(2).to_string())
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    for (cause_, ctry), g in f.groupby(["cause", "country"]):
        print(f"\n--- {cause_} / {ctry} ({len(g):,}) examples:")
        for _, r in g.sample(min(3, len(g)), random_state=0).iterrows():
            print(f"  S1: {r['a_business_name']!r} | {r['a_business_address']!r}\n"
                  f"  {r['pool_id'][:2]}: {r['b_business_name']!r} | {r['b_business_address']!r}")


if __name__ == "__main__":
    main()
