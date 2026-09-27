"""Raw pair evidence for the stacker, so it can learn from India/US labels WHEN to trust LightGBM vs the
cross-encoders. On France the two disagree in typed ways: LightGBM over-scores same-name / different-street decoys,
the CE over-scores empty-address records. The stacker only saw the scores, not the address evidence behind them.

Features (normalised views from artefacts/<split>/s*.parquet):
  r_name_tset    token-set ratio of the name cores
  r_name_eq      name cores identical
  r_addr_tset    token-set ratio of the cleaned addresses
  r_street_tset  token-set ratio of the addresses with digits removed (street / city words only)
  r_num_eq       first house number identical (both present)
  r_num_conflict both have numbers and share none
Cached per split in artefacts/exp/raw_<split>.parquet (keyed by s1_id, pool_id).
"""
import os
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from entity_matcher.cpus import n_cpus

COLS = ["r_name_tset", "r_name_eq", "r_addr_tset", "r_street_tset", "r_num_eq", "r_num_conflict"]
_DIG = re.compile(r"\d+")


def _compute(pairs, split):
    cols = ["entity_id", "name_core", "addr_clean", "nums"]
    s1 = pd.read_parquet(f"artefacts/{split}/s1.parquet", columns=cols).set_index("entity_id")
    pool = pd.concat([pd.read_parquet(f"artefacts/{split}/s{k}.parquet", columns=cols) for k in (2, 3)]).set_index("entity_id")
    a = s1.reindex(pairs["s1_id"].to_numpy()).fillna("")
    b = pool.reindex(pairs["pool_id"].to_numpy()).fillna("")
    w = n_cpus()
    out = pd.DataFrame({"s1_id": pairs["s1_id"].to_numpy(), "pool_id": pairs["pool_id"].to_numpy()})
    an, bn = a["name_core"].tolist(), b["name_core"].tolist()
    aa, ba = a["addr_clean"].tolist(), b["addr_clean"].tolist()
    out["r_name_tset"] = cpdist(an, bn, scorer=fuzz.token_set_ratio, workers=w, dtype=np.float32) / 100
    out["r_name_eq"] = (np.array(an, dtype=object) == np.array(bn, dtype=object)).astype(np.float32)
    out["r_addr_tset"] = cpdist(aa, ba, scorer=fuzz.token_set_ratio, workers=w, dtype=np.float32) / 100
    sa = [_DIG.sub(" ", x) for x in aa]
    sb = [_DIG.sub(" ", x) for x in ba]
    out["r_street_tset"] = cpdist(sa, sb, scorer=fuzz.token_set_ratio, workers=w, dtype=np.float32) / 100
    na = [set(str(x).split()) for x in a["nums"]]
    nb = [set(str(x).split()) for x in b["nums"]]
    fa = [str(x).split()[0] if str(x).split() else "" for x in a["nums"]]
    fb = [str(x).split()[0] if str(x).split() else "" for x in b["nums"]]
    out["r_num_eq"] = np.array([float(x != "" and x == y) for x, y in zip(fa, fb)], dtype=np.float32)
    out["r_num_conflict"] = np.array([float(bool(x) and bool(y) and x.isdisjoint(y)) for x, y in zip(na, nb)],
                                     dtype=np.float32)
    # an empty address gives token-set ratio 0 against anything; mark it missing so it does not read as "different"
    empty = (np.array([x == "" for x in ba]) | np.array([x == "" for x in aa]))
    for c in ("r_addr_tset", "r_street_tset"):
        out.loc[empty, c] = np.nan
    return out


def raw_features(pairs, split):
    """COLS for the rows of pairs (same order), computed once per split and cached."""
    path = f"artefacts/exp/raw_{split}.parquet"
    key = pairs[["s1_id", "pool_id"]]
    have = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame(columns=["s1_id", "pool_id"] + COLS)
    miss = key.merge(have[["s1_id", "pool_id"]], on=["s1_id", "pool_id"], how="left", indicator=True)
    miss = miss.loc[miss["_merge"] == "left_only", ["s1_id", "pool_id"]].drop_duplicates()
    if len(miss):
        print(f"raw features: computing {len(miss):,} pairs ({split})", flush=True)
        have = pd.concat([have, _compute(miss, split)], ignore_index=True)
        os.makedirs("artefacts/exp", exist_ok=True)
        have.to_parquet(path, index=False)
    return key.merge(have, on=["s1_id", "pool_id"], how="left")[COLS].set_index(pairs.index)
