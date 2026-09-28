"""Fixed fit / dev / lockbox split of the training S1s for the improvement loop.

Only fold-0 S1s (crc32(s1_id) % 5 == 0) are split: every cross-encoder in handoff/ was trained on
folds 1-4, so fold 0 is the only place where all base scores (LightGBM OOF + every CE) are
out-of-sample. Groups are S1 entities: the ground truth is one-to-one (no S2/S3 id under two S1s),
so an S1 plus its matches is the whole cluster and never straddles partitions.
Stratified by country x number of true matches (0, 1, 2, 3, 4+), seed 42, 70 / 15 / 15.

  python -m src.splits   -> artefacts/splits/{fit,dev,lockbox}.parquet (s1_id, country), splits.json
"""
import json
import os
import zlib

import numpy as np
import pandas as pd

from src.io_utils import read_tsv

OUT = "artefacts/splits"
FRACS = {"fit": 0.70, "dev": 0.15, "lockbox": 0.15}
SEED = 42


def main():
    gt = read_tsv("dataset/train/train_ground_truth.tsv")
    s1 = read_tsv("dataset/train/train_source1.tsv", usecols=["entity_id", "country"])
    df = gt.rename(columns={"source1_entity_id": "s1_id"})
    df = df[[zlib.crc32(s.encode()) % 5 == 0 for s in df["s1_id"]]].copy()
    df["country"] = df["s1_id"].map(dict(zip(s1["entity_id"], s1["country"])))
    n = df["matched_entity_ids"].map(lambda m: len([x for x in m.split(",") if x]))
    df["stratum"] = df["country"] + "_" + n.clip(upper=4).astype(str)
    rng = np.random.default_rng(SEED)
    parts = {k: [] for k in FRACS}
    for _, g in df.sort_values("s1_id").groupby("stratum", sort=True):
        idx = rng.permutation(len(g))
        a, b = int(round(len(g) * FRACS["fit"])), int(round(len(g) * (FRACS["fit"] + FRACS["dev"])))
        for name, sl in (("fit", idx[:a]), ("dev", idx[a:b]), ("lockbox", idx[b:])):
            parts[name].append(g.iloc[sl][["s1_id", "country", "stratum"]])
    os.makedirs(OUT, exist_ok=True)
    meta = {"seed": SEED, "universe": "fold-0 S1 (crc32 % 5 == 0)", "n_fold0": len(df)}
    for name, lst in parts.items():
        p = pd.concat(lst).sort_values("s1_id").reset_index(drop=True)
        p[["s1_id", "country"]].to_parquet(os.path.join(OUT, f"{name}.parquet"), index=False)
        meta[name] = {"n_s1": len(p), "by_country": p["country"].value_counts().to_dict(),
                      "singleton_share": round(float((p["stratum"].str.endswith("_0")).mean()), 4)}
    ids = [set(pd.read_parquet(os.path.join(OUT, f"{k}.parquet"))["s1_id"]) for k in FRACS]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "partitions overlap"
    with open(os.path.join(OUT, "splits.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
