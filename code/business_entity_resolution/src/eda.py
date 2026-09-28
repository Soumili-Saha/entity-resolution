"""Quick EDA on the training split: sizes, singletons, matches per S1, one-to-one check,
country consistency and non-Latin script share.

Usage: python -m src.eda --data dataset
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

from src.io_utils import read_tsv

NON_LATIN = re.compile("[^\x00-\u024f]")


def main():
    """Print the EDA report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    args = ap.parse_args()
    tr = os.path.join(args.data, "train")
    gt = read_tsv(os.path.join(tr, "train_ground_truth.tsv"))
    lists = gt["matched_entity_ids"].str.split(",")
    n_match = lists.map(lambda l: sum(1 for x in l if x))
    print(f"S1 rows in GT: {len(gt):,} | singletons: {(n_match == 0).mean():.2%}")
    print("matches per S1:", n_match.value_counts().sort_index().to_dict())
    ex = gt.assign(m=lists).explode("m")
    ex = ex[ex["m"] != ""]
    print(f"true pairs: {len(ex):,} | S2 share {ex['m'].str.startswith('S2').mean():.2%}")
    print(f"pool ids under >1 S1 (one-to-one violations): {ex.loc[ex['m'].duplicated(keep=False), 'm'].nunique():,}")
    s1c = None
    for k in (1, 2, 3):
        df = read_tsv(os.path.join(tr, f"train_source{k}.tsv"))
        print(f"\nsource{k}: {len(df):,} rows | country {df['country'].value_counts().to_dict()}")
        print(f"  empty addr {(df['business_address'] == '').mean():.2%} | non-latin name "
              f"{df['business_name'].map(lambda s: bool(NON_LATIN.search(s))).mean():.2%}")
        if k == 1:
            s1c = dict(zip(df["entity_id"], df["country"]))
        else:
            pc = dict(zip(df["entity_id"], df["country"]))
            sub = ex[ex["m"].str.startswith(f"S{k}")]
            same = np.mean([s1c.get(a) == pc.get(b) for a, b in zip(sub["source1_entity_id"], sub["m"])])
            print(f"  true pairs with same country label: {same:.2%}")


if __name__ == "__main__":
    main()
