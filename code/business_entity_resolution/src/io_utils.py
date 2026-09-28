"""TSV reading/writing and entity-ID checks. Always read with sep='\t'."""
import os

import pandas as pd


def read_tsv(path, usecols=None):
    """Read a challenge TSV as strings, keeping empty ID lists as ''."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=usecols, engine="pyarrow")


def load_sources(data_dir, split):
    """Return (s1, pool) for a split; pool = Source 2 + Source 3 with a 'source' column (2/3)."""
    base = os.path.join(data_dir, split)
    s1 = read_tsv(os.path.join(base, f"{split}_source1.tsv"))
    parts = []
    for k in (2, 3):
        df = read_tsv(os.path.join(base, f"{split}_source{k}.tsv"))
        df["source"] = k
        parts.append(df)
    s1["source"] = 1
    return s1, pd.concat(parts, ignore_index=True)


def load_ground_truth(data_dir):
    """Ground truth as {s1_id: set(matched ids)}; singletons map to an empty set."""
    gt = read_tsv(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
    return {s: {x for x in m.split(",") if x} for s, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"])}


def write_tsv(s1_ids, id_lists, path, list_col):
    """Write one row per S1 id with a comma-joined, de-duplicated ID list (possibly empty)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{list_col}\n")
        for s1_id in s1_ids:
            ids = list(dict.fromkeys(id_lists.get(s1_id, ())))
            f.write(f"{s1_id}\t{','.join(ids)}\n")


def source_of(entity_id):
    """Return 'S1', 'S2' or 'S3' from an entity_id prefix."""
    return entity_id[:2]
