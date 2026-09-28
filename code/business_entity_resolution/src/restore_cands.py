"""Rebuild artefacts/test/cands.parquet (s1_idx, pool_idx) from the stored step-3 candidate set
(handoff/lgbm_ranker_v1/candidate_pairs.tsv.gz.part*), so src.run.write_submission works without
re-running the GPU blocking stage. Indices follow artefacts/test/s1.parquet and s2 + s3 order.

  python -m src.restore_cands
"""
import glob
import gzip
import io

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

D = "artefacts/test"


def main():
    parts = sorted(glob.glob("handoff/lgbm_ranker_v1/candidate_pairs.tsv.gz.part*"))
    raw = b"".join(open(p, "rb").read() for p in parts)
    s1 = pd.read_parquet(f"{D}/s1.parquet", columns=["entity_id"])["entity_id"]
    pool = pd.concat([pd.read_parquet(f"{D}/s{k}.parquet", columns=["entity_id"])["entity_id"] for k in (2, 3)],
                     ignore_index=True)
    s1_idx = pd.Series(np.arange(len(s1)), index=s1.to_numpy())
    pool_idx = pd.Series(np.arange(len(pool)), index=pool.to_numpy())
    t = pacsv.read_csv(io.BytesIO(gzip.decompress(raw)),
                       parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                       convert_options=pacsv.ConvertOptions(column_types={"source1_entity_id": pa.string(),
                                                                          "candidate_entity_ids": pa.string()},
                                                            strings_can_be_null=False))
    del raw
    lists = pc.split_pattern(t["candidate_entity_ids"], ",")
    lens = pc.list_value_length(lists).to_numpy(zero_copy_only=False)
    ids = pc.list_flatten(lists)
    keep = pc.not_equal(ids, "").to_numpy(zero_copy_only=False)
    a_idx = s1_idx.index.get_indexer(t["source1_entity_id"].to_pandas())
    rows = np.repeat(a_idx, lens)[keep]
    b = pool_idx.index.get_indexer(ids.to_pandas()[keep])
    assert (rows >= 0).all() and (b >= 0).all(), "candidate ids missing from the test sources"
    out = pd.DataFrame({"s1_idx": rows.astype(np.int32), "pool_idx": b.astype(np.int32)})
    out.to_parquet(f"{D}/cands.parquet", index=False)
    print(f"{len(out):,} candidate pairs for {out['s1_idx'].nunique():,} of {len(s1):,} test S1 "
          f"({len(out) / len(s1):.1f} per S1)")


if __name__ == "__main__":
    main()
