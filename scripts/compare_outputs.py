"""Per-country diff of two submissions: share of S1 whose predicted set changed, and match counts."""
import sys

import pandas as pd

a, b = sys.argv[1], sys.argv[2]


def load(p):
    d = pd.read_csv(p, sep="\t", dtype=str).fillna("")
    return dict(zip(d["source1_entity_id"], d["matched_entity_ids"].map(lambda x: frozenset(x.split(",")) - {""})))


A, B = load(a), load(b)
s1 = pd.read_parquet("artefacts/test/s1.parquet", columns=["entity_id", "country_norm"])
rows = []
for s, c in zip(s1["entity_id"], s1["country_norm"]):
    x, y = A.get(s, frozenset()), B.get(s, frozenset())
    rows.append((c, x != y, len(x), len(y), len(x & y), len(x) == 0, len(y) == 0))
d = pd.DataFrame(rows, columns=["country", "changed", "na", "nb", "both", "empty_a", "empty_b"])
g = d.groupby("country").agg(S1=("changed", "size"), changed=("changed", "mean"), matches_a=("na", "sum"),
                             matches_b=("nb", "sum"), shared=("both", "sum"), empty_a=("empty_a", "mean"),
                             empty_b=("empty_b", "mean"))
print(g.round(4).to_string())
