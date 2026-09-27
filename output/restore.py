"""Rebuild matching_results.tsv (and candidate_pairs.tsv where parts exist) from the committed .gz files.

  python output/restore.py              # final build (v14_last)
  python output/restore.py v13_union    # any build folder

v13_union shares its candidate set with v14_last, so its candidates are restored from v14_last's parts.
"""
import glob
import gzip
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED_CANDIDATES = {"v13_union": "v14_last"}


def gunzip(src_parts, dst):
    """Concatenate gzip parts in order and decompress them to dst."""
    joined = dst + ".gz.tmp"
    with open(joined, "wb") as out:
        for p in src_parts:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out)
    with gzip.open(joined, "rb") as f, open(dst, "wb") as out:
        shutil.copyfileobj(f, out)
    os.remove(joined)
    print("wrote", dst)


if __name__ == "__main__":
    build = sys.argv[1] if len(sys.argv) > 1 else "v14_last"
    d = os.path.join(HERE, build)
    gunzip([os.path.join(d, "matching_results.tsv.gz")], os.path.join(d, "matching_results.tsv"))
    src = os.path.join(HERE, SHARED_CANDIDATES.get(build, build))
    parts = sorted(glob.glob(os.path.join(src, "candidate_pairs.tsv.gz.part*")))
    if parts:
        gunzip(parts, os.path.join(d, "candidate_pairs.tsv"))
    else:
        print(f"no candidate parts for {build} (only matching_results is stored)")
