"""One command to regenerate output/matching_results.tsv and output/candidate_pairs.tsv from the
raw challenge data: normalise -> fine-tune encoder -> embed -> block -> features -> LightGBM ->
decide/export (+ validator). Stages whose cached artefacts exist are NOT skipped unless --resume.

Usage (from the folder containing src/, configs/, dataset/):
    python -m src.reproduce            # full run (~4-5 h on RTX 4060 + 16 GB RAM)
    python -m src.reproduce --resume   # skip steps whose outputs already exist
"""
import argparse
import os
import subprocess
import sys


def step(args, done_marker, resume):
    """Run one pipeline command unless --resume and its output already exists."""
    if resume and done_marker and os.path.exists(done_marker):
        print(f"== skip (exists: {done_marker}): {' '.join(args)}", flush=True)
        return
    print(f"== run: {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, "-m", *args], check=True)


def main():
    """Execute the full pipeline in order."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    A = "artefacts"
    plan = [
        (["src.run", "--stage", "normalize", "--split", "train"], f"{A}/train/s3.parquet"),
        (["src.run", "--stage", "normalize", "--split", "test"], f"{A}/test/s3.parquet"),
        (["src.finetune_embed", "--pairs", "400000", "--max_steps", "2500", "--out", f"{A}/embed_ft"],
         f"{A}/embed_ft/model.safetensors"),
        (["src.run", "--stage", "embed", "--split", "train"], f"{A}/train/emb_pool.npy"),
        (["src.run", "--stage", "embed", "--split", "test"], f"{A}/test/emb_pool.npy"),
        (["src.run", "--stage", "block", "--split", "train"], f"{A}/train/cands.parquet"),
        (["src.run", "--stage", "features", "--split", "train"], None),
        (["src.run", "--stage", "rank", "--split", "train"], None),
        (["src.run", "--stage", "block", "--split", "test"], f"{A}/test/cands.parquet"),
        (["src.run", "--stage", "features", "--split", "test"], None),
        (["src.run", "--stage", "rank", "--split", "test"], None),
        (["src.run", "--stage", "decide", "--split", "test"], None),
    ]
    for args, marker in plan:
        step(args, marker, a.resume)


if __name__ == "__main__":
    main()
