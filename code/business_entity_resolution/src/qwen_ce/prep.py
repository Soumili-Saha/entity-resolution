"""Build Qwen CE inputs (run once from the package folder: python src/qwen_ce/prep.py).

train.parquet : step-3 full-data OOF rows with fold != 0 (never fold 0):
                all LightGBM-contested rows (prob in [0.02, 0.98]) + other hard rows (prob >= 0.001), capped,
                + a few easy negatives; label from train_ground_truth.
fold0.parquet : handoff fold-0 rows (lgbm_prob >= 0.001, 684,947) UNION step-3 fold-0 rows contested under
                sigmoid(0.5 logit(lgbm) + 0.5 logit(ce_full)) in [0.02, 0.98]  (XL's fold-0 set, superset-safe).
test.parquet  : test pairs contested under the same rule; lgbm prob from probs_model_full, falling back to the
                handoff v2 lgbm_prob, both variants unioned so XL's 1,267,192-pair set is covered.
"""
import glob, os

import numpy as np
import pandas as pd

I = "artefacts/qwen_ce/inputs"   # written here
D = "dataset"
H = "handoff"
os.makedirs(I, exist_ok=True)
N_TRAIN, N_EASY, SEED = 1_200_000, 60_000, 7
rng = np.random.default_rng(SEED)
rd = lambda p: pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(H, p)))], ignore_index=True)


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def contested(lg, ce):
    z = 1 / (1 + np.exp(-(0.5 * logit(lg) + 0.5 * logit(ce))))
    return (z >= 0.02) & (z <= 0.98)


gt = pd.read_csv(f"{D}/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
gt = gt.assign(pool_id=gt.matched_entity_ids.str.split(",")).explode("pool_id")
gt = gt[gt.pool_id != ""][["source1_entity_id", "pool_id"]].rename(columns={"source1_entity_id": "s1_id"})
gt["label"] = np.int8(1)
print("true pairs", len(gt))

# ---- train: step-3 OOF, fold != 0 ----
oof = rd("lgbm_ranker_v1/oof_train_full_part*.parquet")
print("oof", oof.shape, oof.fold.value_counts().sort_index().to_dict())
nz = oof[oof.fold != 0]
cont = nz[(nz.prob >= 0.02) & (nz.prob <= 0.98)]
hard = nz[(nz.prob >= 0.001) & ~((nz.prob >= 0.02) & (nz.prob <= 0.98))]
easy = nz[nz.prob < 0.001]
print(f"fold!=0: contested {len(cont):,} | other hard {len(hard):,} | easy {len(easy):,}")
parts = [cont.sample(min(len(cont), N_TRAIN - N_EASY), random_state=SEED)]
left = N_TRAIN - N_EASY - len(parts[0])
if left > 0:
    parts.append(hard.sample(min(left, len(hard)), random_state=SEED))
parts.append(easy.sample(N_EASY, random_state=SEED))
train = pd.concat(parts)[["s1_id", "pool_id"]]
train = train.merge(gt, on=["s1_id", "pool_id"], how="left").fillna({"label": 0}).astype({"label": "int8"})
train = train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
print(f"train {len(train):,} pos rate {train.label.mean():.3f}")
train.to_parquet(f"{I}/train.parquet", index=False)

# ---- fold 0 ----
ho = rd("ce_training_pairs/train_pairs_part*.parquet")
ho0 = ho[(ho.fold == 0) & (ho.lgbm_prob >= 0.001)][["s1_id", "pool_id"]]
cf0 = pd.read_parquet(f"{H}/xlmr_large_ce_v2/ce_oof_fold0_part00.parquet")
o0 = oof[oof.fold == 0][["s1_id", "pool_id", "prob"]].merge(cf0, on=["s1_id", "pool_id"])
s3c = o0[contested(o0.prob, o0.ce_prob)][["s1_id", "pool_id"]]
fold0 = pd.concat([ho0, s3c]).drop_duplicates()
fold0 = fold0.merge(gt, on=["s1_id", "pool_id"], how="left").fillna({"label": 0}).astype({"label": "int8"})
fold0["in_handoff"] = fold0.set_index(["s1_id", "pool_id"]).index.isin(ho0.set_index(["s1_id", "pool_id"]).index)
print(f"fold0 {len(fold0):,} (handoff {int(fold0.in_handoff.sum()):,}, step-3 contested {len(s3c):,}) pos {fold0.label.mean():.3f}")
fold0.to_parquet(f"{I}/fold0.parquet", index=False)
del oof, nz, ho

# ---- test ----
lg = rd("lgbm_ranker_v1/probs_model_full_part*.parquet")
cf = rd("xlmr_large_ce_v2/ce_test_part*.parquet")
hot = rd("ce_training_pairs/test_pairs_part*.parquet")
m = cf.merge(lg, on=["s1_id", "pool_id"], how="left").merge(hot, on=["s1_id", "pool_id"], how="left")
mask = np.zeros(len(m), dtype=bool)
for p in (m.prob.fillna(0.0), m.prob.fillna(m.lgbm_prob).fillna(0.0), m.lgbm_prob.fillna(m.prob).fillna(0.0),
          m.prob.fillna(m.ce_prob)):
    mask |= contested(p.values, m.ce_prob.values)
test = m[mask][["s1_id", "pool_id"]].reset_index(drop=True)
print(f"test contested (union of fill variants) {len(test):,}")
test.to_parquet(f"{I}/test.parquet", index=False)
