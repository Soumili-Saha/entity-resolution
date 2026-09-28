# Step 3: full data (all 2.2M train S1) + decoy features, configs/full.yaml

```
[rank:train] 54,593,626 pairs, 7,578,192 positives, 2,206,821 S1
[rank:train] 54,593,626 pairs, 7,578,192 positives, 2,206,821 S1
  fold 0: best_iter 2999, val logloss 0.0165
  fold 1: best_iter 2991, val logloss 0.0166
  fold 2: best_iter 3000, val logloss 0.0166
  fold 3: best_iter 2995, val logloss 0.0166
  fold 4: best_iter 2999, val logloss 0.0166
  decision params {'method': 'expf', 'alpha': 3.0, 'one_to_one': True} -> macro F0.5 0.9794
[rank:train] stage1 OOF macro F0.5 = 0.9794 on 300,000 S1
  fold 0: best_iter 539, val logloss 0.0156
  fold 1: best_iter 774, val logloss 0.0157
  fold 2: best_iter 776, val logloss 0.0158
  fold 3: best_iter 652, val logloss 0.0157
  fold 4: best_iter 646, val logloss 0.0157
  decision params {'method': 'expf', 'alpha': 1.5, 'one_to_one': True} -> macro F0.5 0.9805
[rank:train] stage2 OOF macro F0.5 = 0.9805
[export] 1,732,544 S1 rows, 5.6% empty, 3.39 matches per S1
WARNING: ID-existence check is OFF (the default) — not checking that matched/candidate IDs exist in the test set. Every other rule is still checked. Re-run with --check-ids to enable it (needs test_source2/3.tsv; uses more memory). A nonexistent ID only lowers your score, never rejects your submission.
PASS — no blocking issues found. Safe to submit.
```

- oof_train_full_part*.parquet: s1_id, pool_id, prob (final-stage OOF), fold
- probs_model_full_part*.parquet: test probabilities
- matching_results.tsv.gz / candidate_pairs.tsv.gz(.partNN): decide output (gunzip; cat parts first)
