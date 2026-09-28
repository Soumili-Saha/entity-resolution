# Transductive self-training on unseen-country test records

Pseudo-labels from our own models only (lgbm v2, ce_base, ce_large on the provided test candidates); no external data.
Continues from artefacts/ce_large/final at lr 5e-6, 1 epoch, mixed 1:1 with labelled train pairs.

## pseudo_stats.json
```
{
 "unseen_countries": [
  "new test country"
 ],
 "unseen_s1": 259452,
 "unseen_pairs": 2102556,
 "pseudo_pos": 600000,
 "pseudo_decoy_neg": 56551,
 "pseudo_rand_neg": 745038,
 "pos_candidates": 746968,
 "decoy_candidates": 56551,
 "replay_pairs": 1401589,
 "train_pairs": 2803178,
 "mean_prob_change_vs_large": -0.022206848487257957
}```
## fold0_metrics.json (US/India sanity check; should not degrade vs ce_large)
```
{
 "n": 684947,
 "ce_v3_unseen": {
  "auc": 0.9983840307048537,
  "logloss": 0.046165992979404956
 },
 "ce_large_reference": {
  "auc": 0.99839,
  "logloss": 0.0457
 }
}```

## Entity-level held-out macro F0.5 on fold 0 (src/blend_eval.py; no-harm check vs ce_large: ce 0.9812, logit_avg_w0.5 0.9871)
```
fold-0: 99,964 S1 | 684,947 pairs with lgbm_prob >= 0.001 | CE missing on 0
  lgbm             held-out macro F0.5 0.9723 | india 0.9677 | us 0.9754 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  ce               held-out macro F0.5 0.9814 | india 0.9828 | us 0.9805 | params(all) {'method': 'expf', 'alpha': 3.0, 'one_to_one': True}
  prob_avg_w0.3    held-out macro F0.5 0.9804 | india 0.9792 | us 0.9812 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  logit_avg_w0.3   held-out macro F0.5 0.9863 | india 0.9867 | us 0.9860 | params(all) {'method': 'expf', 'alpha': 1.5, 'one_to_one': True}
  prob_avg_w0.5    held-out macro F0.5 0.9834 | india 0.9833 | us 0.9834 | params(all) {'method': 'expf', 'alpha': 1.0, 'one_to_one': True}
  logit_avg_w0.5   held-out macro F0.5 0.9872 | india 0.9877 | us 0.9868 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  prob_avg_w0.7    held-out macro F0.5 0.9854 | india 0.9858 | us 0.9851 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  logit_avg_w0.7   held-out macro F0.5 0.9863 | india 0.9868 | us 0.9861 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  prob_avg_w0.85   held-out macro F0.5 0.9843 | india 0.9852 | us 0.9837 | params(all) {'method': 'expf', 'alpha': 3.0, 'one_to_one': True}
  logit_avg_w0.85  held-out macro F0.5 0.9845 | india 0.9848 | us 0.9843 | params(all) {'method': 'expf', 'alpha': 3.0, 'one_to_one': True}
```
- ce_test_unseen_part*.parquet: s1_id, pool_id, ce_prob for ALL test pairs of unseen-country S1 (used for unseen-country S1s only)
- ce_oof_fold0_part*.parquet: fold-0 scores from the adapted model
