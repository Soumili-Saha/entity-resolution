# Transductive self-training on unseen-country test records (France), round 2

Pseudo-labels from the pipeline's own models only (step-3 LightGBM, ce_large, ce_france round 1 on 500k-sample + step-3 test candidates); no external data.
Continues from artefacts/ce_large/final at lr 5e-6, 1 epoch, mixed 1:1 with labelled train pairs.

## pseudo_stats.json
```
{
 "unseen_countries": [
  "France"
 ],
 "unseen_s1": 259452,
 "unseen_pairs": 2234246,
 "signals": "lgbm_full, ce_large, ce_france_r1",
 "pseudo_pos": 773613,
 "pseudo_decoy_neg": 107730,
 "pseudo_rand_neg": 944327,
 "pos_candidates": 773613,
 "decoy_candidates": 107730,
 "replay_pairs": 1825670,
 "train_pairs": 3651340,
 "mean_prob_change_vs_prev": -0.010970879346132278
}```
## fold0_metrics.json (US/India sanity check; should not degrade vs ce_large)
```
{
 "n": 684947,
 "ce_france": {
  "auc": 0.9984009858792267,
  "logloss": 0.04565634617057701
 },
 "ce_large_reference": {
  "auc": 0.99839,
  "logloss": 0.0457
 }
}```

## Entity-level held-out macro F0.5 on fold 0 (entity_matcher/blend_eval.py; no-harm check vs ce_large: ce 0.9812, logit_avg_w0.5 0.9871)
```
fold-0: 99,964 S1 | 684,947 pairs with lgbm_prob >= 0.001 | CE missing on 0
  lgbm             held-out macro F0.5 0.9723 | india 0.9677 | us 0.9754 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  ce               held-out macro F0.5 0.9816 | india 0.9830 | us 0.9807 | params(all) {'method': 'expf', 'alpha': 3.0, 'one_to_one': True}
  prob_avg_w0.3    held-out macro F0.5 0.9805 | india 0.9794 | us 0.9813 | params(all) {'method': 'expf', 'alpha': 1.5, 'one_to_one': True}
  logit_avg_w0.3   held-out macro F0.5 0.9863 | india 0.9867 | us 0.9860 | params(all) {'method': 'expf', 'alpha': 1.5, 'one_to_one': True}
  prob_avg_w0.5    held-out macro F0.5 0.9834 | india 0.9833 | us 0.9835 | params(all) {'method': 'expf', 'alpha': 1.0, 'one_to_one': True}
  logit_avg_w0.5   held-out macro F0.5 0.9871 | india 0.9875 | us 0.9869 | params(all) {'method': 'expf', 'alpha': 1.5, 'one_to_one': True}
  prob_avg_w0.7    held-out macro F0.5 0.9854 | india 0.9858 | us 0.9851 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  logit_avg_w0.7   held-out macro F0.5 0.9863 | india 0.9865 | us 0.9862 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  prob_avg_w0.85   held-out macro F0.5 0.9842 | india 0.9850 | us 0.9836 | params(all) {'method': 'expf', 'alpha': 3.0, 'one_to_one': True}
  logit_avg_w0.85  held-out macro F0.5 0.9846 | india 0.9848 | us 0.9845 | params(all) {'method': 'expf', 'alpha': 3.0, 'one_to_one': True}
```
- ce_test_unseen_part*.parquet: s1_id, pool_id, ce_prob for ALL test pairs of unseen-country S1 (use for France only)
- ce_oof_fold0_part*.parquet: fold-0 scores from the adapted model
