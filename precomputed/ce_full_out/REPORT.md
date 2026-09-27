# Cross-encoder retrained on full step-3 candidates (continued training of CE-large)

- Init artefacts/ce_large/final, lr 5e-6, bs 128, 1 epoch, bf16, max_len 128
- Train: step-3 OOF rows (all 2.2M S1) with fold_of(s1_id,5) != 0: all positives + negatives with prob >= 0.001 + 5% of the rest, capped at 4M
- ce_oof_fold0: step-3 fold-0 pairs (prob >= 0.001) UNION 500k-sample fold-0 pairs (lgbm_prob >= 0.001)
- ce_test: probs_model_full pairs (prob >= 0.001) UNION 500k-sample test pairs

## fold0_metrics.json
```
{
 "n_union": 2755596,
 "n_step3_rows": 2635403,
 "pos_step3_rows": 1515931,
 "ce_full_on_step3_rows": {
  "auc": 0.9980883623817856,
  "logloss": 0.04916246261186566
 },
 "step3_lgbm_on_step3_rows": {
  "auc": 0.996289284527138,
  "logloss": 0.06402155206681674
 },
 "minutes_so_far": 6.4
}
```
## train_info.json
```
{
 "train_pairs": 4000000,
 "train_pos": 2310794,
 "steps": 31250,
 "train_minutes": 133.9
}
```
## Entity-level held-out macro F0.5 (entity_matcher/blend_eval.py on the 500k-sample 99,964 fold-0 S1; reference ce_large 0.9812 / blend 0.9871)
```
fold-0: 99,964 S1 | 684,947 pairs with lgbm_prob >= 0.001 | CE missing on 0
  lgbm             held-out macro F0.5 0.9723 | india 0.9677 | us 0.9754 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  ce               held-out macro F0.5 0.9824 | india 0.9839 | us 0.9814 | params(all) {'method': 'expf', 'alpha': 3.0, 'one_to_one': True}
  prob_avg_w0.3    held-out macro F0.5 0.9806 | india 0.9793 | us 0.9814 | params(all) {'method': 'expf', 'alpha': 1.5, 'one_to_one': True}
  logit_avg_w0.3   held-out macro F0.5 0.9865 | india 0.9868 | us 0.9863 | params(all) {'method': 'expf', 'alpha': 1.0, 'one_to_one': True}
  prob_avg_w0.5    held-out macro F0.5 0.9837 | india 0.9836 | us 0.9837 | params(all) {'method': 'expf', 'alpha': 1.0, 'one_to_one': True}
  logit_avg_w0.5   held-out macro F0.5 0.9875 | india 0.9880 | us 0.9872 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  prob_avg_w0.7    held-out macro F0.5 0.9857 | india 0.9863 | us 0.9853 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  logit_avg_w0.7   held-out macro F0.5 0.9868 | india 0.9870 | us 0.9866 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  prob_avg_w0.85   held-out macro F0.5 0.9846 | india 0.9856 | us 0.9839 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  logit_avg_w0.85  held-out macro F0.5 0.9850 | india 0.9856 | us 0.9847 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
```
    [ce 02:54:34] train sample: core 10,490,256 (pos + prob>=0.001), random fill 1,658,004 -> using 4,000,000
    [ce 02:54:37] fold-0 list: step-3 2,635,403 + 500k-sample 684,947 -> union 2,755,596
    [ce 03:01:18] test list: step-3 10,797,489 + 500k-sample 13,047,424 -> union 13,535,894
