# Cross-encoder results: xlm-roberta-large (NVIDIA H100 80GB)

- `FacebookAI/xlm-roberta-large` (MIT, 560M), num_labels=1, BCEWithLogits; bf16, max_len 128, batch 128,
  lr 1e-5 (AdamW, wd 0.01), 5% warmup + linear decay, 1 epoch, grad clip 1.0
- Train rows: fold != 0; all rows with lgbm_prob >= 0.001 + 10% of the rest (seed 42), same as ce_out
  -> **3,451,055 pairs** (1,373,809 positive), 26,962 steps, **49.2 min**

## Fold-0 held-out (rows with lgbm_prob >= 0.001; 684,947 pairs, 342,905 positive)

| | AUC | logloss |
|---|---|---|
| ce_large | **0.99839** | **0.04570** |
| lgbm_prob (v2) | 0.99497 | 0.07768 |
| ce base (ce_out, for reference) | 0.99817 | 0.04916 |

## Files
- `ce_oof_fold0_part*.parquet`: s1_id, pool_id, ce_prob (float32), sorted by s1_id, zstd
- `ce_test_part*.parquet`: all 13,047,424 test pairs, 5 parts

## Entity-level held-out macro F0.5 (entity_matcher/blend_eval.py, fold-0 99,964 S1, decision tuned on one half / scored on the other, both ways)
```
fold-0: 99,964 S1 | 684,947 pairs with lgbm_prob >= 0.001 | CE missing on 0
  lgbm             held-out macro F0.5 0.9723 | india 0.9677 | us 0.9754 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  ce               held-out macro F0.5 0.9812 | india 0.9823 | us 0.9805 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  prob_avg_w0.3    held-out macro F0.5 0.9805 | india 0.9796 | us 0.9811 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  logit_avg_w0.3   held-out macro F0.5 0.9859 | india 0.9862 | us 0.9858 | params(all) {'method': 'expf', 'alpha': 1.0, 'one_to_one': True}
  prob_avg_w0.5    held-out macro F0.5 0.9833 | india 0.9833 | us 0.9834 | params(all) {'method': 'expf', 'alpha': 1.0, 'one_to_one': True}
  logit_avg_w0.5   held-out macro F0.5 0.9871 | india 0.9875 | us 0.9869 | params(all) {'method': 'expf', 'alpha': 1.5, 'one_to_one': True}
  prob_avg_w0.7    held-out macro F0.5 0.9854 | india 0.9859 | us 0.9850 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  logit_avg_w0.7   held-out macro F0.5 0.9864 | india 0.9867 | us 0.9862 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  prob_avg_w0.85   held-out macro F0.5 0.9840 | india 0.9850 | us 0.9834 | params(all) {'method': 'expf', 'alpha': 3.0, 'one_to_one': True}
  logit_avg_w0.85  held-out macro F0.5 0.9845 | india 0.9849 | us 0.9843 | params(all) {'method': 'expf', 'alpha': 3.0, 'one_to_one': True}
```
