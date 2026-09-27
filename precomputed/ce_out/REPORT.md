# Cross-encoder results (NVIDIA H100 80GB)

## Model and training
- `FacebookAI/xlm-roberta-base` (MIT, 278M), `AutoModelForSequenceClassification`, num_labels=1, BCEWithLogits
- bf16 autocast, max_len 128, batch 256, lr 2e-5 (AdamW, wd 0.01), 5% warmup + linear decay, 1 epoch, grad clip 1.0
- Train rows: fold != 0 only; all rows with lgbm_prob >= 0.001 + 10% of the rest (seed 42)
  -> **3,451,055 pairs** (1,373,809 positive), 13,481 steps, **42.2 min** training
- Text: `business_name | business_address` from the source TSVs (S1 text, candidate text)

## Fold-0 held-out (rows with lgbm_prob >= 0.001; 684,947 pairs, 342,905 positive)

| | AUC | logloss |
|---|---|---|
| ce_prob | **0.99817** | **0.04916** |
| lgbm_prob | 0.99497 | 0.07768 |

## Files
- `ce_oof_fold0_part*.parquet`: s1_id, pool_id, ce_prob (float32), sorted by s1_id, zstd
- `ce_test_part*.parquet`: all 13,047,424 test pairs (5 parts, 8–22 MB), scored in 23.0 min

## Entity-level held-out macro F0.5 (entity_matcher/blend_eval.py, fold-0 99,964 S1, decision tuned on one half / scored on the other, both ways)
```
fold-0: 99,964 S1 | 684,947 pairs with lgbm_prob >= 0.001 | CE missing on 0
  lgbm             held-out macro F0.5 0.9723 | india 0.9677 | us 0.9754 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  ce               held-out macro F0.5 0.9802 | india 0.9817 | us 0.9791 | params(all) {'method': 'expf', 'alpha': 3.0, 'one_to_one': True}
  prob_avg_w0.3    held-out macro F0.5 0.9801 | india 0.9789 | us 0.9809 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  logit_avg_w0.3   held-out macro F0.5 0.9851 | india 0.9853 | us 0.9849 | params(all) {'method': 'expf', 'alpha': 1.0, 'one_to_one': True}
  prob_avg_w0.5    held-out macro F0.5 0.9829 | india 0.9829 | us 0.9828 | params(all) {'method': 'expf', 'alpha': 1.0, 'one_to_one': True}
  logit_avg_w0.5   held-out macro F0.5 0.9865 | india 0.9869 | us 0.9863 | params(all) {'method': 'expf', 'alpha': 1.5, 'one_to_one': True}
  prob_avg_w0.7    held-out macro F0.5 0.9845 | india 0.9854 | us 0.9839 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  logit_avg_w0.7   held-out macro F0.5 0.9857 | india 0.9863 | us 0.9852 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  prob_avg_w0.85   held-out macro F0.5 0.9830 | india 0.9842 | us 0.9822 | params(all) {'method': 'expf', 'alpha': 3.0, 'one_to_one': True}
  logit_avg_w0.85  held-out macro F0.5 0.9839 | india 0.9846 | us 0.9835 | params(all) {'method': 'expf', 'alpha': 3.0, 'one_to_one': True}
```
