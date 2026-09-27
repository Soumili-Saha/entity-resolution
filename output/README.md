# Outputs

One folder per build. Each folder holds `matching_results.tsv.gz` and the recipe that produced it (`blend.json` or
`union.json`). Candidate sets are stored only for v1 and the final build, gzip-split into parts under 90 MB.

```bash
python output/restore.py              # final build → output/v14_last/{matching_results,candidate_pairs}.tsv
python output/restore.py v13_union    # any other build
```

**Validation score** is held-out macro F0.5 on labelled India/US S1s (see `docs/methodology.md` §2).
**Test score** is macro F0.5 on the hidden test labels (India, US and France), scored externally.

| build | what changed | validation F0.5 | test F0.5 |
|---|---|---|---|
| `v1` | level 1: fine-tuned e5 + token blocking, LightGBM stage 1 + 2, 200k S1 | 0.9699 | |
| `v2` | level 1 on 500k S1 | 0.9719 | |
| `v3_blend` | logit blend LightGBM + cross-encoder (xlm-r base) | 0.9865 | 0.98068 |
| `v3_fr_strict` | v3, France logit −1 | 0.9865 | 0.98092 |
| `v3_fr_loose` | v3, France logit +1 | 0.9865 | |
| `v4_stack_fr1` | stacker: LightGBM + CE + competition context, France −1 | 0.9871 | 0.98206 |
| `v4_stack_fr2` | v4, France −2 | 0.9871 | |
| `v5_large_fr1` | + CE large | 0.9877 | 0.98355 |
| `v6_france_st` | + France self-trained CE in the CE-large slot | | |
| `v7a_full_frst` / `v7b_full` | stacker on full-data LightGBM, with / without France self-training | 0.9895 | |
| `v8_full_all` | 434k full-data fold-0 S1: + CE full, France swap | 0.9899 | 0.98514 |
| `v8b_full_nofrst` / `v8c_fr0` | v8 without France self-training / without France shift | 0.9899 | – / 0.98504 |
| `v9_xl` | + xl cross-encoder | | |
| `v10a_e2` / `v10b_e2_nofrst` | CE second epoch replacing CE full | 0.9901 | 0.98487 / – |
| `v11_qwen` | + Qwen3-4B LoRA cross-encoder | 0.9903 | 0.98620 |
| `v12_mylgbm` | + companion LightGBM as a 6th stacker input (India/US; France from v11) | 0.9901 dev (Δ +0.00034) | 0.98631 |
| `v13_iu` | v12 + companion-only pairs, India/US only (18,920) | | |
| **`v13_union`** | v12 + companion-only pairs, all countries (28,319) | | **0.988** |
| `v14_last` | v13_union + 10,779 companion pairs the stacker rejected at p ≥ 0.3 | | pending |

Recipes: v11 → `entity_matcher.stack_submit`, v12 → `entity_matcher.loop_eval --final`,
v13 / v14 → `entity_matcher.union_submit`. The exact commands are in the main README.
