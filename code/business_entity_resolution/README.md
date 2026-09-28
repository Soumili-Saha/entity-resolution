# Business Entity Resolution

Matches every Source 1 business record to its Source 2 / Source 3 records. This folder rebuilds the submitted
`output/matching_results.tsv` and `output/candidate_pairs.tsv` (in the submission root, `../../output/`)
byte for byte.

## 1. Repository layout

```
business_entity_resolution/
├── README.md
├── requirements.txt
├── configs/            run settings (full.yaml = the setting used for the submission)
├── dataset/            place the challenge TSV files here (empty in the package)
├── handoff/            stored model scores used by the final steps (see section 2)
├── src/                all code (see section 6)
│   ├── qwen_ce/        Qwen3-4B LoRA cross-encoder (training + scoring)
│   └── pipeline_v2/    second matching pipeline (library + notebooks)
└── utils/
    └── validate_submission.py   format / ID checker for the two output files
```

Three folders are created inside `code/business_entity_resolution/` when the commands in section 4 run. They
are not part of the submission. The submitted files are in the top-level `output/` folder of the submission,
which is a different folder:

| folder | created by | holds |
|---|---|---|
| `code/business_entity_resolution/artefacts/` | every step | cached intermediate data: cleaned records (`train/`, `test/`), fit / dev / lockbox split (`splits/`), stacker runs with their probabilities and reports (`exp/<tag>/`) |
| `code/business_entity_resolution/output/` | steps 7–9 | the three rebuilt versions `v12_mylgbm/`, `v13_union/`, `v14_last/`; step 10 checks that `v14_last/` equals the submitted files |
| `code/business_entity_resolution/logs/` | steps 5–7 | experiment log (`experiments.csv`, one row per stacker run) |

```
<submission root>/
├── output/                                  submitted files (shipped)
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
└── code/business_entity_resolution/
    ├── artefacts/                           created by the rebuild
    ├── logs/                                created by the rebuild
    └── output/                              created by the rebuild
        ├── v12_mylgbm/
        ├── v13_union/
        └── v14_last/                        must equal <submission root>/output/ (step 10)
```

## 2. Stored model scores (`handoff/`)

**Where to get them:** the stored scores (~2.3 GB) are in the GitHub repository
**https://github.com/Soumili-Saha/entity-resolution**, folder `code/business_entity_resolution/handoff/`. The submission ZIP leaves them out
because of its 1 GB size limit. To add them to the submission folder:

```bash
git clone --depth 1 https://github.com/Soumili-Saha/entity-resolution.git
cp -r entity-resolution/code/business_entity_resolution/handoff/* <submission root>/code/business_entity_resolution/handoff/
```

These are the **scores** each model produced for every candidate pair (`s1_id, pool_id, probability`),
not the model weights. The final steps only need these scores, and the fine-tuned transformer weights
(xlm-roberta-large ×3, Qwen3-4B) are many GB; section 5 gives the training commands and settings for each.

| folder | model | contents | used in |
|---|---|---|---|
| `lgbm_ranker_v1` | e5-small blocking + two-stage LightGBM (main ranker) | test candidates (`candidate_pairs.tsv.gz.part*`), OOF probabilities for all train pairs (`oof_train_full_part*`), test probabilities (`probs_model_full_part*`), its own match list (`matching_results.tsv.gz`, for reference) | steps 3, 5–7 |
| `xlmr_base_ce_v1` | xlm-roberta-base cross-encoder | fold-0 train scores (`ce_oof_fold0_*`, `ce_extra_fold0_*`), test scores (`ce_test_*`, `ce_extra_test_*`) | steps 5–7 |
| `xlmr_large_ce_v1` | xlm-roberta-large cross-encoder | same layout | steps 5–7 |
| `xlmr_large_ce_v2` | xlm-roberta-large, continued on the full training data | same layout | steps 5–7 |
| `xlmr_large_ce_v3_unseen` | xlm-roberta-large, self-trained on test records of the new test country (absent from training) | fold-0 scores, new-country test scores (`ce_test_unseen_*`) | steps 5–7 (replaces `xlmr_large_ce_v1` on the new country) |
| `qwen3_4b_lora_ce` | Qwen3-4B-Base + LoRA classifier | fold-0 scores, test scores on contested pairs + all new-country pairs | steps 5–7 |
| `lgbm_ranker_v2` | pipeline-v2 LightGBM | its probability for every stacker pair (train fold 0 and test) | steps 6–7 |
| `lgbm_v2_xlmr_base_v2_matches` | pipeline-v2 final matches (LightGBM v2 + xlm-roberta-base v2 blend) | `matching_results.tsv.gz` | steps 8–9 |
| `stacker_v1_matches` | stacker v1 (all scores above except LightGBM v2) | `matching_results.tsv.gz`, build settings `blend.json` | step 7 (new-country rows) |
| `ce_training_pairs` | pair sample for cross-encoder training | all candidates of 500k train S1s with OOF LightGBM prob, label and fold (`train_pairs_part*`); test candidates with LightGBM prob ≥ 0.001 (`test_pairs_part*`) | section 5 only |
| `pipeline_v2_work` | pipeline v2 work folder (written by its notebooks; 5.6 GB, not included in the repository or the ZIP) | `prep/` cleaned records; `run_frac0.15_seed42/` train blocks, features, OOF LightGBM; `final_v4/` LightGBM stage-1/2 models + vocab; `test_run_v4/` test blocks and LightGBM test scores; `ce/` cross-encoder model, scores (base, large, self-trained) and blend stacker | source of `lgbm_ranker_v2` (`src/export_mylgbm.py`) and `lgbm_v2_xlmr_base_v2_matches` (`test_v4.ipynb`); not read by section 4 |

`REPORT.md`, `fold0_metrics.json`, `train_info.json` and `pseudo_stats.json` in these folders record each
model's training data, settings and fold-0 metrics.

## 3. How the final file is built

```
dataset/*.tsv ─► normalise (steps 1-2) ─► artefacts/train, artefacts/test
                                                  │
handoff/lgbm_ranker_v1 ─► test candidates (step 3), fit / dev / lockbox split (step 4)
                                                  │
     scores: lgbm_ranker_v1 · xlmr_base_ce_v1 · xlmr_large_ce_v1 · xlmr_large_ce_v2
             xlmr_large_ce_v3_unseen · qwen3_4b_lora_ce · lgbm_ranker_v2
                                                  │
                         LightGBM stacker (steps 5-7)
                         scores + per-S1 / per-candidate competition features
                         → one-to-one assignment + expected-F0.5 set selection
                         India / US predicted; new-country rows from stacker_v1_matches
                                                  │
                                         output/v12_mylgbm
                                                  │
     + pipeline-v2 pairs that were never a candidate, S2/S3 record still free (step 8)
                                                  │
                                         output/v13_union
                                                  │
     + pipeline-v2 pairs that were a candidate but not chosen, S2/S3 record still free,
       stacker probability ≥ 0.3 (step 9)
                                                  │
                                         output/v14_last  =  submitted files
```

## 4. Rebuild the submitted files (CPU, ~1 h)

Setup:

```bash
pip install -r requirements.txt      # Python 3.11+; 16 GB RAM; runs on CPU
```

Copy the stored scores into `code/business_entity_resolution/handoff/` (section 2) if the folder only holds its
`README.md`.

Put the challenge files in `dataset/`:

```
dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

Then run every command from the `code/business_entity_resolution/` folder of the submission (the folder that
contains `src/`, `configs/` and `handoff/`):

```bash
cd <submission root>/code/business_entity_resolution
```


```bash
V12="--extra ce_large=handoff/xlmr_large_ce_v1 --extra ce_full=handoff/xlmr_large_ce_v2 \
     --extra qwen=handoff/qwen3_4b_lora_ce --extra mylgbm=handoff/lgbm_ranker_v2 \
     --fill qwen=ce_full --fill mylgbm=lgbm \
     --eval_fill qwen=ce_full:0.02:0.98 --eval_fill mylgbm=lgbm"
```

| # | command | what it does | time |
|---|---|---|---|
| 1 | `python -m src.run --stage normalize --split train --config configs/full.yaml` | clean train names / addresses | 4 min |
| 2 | `python -m src.run --stage normalize --split test --config configs/full.yaml` | clean test names / addresses | 4 min |
| 3 | `python -m src.restore_cands` | load test candidates from `handoff/lgbm_ranker_v1` | 1 min |
| 4 | `python -m src.splits` | fixed fit / dev / lockbox split of the fold-0 train S1s (70 / 15 / 15, seed 42) | 1 min |
| 5 | `python -m src.loop_eval --tag E000_champion` | stacker without LightGBM v2: fit on *fit*, score on *dev* → 0.989752 (reference) | 10 min |
| 6 | `python -m src.loop_eval --tag E001_mylgbm $V12` | stacker with LightGBM v2 → dev 0.990090 (Δ +0.00034, 95 % CI +0.00020…+0.00046) | 12 min |
| 7 | `python -m src.loop_eval --tag E001_mylgbm $V12 --final output/v12_mylgbm` | refit on all fold-0 S1s, predict test, decide → `output/v12_mylgbm` | 15 min |
| 8 | `python -m src.union_submit --base output/v12_mylgbm --other handoff/lgbm_v2_xlmr_base_v2_matches/matching_results.tsv.gz --out output/v13_union` | + 28,319 pairs (India 14,710, US 4,210, new country 9,399) | 5 min |
| 9 | `python -m src.rescue_submit --base output/v13_union --out output/v14_last` | + 10,779 pairs (India 1,086, US 1,021, new country 8,672) | 3 min |
| 10 | `cmp output/v14_last/matching_results.tsv ../../output/matching_results.tsv && cmp output/v14_last/candidate_pairs.tsv ../../output/candidate_pairs.tsv` | identical to the submitted files | 1 min |

Steps 7–9 end with `utils/validate_submission.py` (PASS). Step 7 saves the stacker's test probabilities to
`artefacts/exp/E001_mylgbm/final_test_probs.parquet`; step 9 reads them.

**Merge rules.**
- *Union* (step 8): a pipeline-v2 pair is added if the main blocking never proposed it and its S2/S3 record is
  not matched to any S1 yet. On labelled training data this pair type is 92.3 % correct, well above the ~70 %
  precision at which adding a pair stops helping F0.5. Added pairs are appended to the candidate list.
- *Rescue* (step 9): a pipeline-v2 pair is added if it is a main-pipeline candidate that was not chosen, its
  S2/S3 record is still free, and the step-7 stacker probability is ≥ 0.3. The candidate list is unchanged.

## 5. How the stored scores were produced

All commands in this section are also run from `code/business_entity_resolution/`.

### `lgbm_ranker_v1` (AWS EC2 g5.2xlarge, NVIDIA A10G GPU, ~5 h)

```bash
python -m src.run --stage normalize --split train --config configs/full.yaml
python -m src.run --stage normalize --split test  --config configs/full.yaml
python -m src.finetune_embed --pairs 400000 --max_steps 2500 --out artefacts/embed_ft
python -m src.run --stage embed    --split train --config configs/full.yaml
python -m src.run --stage embed    --split test  --config configs/full.yaml
python -m src.run --stage block    --split train --config configs/full.yaml
python -m src.run --stage features --split train --config configs/full.yaml
python -m src.run --stage rank     --split train --config configs/full.yaml   # stage 1 → 2, OOF macro F0.5 0.9805
python -m src.run --stage block    --split test  --config configs/full.yaml
python -m src.run --stage features --split test  --config configs/full.yaml
python -m src.run --stage rank     --split test  --config configs/full.yaml
python -m src.run --stage decide   --split test  --config configs/full.yaml
```

### Cross-encoders

All read `business_name | business_address` of the S1 record, then of the candidate, and were trained only on
train S1s outside fold 0, so the fold-0 scores the stacker learns from are out-of-sample.

| folder | model (licence, size) | training | fold-0 AUC |
|---|---|---|---|
| `xlmr_base_ce_v1` | xlm-roberta-base (MIT, 278M) | 3.45M pairs (fold ≠ 0; LightGBM prob ≥ 0.001 + 10 % of the rest), 1 epoch, lr 2e-5, bf16, max_len 128 | 0.99817 |
| `xlmr_large_ce_v1` | xlm-roberta-large (MIT, 560M) | same pairs, lr 1e-5 | 0.99839 |
| `xlmr_large_ce_v2` | from `xlmr_large_ce_v1` | + 4M pairs from all train S1 candidates, lr 5e-6 | 0.99809 |
| `xlmr_large_ce_v3_unseen` | from `xlmr_large_ce_v1` | self-training on new-country test pairs where LightGBM and the cross-encoders agree, 1:1 with labelled replay, lr 5e-6 | 0.99838 |
| `qwen3_4b_lora_ce` | Qwen3-4B-Base (Apache-2.0, 4.02B) + LoRA r=32, sequence-classification head | 1 epoch on 1.2M contested fold ≠ 0 pairs | 0.99844 |

xlm-roberta base / large (`src/ce_rescore.py`: trains on fold ≠ 0 pairs of `handoff/ce_training_pairs`, then
scores fold 0 and all test pairs):

```bash
python -m src.ce_rescore --model FacebookAI/xlm-roberta-base  --ckpt artefacts/ce_base  --out handoff/xlmr_base_ce_v1
python -m src.ce_rescore --model FacebookAI/xlm-roberta-large --lr 1e-5 --ckpt artefacts/ce_large --out handoff/xlmr_large_ce_v1
```

`xlmr_large_ce_v2` and `xlmr_large_ce_v3_unseen` continue from the `xlmr_large_ce_v1` checkpoint
(`artefacts/ce_large/final`):

```bash
python -m src.ce_large_v2            # + full-data candidates (4M pairs), lr 5e-6 -> handoff/xlmr_large_ce_v2
python -m src.ce_large_v3_unseen     # self-training on unseen-country test pairs -> handoff/xlmr_large_ce_v3_unseen
```

`ce_large_v3_unseen` pseudo-labels unseen-country test pairs with three teachers (LightGBM, xlm-r base,
xlm-r large v1): positives where all three are ≥ 0.93 (at most 600k), one-to-one decoys (the pool record is
another S1's positive and this S1 scores it ≥ 0.065), and easy negatives where all three are ≤ 0.22. These are
mixed 1:1 with labelled train pairs. On the submitted data these thresholds reproduce the recorded pseudo-label
counts (`pseudo_stats.json`) within 0.2 %. Both scripts take `--smoke N`, which runs every stage on N pairs as a quick end-to-end check.

Qwen3-4B LoRA (`src/qwen_ce/`, run from `code/business_entity_resolution/`; needs `peft`):

```bash
python src/qwen_ce/prep.py        # training / fold-0 / contested test pairs -> artefacts/qwen_ce/inputs
python src/qwen_ce/qwen_ce.py     # LoRA fine-tune (1 epoch, bs 64, lr 1e-4, seed 7), scores fold 0 + contested test pairs
python -m src.unseen_pairs         # unseen-country test pairs -> artefacts/qwen_ce/inputs/unseen_all.parquet
python src/qwen_ce/qwen_score.py --pairs artefacts/qwen_ce/inputs/unseen_all.parquet   # -> handoff/qwen3_4b_lora_ce/ce_test_unseen_all_*
```

### Pipeline v2 (`src/pipeline_v2/`, notebooks run on Google Colab, GPU and TPU runtimes)

| step | file | output |
|---|---|---|
| 0 | `new_eda.ipynb` | exploratory data analysis; builds `indic_dict.json` (Indic → English word dictionary, 1,302 entries, learned from train pairs), which step 1 reads. The built file is included as `src/pipeline_v2/indic_dict.json` |
| 1 | `er_validation_v4.ipynb` (uses `er_pipeline.py`) | blocking (location partitions, exact keys + TF-IDF kNN), LightGBM stage 1 → candidate expansion → stage 2, 3-fold OOF; final models |
| 2 | `test_v4.ipynb` | test blocking, predictions, decision |
| 3 | `ce_colab.ipynb` | xlm-roberta-base cross-encoder on contested pairs + stacked blend → `handoff/lgbm_v2_xlmr_base_v2_matches/` |
| 4 | `python -m src.export_mylgbm` | LightGBM v2 probability for every stacker pair → `handoff/lgbm_ranker_v2/` |

Set `INPUT_BASE` / `ROOT` in the notebooks' configuration cell to this folder.

### `stacker_v1_matches`

```bash
python -m src.stack_submit --out handoff/stacker_v1_matches --features extra \
  --extra ce_large=handoff/xlmr_large_ce_v1 --extra ce_full=handoff/xlmr_large_ce_v2 \
  --extra qwen=handoff/qwen3_4b_lora_ce --fill qwen=ce_full \
  --swap ce_large=handoff/xlmr_large_ce_v3_unseen --shift unseen:-1 --alpha 1.5 \
  --pairs "handoff/lgbm_ranker_v1/oof_train_full_part*.parquet" \
  --lgbm_test "handoff/lgbm_ranker_v1/probs_model_full_part*.parquet"
```

## 6. Source map (`src/`)

| file | role |
|---|---|
| `run.py`, `reproduce.py` | main-pipeline CLI and stages |
| `normalize.py` | country-agnostic name / address cleaning |
| `finetune_embed.py`, `embed.py` | InfoNCE fine-tuning of multilingual-e5-small; encoding and chunked exact kNN |
| `blocking.py` | per-country rare-token IDF pass ∪ embedding pass, ≤ 25 candidates per S1 |
| `features.py`, `ranker.py` | pair features; LightGBM with S1-grouped folds and stage-2 context model |
| `decide.py` | one-to-one assignment + expected-F0.5 set selection |
| `evaluate.py`, `local_score.py` | exact per-entity F0.5, folds, bootstrap |
| `raw_feats.py`, `stack_eval.py`, `blend_eval.py` | stacker features and evaluation |
| `stack_submit.py` | stacker v1 build (`stacker_v1_matches`) |
| `splits.py`, `loop_eval.py` | fit / dev / lockbox protocol; stacker with LightGBM v2 (steps 5–7) |
| `export_mylgbm.py` | LightGBM v2 probabilities → stacker input |
| `union_submit.py` | step 8 |
| `rescue_submit.py` | step 9 |
| `restore_cands.py` | step 3 |
| `eda.py` | dataset facts |
| `io_utils.py`, `cpus.py`, `no_throttle.py` | I/O helpers, thread settings |
| `ce_rescore.py` | xlm-roberta cross-encoder training + scoring (`xlmr_base_ce_v1`, `xlmr_large_ce_v1`) |
| `ce_large_v2.py`, `ce_large_v3_unseen.py` | continued training of xlm-r large: full data (v2), unseen-country self-training (v3) |
| `unseen_pairs.py` | test candidate pairs of S1s whose country is absent from training |
| `qwen_ce/` | Qwen3-4B LoRA cross-encoder: `prep.py` (inputs), `prompts.py` (pair → tokens), `qwen_ce.py` (train + score), `qwen_score.py` (score any pair list) |
| `pipeline_v2/` | pipeline v2: `er_pipeline.py` + notebooks |

## 7. Determinism

Seeds are fixed (42) and folds are a crc32 hash of the S1 id, so section 4 rebuilds the submitted files exactly.
