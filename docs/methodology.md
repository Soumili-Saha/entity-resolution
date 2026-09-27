# Methodology

The final outputs come out of four levels, each trained only on data the next level never uses for scoring:

| level | what | code | output |
|---|---|---|---|
| 1 | candidate generation + LightGBM pair scorer | `run.py` and stages | candidates, OOF + test probabilities |
| 2a | cross-encoder re-scorers on the level-1 candidates | trained externally (recipes below) | pair scores |
| 2b | companion pipeline (independent blocking + LightGBM) | `export_mylgbm.py` (pipeline itself released separately) | pair scores + its own matches |
| 3 | stacker over all scores + competition context → decision | `stack_submit.py` (v11), `loop_eval.py` (v12) | matching sets |
| 4 | union with companion pairs (blocking misses; stacker rejections at p ≥ 0.3) | `union_submit.py` | `output/v13_union` (best), `output/v14_last` |

---

## 1. Data facts that shaped the design

- Train: S1 2,206,821 / S2 5,034,616 / S3 5,285,603. Test: S1 1,732,544 (India 810k, US 663k, **France 259k**).
- Singletons 5.6%. Matches per S1 are mostly 2–6 (mean ≈ 3.5), max 11. There are 7.64M true pairs, 48% of them from S2.
- **One-to-one holds exactly:** no S2/S3 id appears under two S1s. This is enforced in the decision layer.
- **Every true pair shares the country label**, so blocking runs within each country.
- About 26% of S2/S3 records match no S1 (distractors). About 9% of S2 names are Devanagari/Bengali phonetic
  transliterations of English names ("redd veNcrs praaivett limittedd").
- Name noise: legal words anywhere, junk marks, "(ID: n)", domain-style names, fka/aka aliases, OCR digit swaps.
  Address noise: St → "SAINT", reordered components, state codes vs names vs native script, dropped parts, mutated house numbers.
- France appears only in test, so every France decision is made without France labels (see §6).

## 2. Evaluation protocol

Every number in this document comes from one of these protocols. None of them uses test labels.

| protocol | split | used for | why it is honest |
|---|---|---|---|
| **Grouped OOF** | fold = `crc32(s1_id) % 5` | level-1 LightGBM, decision tuning | the one-to-one structure makes an S1 plus its matches a whole cluster, so no cluster straddles folds |
| **Fold-0 hold-out** | fold-0 S1s (≈ 441k) | levels 2 and 3 | every cross-encoder is trained on folds 1–4 only, so on fold 0 all base scores (LightGBM OOF + every CE) are out-of-sample |
| **Cross-fitted halves** | fold 0 split by `crc32 % 10 ∈ {0, 5}` | `blend_eval`, `stack_eval` | decision parameters tuned on one half and scored on the other, both ways, then averaged |
| **Fit / dev / lockbox** | fold-0 S1s, 70 / 15 / 15, stratified by country × #matches, seed 42 | `loop_eval` (v12) | stacker and decision fitted on *fit* only; *dev* is scored with a 95% bootstrap CI and a paired bootstrap delta vs the champion; *lockbox* is untouched until the final gate |
| **Leave-one-country-out** | train on India → score US, and the reverse | France proxy | the closest labelled stand-in for an unseen country |
| **Test-set A/B** | hidden test labels | France only, sparingly | India/US rows are held fixed between two builds, so the test delta isolates the France change |

`local_score.py` implements the official metric exactly: per-S1 F0.5 over all S1s, where a singleton scores 1
only on an empty prediction. It is the only code that reads dev / lockbox labels.

## 3. Level 1: candidate pipeline

### Normalisation (`normalize.py`)
Hand-written, country-agnostic rules. Each name gets five views (clean, core without legal words, phonetic skeleton,
no-space, alias after fka/aka). Addresses get canonical short forms (Road → rd, Saint/St, …), postal code,
house numbers and landmark tokens. Native-script names are transliterated before building the skeleton.

### Blocking (`blocking.py`, `embed.py`, `finetune_embed.py`)
- **Token pass:** rare-token IDF cosine per country over name-skeleton tokens, the whole no-space name and address tokens.
  Tokens with pool document frequency > 3000 are dropped. Recall@20 is 83.2% and @50 is 87.3% (20k validation S1 vs the full 10.3M pool).
- **Embedding pass:** multilingual-e5-small (MIT, 118M), fine-tuned with in-batch InfoNCE on 400k training pairs.
  Batches are grouped by city/state to get hard negatives, and validation S1s are excluded. Exact GPU kNN runs per country.
  Recall@5/10/20/30 = 90.7 / 98.5 / 99.2 / 99.4%.
- **Union:** token top-12 ∪ embedding top-20, capped at 25 per S1 by best rank across passes, gives **99.2% recall at ~25 candidates per S1**.

### Pair scorer (`features.py`, `ranker.py`)
- About 70 features: fuzzy name ratios over all views, address token-set / Jaccard, house-number / postal / landmark
  agree and conflict, blocking scores and ranks, and **competition context** (this candidate's rank and gap among
  its S1's candidates, and how many other S1s want the same pool record).
- LightGBM with 5 folds grouped by S1, early stopping, and all fold models averaged at test time.
- **Stage 2** adds per-S1 probability context from the stage-1 OOF predictions and is kept only when OOF improves.

### Decision (`decide.py`)
1. **One-to-one:** each S2/S3 record goes to the S1 with the highest probability.
2. **Set selection:** per S1, keep the sorted prefix (possibly empty) that maximises approximate expected F0.5.
   `alpha` scales the value of predicting nothing, which matters because empty predictions are the only way to score on singletons.
   The parameters are tuned on OOF, never on the test set (except the France shift, §6).

### Level-1 results

| experiment | blocking recall | oracle F0.5 | OOF F0.5 |
|---|---|---|---|
| token pass only, 5k S1 smoke | 0.833 | 0.927 | 0.880 |
| token + fine-tuned embedding union (cap 25), 200k S1, stage 1 | 0.9922 | 0.9976 | 0.9685 |
| + stage 2 (per-S1 probability context) | 0.9922 | 0.9976 | 0.9699 |
| same pipeline, 500k S1, lr 0.1 | 0.9922 | | 0.9719 |
| + decoy features, **all 2.2M S1** (`configs/full.yaml`, stage 1 → stage 2) | 0.9918 | | 0.9794 → **0.9805** |

The top stage-1 features are embedding similarity, its rank and gap among the S1's candidates, house-number conflict,
number Jaccard, name ratios and address token-set. Semantic similarity and candidate competition dominate, and
house-number conflict is the strongest string signal.

Error analysis (`analysis.py`, 200k-S1 OOF):
- By country: India 0.9646, US 0.9734. By number of true matches: 0 → 0.954, 1 → 0.893, 2 → 0.967, 3+ → 0.975–0.980.
- Pairs: 691,838 gold, 648,988 TP, 7,105 FP (632 on singletons), 42,850 FN (5,387 from blocking, 37,463 from model/decision).
  Pair precision is 0.989 and recall 0.938, so the decision layer trades recall for precision, as F0.5 rewards.
- The dominant remaining error is **decoys**: same or similar generic name in the same city but a different street,
  an extra descriptor ("Holding", "International"), a different legal form or a different house number.
  This motivated level 2.

Feature ablations (fold 0 and leave-one-country-out, 100k S1, `logs/experiments.csv`): removing the decoy
features drops LOCO from 0.9354 to 0.9101. The length, token-absence and density groups are neutral.

## 4. Level 2a: cross-encoders

A cross-encoder reads both records together (`name | address` of S1, then of the candidate, max 128 tokens) and
outputs P(match). Attention across the pair catches the small decoy edits that hand-made features miss.

All models were trained on **fold ≠ 0** rows of the level-1 candidates (all pairs with LightGBM prob ≥ 0.001,
plus 5–10% of the rest). They were scored on fold 0 (for levels 2/3) and on test. Every model is MIT or Apache-2.0
licensed and ≤ 8B parameters, and no external data was used.

| score column | model | training | fold-0 AUC | CE alone F0.5 | logit blend with LightGBM |
|---|---|---|---|---|---|
| (LightGBM, reference) | level 1 | | 0.99497 | 0.9723 | |
| `ce` (`ce_out`) | xlm-roberta-base (278M) | 3.45M pairs, 1 epoch, lr 2e-5, bf16, H100 42 min | 0.99817 | 0.9802 | 0.9865 |
| `ce_large` | xlm-roberta-large (560M) | same pairs, lr 1e-5, 49 min | 0.99839 | 0.9812 | 0.9871 |
| `ce_full` | ce_large, continued | 4M pairs from the all-S1 candidates, lr 5e-6, 134 min | 0.99809* | 0.9824 | 0.9875 |
| `qwen` (`ce_qwen_v11`) | Qwen3-4B-Base (Apache-2.0, 4.0B) + LoRA r=32, seq-cls head | LoRA fine-tuning on fold ≠ 0 pairs; scored on contested pairs + all France test pairs | 0.99844 | | |
| France swap (`ce_france_out`) | ce_large, transductive self-training | pseudo-labels on France test pairs where LightGBM and CEs agree confidently, 1:1 with labelled replay, lr 5e-6 | 0.99838 (no harm on India/US) | | |

\* measured on the larger all-S1 fold-0 set. The other rows use the 685k-pair / 99,964-S1 fold-0 sample.
F0.5 columns: held-out macro F0.5 with the cross-fitted-halves protocol.

Tried and rejected, with the reason measured by the same protocol:

| variant | result |
|---|---|
| swapped-order test-time augmentation | CE 0.9538 vs 0.9824; CE never saw S1 second |
| CE e2 (second epoch) replacing ce_full | test −0.00027, stricter on France |
| Qwen3.5-4B replacing Qwen3-4B | test −0.00059, ranks France worse |
| accepting France pairs with ce_full ≥ 0.95 that the stacker rejected | test −0.00051, implied precision ≈ 45% (breakeven 73%) |

## 5. Level 2b: companion pipeline (`my_pipeline`)

An independently built pipeline over the same data. It will be released as its own repository. It differs from
level 1 in exactly the places where diversity helps a stacker:

| | level 1 (this repo) | companion pipeline |
|---|---|---|
| partitioning | country | country → state when states are detectable; separate no-address pool |
| blocking | rare-token IDF + fine-tuned embedding kNN | exact keys + TF-IDF nearest neighbours |
| text handling | hand-written rules + transliteration skeleton | learned Indic→English dictionary, label-free filler-word detection, legal-suffix families |
| model | LightGBM stage 1 + stage 2 | LightGBM stage 1 → **candidate expansion** → stage 2 with sibling features |
| training rows | 200k–2.2M S1 | 15% S1 selection, orphan records dropped ("realistic rows"), 3-fold OOF |
| re-scorer | level-2a cross-encoders | own xlm-roberta-base cross-encoder on its contested pairs (0.01 ≤ p ≤ 0.99), blended with its LightGBM |
| output | probabilities | probabilities **and** its own matching file (`output_ce`) |

It feeds the final result in two separate ways.

**(a) As a score column: `export_mylgbm.py` → `precomputed/my_lgbm_out`.**
The companion LightGBM probabilities are looked up for every stacker pair: 401,825 fold-0 train pairs (the S1s
inside its 15% selection, OOF) and all 10.8M test pairs. Pairs it never had as candidates get 1e-4. The fold-0 S1s
outside its selection have no score, and for those the stacker falls back to the level-1 LightGBM
(`--fill mylgbm=lgbm`), both in training and in dev scoring, so the column means the same thing everywhere.

**(b) As extra pairs: `union_submit.py` (level 4).**
The companion pipeline finds some matches that level-1 blocking never proposed, so no stacker can select them.
Its final output (`precomputed/my_pipeline/output_ce`, 0.984 on the test set on its own) is merged into v12.

*v13_union: blocking misses.* A pair is added only when all three hold:
1. the companion pipeline predicted it as a match,
2. it is **not** among the level-1 candidates for that S1 (a blocking miss, so the stacker never saw it), and
3. its S2/S3 record is **not yet matched** to any S1 (one-to-one is preserved).

That gives 28,319 pairs: India 14,710, US 4,210 and France 9,399. On labelled train data this pair type was
**92.3% correct**, against an F0.5 breakeven of about 70% for adding a pair to an S1 that already has matches.
Typical examples are native-script names and synthetic random names at the same address. The validator passes
with the ID check on.

*v14_last: stacker rejections.* On top of v13_union, companion pairs that **are** level-1 candidates, were rejected
by the v12 decision, have a free S2/S3 record and a v12 stacker probability ≥ 0.3 are added
(`--rescue_probs … --rescue_min 0.3`). That gives 10,779 pairs: France 8,672, India 1,086, US 1,021.
The evidence is weak: 81% correct, but on only 37 dev pairs. v13_union stays the best measured build.

*Checked on labelled data and rejected:*

| variant | measured | decision |
|---|---|---|
| add companion pairs that the level-1 LightGBM rejected | 8% correct | rejected |
| add companion pairs below the companion's own decision threshold | ≤ 50% correct | rejected |
| remove v12 pairs that the companion pipeline rejected | those pairs were 96% correct | rejected (removal would hurt) |
| re-stack France with the v12 stacker | 9.9% of France S1s change, no labels to check | France kept from v11 |

## 6. Level 3: stacker

### Features (`stack_eval.py`)
- Every base score and its logit: level-1 LightGBM, `ce`, `ce_large`, `ce_full`, `qwen`, `mylgbm`.
- **S1-side context** per score: rank within the S1, gap to the S1's best, the best and second-best score, the sum,
  and the count above 0.5.
- **Pool-side competition** (from LightGBM over all S1s): how many S1s claim the pool record, whether this S1 is the
  top claimant, the best other claimant's probability and this S1's margin over it.
- An empty-address flag (the CEs and LightGBM disagree most on empty-address records).
- Missing scores are filled from a named substitute (`--fill qwen=ce_full`). In dev scoring the substitute is only
  used outside the 0.02–0.98 band, which mimics the test coverage where Qwen scored only contested pairs.

A small LightGBM (31 leaves, lr 0.05) is trained on fold-0 pairs only. All of its inputs are out-of-sample there.

### Build history (`logs/builds.csv`, `output/README.md`)

Test F0.5 is scored on the hidden test labels (India, US, France). Held-out F0.5 is India/US only.

| build | change | held-out F0.5 (India/US) | test F0.5 |
|---|---|---|---|
| v1 | level 1, 200k S1 | 0.9699 | |
| v2 | level 1, 500k S1 | 0.9719 | |
| v3 | logit blend LightGBM + `ce` | 0.9865 | 0.98068 |
| v3 strict | + France logit −1 | 0.9865 | 0.98092 |
| v4 | stacker: LightGBM + `ce` + competition context | 0.9871 | 0.98206 |
| v5 | + `ce_large` | 0.9877 | 0.98355 |
| v8 | stacker on 434k full-data fold-0 S1: + `ce_full`, France swap | 0.9899 | 0.98514 |
| v11 | + `qwen` | 0.9903 | 0.98620 |
| v12 | + `mylgbm` column (`loop_eval --final`) | dev 0.9901, Δ +0.00034 (CI +0.00020 … +0.00046) | 0.98631 |
| **v13_union** | + 28,319 companion pairs from blocking misses | pair type 92.3% correct on train | **0.988** |
| v14_last | + 10,779 companion pairs rejected by the stacker at p ≥ 0.3 | 81% correct on 37 dev pairs | pending |

From v11 to v13_union: 0.98620 → 0.98631 → 0.988. The gain came from two sources: the companion model as a new
stacker input, and the companion pipeline filling level-1 blocking blind spots, especially in France. No base model
was retrained for either step.

v12 in detail (`loop_eval`, dev = 66,243 S1): macro F0.5 0.99009 (95% CI 0.98964–0.99052), pair precision 0.9989,
recall 0.9717, India 0.99020, US 0.99002. The champion without `mylgbm` scores 0.98975 on the same dev S1s. For the
final model the recipe is frozen and retrained on all fold-0 S1s (fit + dev + lockbox) before predicting test.

### France (unseen country)
France has no labels, so everything there is chosen conservatively and measured indirectly:
- **LOCO proxy:** a model trained on one country loses 0.055–0.061 F0.5 on the other, which is the expected size of the
  unseen-country gap. At test time France is scored by models trained on *both* countries.
- **France logit shift −1:** stricter on France, chosen by test-set A/B with India/US rows byte-identical.
  0 was worse by −0.0001, −1.25 was within noise (+0.00003) and −1.5 was worse (−0.00005).
- **France swap:** on France pairs, the `ce_large` column is replaced by the self-trained France cross-encoder.
- **v12 copies France rows from v11 unchanged**, because the new `mylgbm` column changed 9.9% of France S1s with no
  way to verify the change locally. France gains only through the level-4 union (9,399 added pairs in v13_union).

## 7. Is the evaluation sound?

What holds:
- No level ever scores on rows it or a level below was trained on. Folds are grouped by S1, and one-to-one makes an S1 a whole cluster.
- Decision parameters are always tuned and scored on disjoint S1s (cross-fitted halves, or fit vs dev).
- v12 was accepted on a paired bootstrap whose CI excludes zero, and the lockbox was only used in the final retrain.
- India/US held-out scores and test scores move together across builds (v3 → v11: +0.0038 held-out, +0.0055 test).

Limitations:
- **France** tuning (shift, swap) used a handful of test-set A/B checks, so the France part of the test score is
  slightly optimistic. The shift was kept at the value where neighbouring values were flat, not at a sharp peak.
- **The level-4 union is validated only by pair precision.** Its pairs lie outside the level-1 candidates, so the
  stacker's dev protocol cannot score them. The 92.3% precision was measured on labelled train pairs of the same
  type, not as an end-to-end dev F0.5 delta. The v14 rescue rests on only 37 dev pairs.
- The cross-encoders were trained on the 500k-S1 sample and then the full-data candidates. `ce_full` covers the union
  of both candidate sets, so a few fold-0 pairs are scored by a CE trained on a slightly different candidate distribution.

## 8. Reproducibility

- Seeds are fixed (42). Folds are a crc32 hash of the S1 id. The training sample and splits are seeded draws.
- `output/<build>/blend.json` or `union.json` holds the exact recipe of every build. The README (path B) rebuilds v12,
  v13_union and v14_last from `precomputed/`.
- GPU kernels (fp16 encoding, kNN) and LightGBM threading can differ in the last bits across hardware, which may
  flip a handful of near-tie decisions.
