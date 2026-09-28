# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** HACKANESH  
**Team Members:** Soumili Saha (team leader), Soumyajit Pramanik, Supratik Das, Sougata Daw  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We match each Source 1 (S1) business to its Source 2 / Source 3 records with two independently designed
pipelines. Each pipeline has its own blocking, a two-stage LightGBM and transformer cross-encoders. A
LightGBM stacker combines all scores, and a decision rule picks, for each S1, the match set with the highest
expected F0.5 under a one-to-one constraint. Merging the two pipelines lifted the public score from 0.9862
(one pipeline) to **0.988**. Held-out macro F0.5 is 0.990.

---

## 2. Methodology

### 2.1 Problem Analysis
These findings come from our exploratory data analysis. The full analysis, with the tables, examples and
measurements behind every point, is in
**`code/business_entity_resolution/src/pipeline_v2/new_eda.ipynb`**.
Short dataset facts are also produced by `src/eda.py`.

**Size and structure**
- **Size:** train S1 2.2M, S2 5.0M, S3 5.3M; test S1 1.7M, S2 4.9M, S3 5.1M. Even inside one country there
  are trillions of possible pairs (US alone ≈ 1.32M × 6.19M ≈ 8 trillion), so blocking is required.
- **Country is a perfect hard block:** a true match never crosses country labels, so records are only compared
  within the same country label, whatever labels appear.
- **Several matches per S1:** only 5.6 % of S1 records have no match; most have 2–6 (≈ 3.5 on average, about
  half from S2 and half from S3). Each source holds several copies of the same business.
- **One-to-one:** every S2/S3 record belongs to at most one S1 (0 violations). At prediction time a record that
  fits two S1s is given only to the best one, a strong precision tool for F0.5.
- **Look-alikes:** ≈ 26 % of S2/S3 records (≈ 1.3M per source) match nothing. They are clean records of other,
  similar businesses and are the main source of false matches.
- **Test differs from train:** test contains a new test country with no training labels (≈ 15 % of test
  S1s, ≈ 14.5 % of test S2/S3) and new vocabulary (retail words such as stores, jewellers, pharmacy that never
  appear in train). Everything must therefore be country- and vocabulary-agnostic.

**Where the noise is**
- **S1 is clean, S2/S3 carry the noise:** S1 has no native script, junk, domain names or all-caps names. The task
  is to bring S2/S3 back to S1's form.
- **Noise was added to matched records, not unmatched ones:** e.g. empty address 4–5 % (matched) vs 0.2–0.4 %
  (unmatched), domain as name 4.5–5.9 % vs 0.5–0.7 %. So the amount of noise is itself a useful signal, and
  blocking needs a name-only path (≈ 5 % of matches have no address).
- **Postal codes are almost never present:** US zip in ≈ 10 % of records, India PIN in ≈ 1 %, the new test
  country ≈ 0.4 %. They can only be a bonus, not a blocking key.
- **Native script is a major gap for India:** ≈ 23 % of India S2 names (≈ 13 % of S3) and ≈ 23 % of addresses are
  in Indic scripts; without transliteration only 6.4 % of these pairs share a word. A dictionary learned from
  train pairs (1,302 words, `indic_dict.json`) recovers them, with a rule-based skeleton fallback (83–86 %) for
  unseen words.

**Name noise**
- Legal suffixes dropped, added, moved or swapped between types (pc→inc, lp→llc, limited→ltd), including
  dotted forms (L.L.C.) and native-script forms.
- Injected filler words (center, services, partners; location prefixes such as downtown, riverside), found
  automatically: a word far more common in S2/S3 than in S1 is an injected filler. This works for any language.
- OCR-style swaps (hea1th, 8low, rn↔m, i↔l), heavy multi-letter typos, injected accents, shuffled or dropped
  words, honorifics (M/s, Sri, Mr), alias markers (dba, aka, formerly).
- Glued / domain-style names (parkerrestaurant.com) and fully random names built from a fixed set of syllables
  (Fayepyra, Zetavio). No name rule can catch those; only the address can.

**Address noise**
- Abbreviations in both directions (Road↔Rd, Street↔St↔Saint), numbers as words, state code vs. full name vs.
  native script, city aliases (Bombay↔Mumbai, Gurgaon↔Gurugram), reordered components, junk (PO Box, PMB, null).
- House numbers corrupted in a specific way, mostly one digit deleted or changed (707→70, 814→813), so exact
  number keys are unsafe and a "delete one digit" key is used.
- Only ≈ 13 % of true address pairs have the same word set, so blocking uses address parts (state, locality,
  numbers), not the whole address. State agrees in 99.8 %+ of pairs and is a safe partition for the US and India.

**What this meant for the design**
- Blocking needs two independent paths, name and address, because some matches have a random name but the
  right address and others the right name but no address.
- Names in the new test country and Indic names are built from a small common vocabulary, so a single name
  word is never a good key; keys must combine name and location.
- Exact keys give high recall but far too many candidates (≈ 280 per S1), so candidates are ranked by a
  similarity score and only the top K per S1 are kept.
- The hardest negatives are near-duplicates (same name and street, different house number) and true copies of
  another S1.

### 2.2 Solution Strategy
Two pipelines generate and score candidates independently. Their results are then merged:

```
                test S1 + S2/S3 records
                 │                     │
      Pipeline 1 blocking      Pipeline 2 blocking
      (IDF + e5 kNN)           (keys + TF-IDF kNN + expansion)
                 │                     │
      LightGBM (2 stages)      LightGBM (2 stages)
      + 5 cross-encoders       + 1 cross-encoder
                 │                     │
                 └──► stacker ◄── Pipeline 2 score
                         │
          one-to-one + expected-F0.5 decision
                         │
     + Pipeline 2 pairs never found by Pipeline 1        (union)
     + Pipeline 2 pairs rejected by Pipeline 1, p ≥ 0.3  (rescue)
                         │
                 matching_results.tsv
```

**Approach Type:** Hybrid: blocking + gradient-boosted classifier + transformer cross-encoders + stacking,
with two independent pipelines.  
**Core Innovation:**
- Two complementary blocking designs: a semantic one (fine-tuned embeddings) and a structural one (exact
  keys, location partitions, expansion).
- A stacker that sees each pair *in competition*: its rank among the S1's candidates, and how many S1s want
  the same record.
- A decision rule that directly maximises expected F0.5 under the one-to-one constraint.

---

## 3. Candidate Generation (Blocking)
Both pipelines first normalise text: transliteration to Latin, lowercasing, removal of legal words and junk
tokens, alias splitting, and short canonical forms for address words (road→rd, street→st, avenue→ave, …).
Blocking always runs within one country.

**Pipeline 1**

| Pass | Method | Kept per S1 |
|---|---|---|
| Rare-token search | IDF-weighted cosine on phonetic name tokens, the space-free name and address tokens; very common tokens dropped | 12 |
| Embedding search | `multilingual-e5-small` fine-tuned on 400k training pairs (contrastive loss, hard negatives from the same city); exact nearest neighbours on `name \| address` | 20 |
| Union | both lists merged, ranked by best rank | **25** |

**Pipeline 2**

| Step | Method | Kept per S1 |
|---|---|---|
| Partitions | country → state when a state can be detected; records without a usable address go to a per-country "no-address" pool | n/a |
| Exact keys | shared name / address keys within the partition (keys covering > 200 records skipped) | n/a |
| TF-IDF kNN | character 3-gram name vectors + address-token vectors | 20 (address) / 10 (no-address) |
| Rank and cut | keys ∪ kNN, ranked by name + address similarity | **30** + **5** |
| Expansion | after the first LightGBM, confident matches seed exact name / glued-name / full-address keys; records sharing a key become new candidates | added |

- **Blocking keys used:** phonetic name-skeleton tokens, space-free name, address tokens (IDF-weighted);
  fine-tuned multilingual-e5 embeddings of `name | address` (Pipeline 1). Exact name / address keys,
  character 3-gram TF-IDF, location partitions and expansion keys from confident matches (Pipeline 2).
- **Candidate pairs generated:** **42,817,126** test pairs in `candidate_pairs.tsv` (24.7 per S1): Pipeline 1's
  42.79M plus the Pipeline 2 pairs added by the union step. That is a reduction of ≈ 99.9994 % against all
  same-country pairs.
- **How you ensured true matches were not lost:**
  - Recall was measured on held-out S1s against the **full** 10.3M-record pool after every change:

    | Pipeline 1 pass | recall @10 | @20 | @30 |
    |---|---|---|---|
    | rare tokens | 78.6 % | 83.2 % | 85.3 % |
    | fine-tuned e5 | 98.5 % | 99.2 % | 99.4 % |
    | union | 98.8 % | 99.4 % | 99.5 % |

  - Final Pipeline 1 recall on all training S1s: **99.0 % (India), 99.3 % (US)**. The new test country
    receives the same candidate volume (24.9 per S1).
  - Pipeline 2 recovers pairs outside Pipeline 1's candidates: 28,319 of its final pairs were never
    Pipeline 1 candidates. On labelled training data this pair type is correct 92.3 % of the time.

---

## 4. Matching Model

**Features used:**
- **Name features:** Levenshtein ratio, partial ratio, token-sort / token-set ratio, Jaro-Winkler,
  Jaccard / overlap on phonetic skeletons, space-free name similarity, exact-core-name match, alias
  similarity, legal-form conflicts, synthetic-name and glued-name detectors.
- **Address features:** token ratio / partial / token-set, token Jaccard / overlap, house-number equality and
  conflict, number-set overlap, postal-code match, landmark overlap, exact-address match, missing flags.
- **Other:** blocking scores and ranks; near-duplicate signals (same name and street, different number);
  competition context (rank and gap to the best among the S1's candidates; how many S1s compete for the same
  S2/S3 record); source (S2 / S3). Country is never used as a feature.

**Model type:**

| Stage | Pipeline 1 | Pipeline 2 |
|---|---|---|
| Pair classifier | LightGBM, 73 features, stage 1 → stage 2 (adds per-S1 probability context), 5 folds grouped by S1, all 2.2M training S1s | LightGBM stage 1 → expansion → stage 2, 3 folds grouped by S1 |
| Cross-encoders (read both records together) | xlm-roberta-base; xlm-roberta-large v1, v2 (full data), v3 (self-trained on unlabelled records of test countries absent from training); Qwen3-4B-Base + LoRA | xlm-roberta-base |
| Combiner | LightGBM stacker, 65 features (scores, logits, per-S1 and per-record competition) + Pipeline 2's LightGBM score | small LightGBM stacker |

**Threshold selection method:** instead of one global threshold, the decision works per S1:
1. **One-to-one:** each S2/S3 record is kept only for its highest-probability S1.
2. **Expected-F0.5 set selection:** sort the S1's candidates by probability and choose the prefix (possibly
   empty) with the highest expected F0.5. The single parameter α = 1.5 was tuned on held-out training S1s.
3. **Countries absent from training:** stacker logits are shifted by −1 before the decision. The shift was
   chosen on a proxy experiment (train on India only, treat the US as unseen), where the stacker was
   over-confident on the unseen country.
4. **Merge thresholds:** a rejected Pipeline 2 pair is added back only if its stacker probability is ≥ 0.3
   and its S2/S3 record is still free.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.990** on held-out training S1s never seen by any model (India 0.9906, US
  0.9900). Public leaderboard: **0.988**.
- **Common false positives (wrong merges):** near-duplicates with the same or similar name at the same
  street and a slightly different house number (`18730 Little Lane` vs `1873 Little Ln`); generic names in
  another city; a different business at the exact same address; name-only records with an empty address.
- **Common false negatives (missed matches):** true matches whose house number was also changed (`14298`
  vs `14306`); a different trade name at the same address; empty addresses with noisy names; native-script
  names. F0.5 favours precision, so the decision accepts some of these.

---

## 6. Conclusion
A fine-tuned embedding search keeps 99 % of true matches in ~25 candidates per S1. LightGBM, cross-encoders
and a competition-aware stacker then reach 0.990 held-out macro F0.5. After that, the largest gain came from
merging a second, independently designed pipeline (0.9862 → 0.988), because its structural blocking finds
pairs the embedding search misses. Key lessons: measure blocking recall against the full pool, use the
one-to-one structure in the decision, and validate unseen-country handling on a leave-one-country-out proxy.

---

## Appendix

### A. Code Artefacts
All code is in `code/business_entity_resolution/`. `README.md` has the full step-by-step commands, and
`requirements.txt` the pinned packages.

```
code/business_entity_resolution/
├── src/                  Pipeline 1, cross-encoders (ce_rescore.py, qwen_ce/), stacker and merge steps
│   └── pipeline_v2/      Pipeline 2 (er_pipeline.py + notebooks)
├── handoff/              stored scores of each model (in the GitHub repository, see README section 2)
├── configs/              run settings (full.yaml)
├── utils/                submission validator
└── dataset/              place the challenge TSV files here
```

**Compute:** AWS EC2 g5.2xlarge (NVIDIA A10G) and Google Colab (GPU and TPU).

Entry points that rebuild `output/matching_results.tsv` and `output/candidate_pairs.tsv` (~1 h on the AWS EC2 instance).
Run them from the `code/business_entity_resolution/` folder:

| Step | Command |
|---|---|
| Normalise | `python -m src.run --stage normalize --split train\|test --config configs/full.yaml` |
| Candidates + split | `python -m src.restore_cands`, `python -m src.splits` |
| Stacker (with Pipeline 2 score) | `python -m src.loop_eval --tag E001_mylgbm … --final output/v12_mylgbm` |
| Union | `python -m src.union_submit --base output/v12_mylgbm … --out output/v13_union` |
| Rescue (final) | `python -m src.rescue_submit --base output/v13_union --out output/v14_last` |

Pipeline 1 from scratch (AWS EC2 g5.2xlarge, NVIDIA A10G GPU): `python -m src.run --stage <normalize|embed|block|features|rank|decide>`.

### B. Additional Results

**Progression**

| Build | Held-out F0.5 | Public LB |
|---|---|---|
| Pipeline 1: LightGBM + xlm-r-base | 0.9865 | 0.9807 |
| + stacker | 0.9871 | 0.9821 |
| + xlm-r-large (v1, v2, v3) | 0.9900 | 0.9851 |
| + Qwen3-4B (Pipeline 1 final) | 0.9903 | 0.9862 |
| + Pipeline 2 LightGBM score in the stacker | 0.9901 (dev split; +0.00034 vs. same split without it) | 0.9863 |
| + union with Pipeline 2 | n/a | **0.988** |
| + rescue (submitted) | n/a | final |

**Comparison of the two pipelines**

| | Pipeline 1 | Pipeline 2 |
|---|---|---|
| Blocking | rare-token IDF ∪ fine-tuned e5 kNN | location partitions, exact keys ∪ char TF-IDF kNN, expansion |
| Candidates per S1 | ≤ 25 | ≤ 30 (address) + ≤ 5 (no-address) + expansion |
| Pair model | LightGBM, 73 features, 2 stages | LightGBM, own features, 2 stages + expansion |
| Cross-encoders | xlm-r-base, 3 × xlm-r-large, Qwen3-4B LoRA | xlm-r-base |
| Parameters | ≈ 6.16B | ≈ 0.28B |
| Public LB alone | 0.9862 | 0.984 |
| Strength | accurate scores for the candidates it finds | finds different pairs (exact keys, expansion) |
| Role in final file | base matches + stacker | stacker input, +28,319 union pairs, +10,779 rescue pairs |

**Tried and dropped:** Qwen3.5-4B, averaging two Qwen adapters, domain-adversarial training, other
rerankers (Qwen3-Reranker-4B, bge-reranker-v2-m3, xlm-roberta-xl), synthetic training pairs and extra text
rules. None improved held-out or leaderboard scores.

**Models and licences:** multilingual-e5-small 118M, xlm-roberta-base 278M (×2, one per pipeline),
xlm-roberta-large 560M (×3), Qwen3-4B-Base 4.02B + LoRA 0.07B; total ≈ **6.44B** parameters (< 8B).
MIT (e5, XLM-RoBERTa, LightGBM) and Apache-2.0 (Qwen3). No external data, APIs or lookups.

---

