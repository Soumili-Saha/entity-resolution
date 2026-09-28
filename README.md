# Business Entity Resolution

Matches every Source 1 business record to its Source 2 / Source 3 records with two independent pipelines
(blocking, LightGBM, transformer cross-encoders), a LightGBM stacker and an expected-F0.5 decision under a
one-to-one constraint.

| path | contents |
|---|---|
| [`code/business_entity_resolution/`](code/business_entity_resolution/) | all code, configs, stored model scores (`handoff/`) and the **step-by-step README** |
| [`output/`](output/) | the submitted `matching_results.tsv` and `candidate_pairs.tsv` (compressed; see its README) |
| [`Documentation_template.md`](Documentation_template.md) | method, results and error analysis |

Start with [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md): section 4
rebuilds the submitted files byte for byte from the stored scores (CPU, ~1 h).
