# Dataset

The source data is not redistributed in this repository. Place the files as follows (paths are set in
`configs/*.yaml`):

```
dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

| file | columns |
|---|---|
| `*_source{1,2,3}.tsv` | `entity_id`, `business_name`, `business_address`, `country` |
| `train_ground_truth.tsv` | `source1_entity_id`, `matched_entity_ids` (comma-separated S2/S3 ids, may be empty) |

Read every file with `sep="\t"`, `dtype=str`, `keep_default_na=False` (see `entity_matcher/io_utils.py`).
