# Submitted files

The two submitted files are stored compressed because of GitHub's 100 MB file limit. To restore them
byte-exact:

```bash
gunzip -c matching_results.tsv.gz > matching_results.tsv
cat candidate_pairs.tsv.gz.part00 candidate_pairs.tsv.gz.part01 candidate_pairs.tsv.gz.part02 | gunzip > candidate_pairs.tsv
```
