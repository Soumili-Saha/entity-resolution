"""Local scoring: exact per-entity F0.5, macro-averaged over all S1 (singletons included)."""
import numpy as np


def f05(pred, gold):
    """Official per-entity score: empty/empty = 1, anything on a singleton = 0."""
    pred, gold = set(pred), set(gold)
    if not gold:
        return 1.0 if not pred else 0.0
    tp = len(pred & gold)
    return 0.0 if tp == 0 else 1.25 * tp / (0.25 * len(gold) + len(pred))


def macro_f05(pred, gold, s1_ids=None):
    """Mean f05 over s1_ids (default: every S1 in gold). pred/gold: {s1_id: set(ids)}."""
    ids = list(gold) if s1_ids is None else s1_ids
    return float(np.mean([f05(pred.get(i, ()), gold.get(i, ())) for i in ids]))


def blocking_recall(candidates, gold, s1_ids=None):
    """Share of true matches that appear in the candidate set (the recall ceiling)."""
    ids = list(gold) if s1_ids is None else s1_ids
    tot = sum(len(gold.get(i, ())) for i in ids)
    hit = sum(len(gold.get(i, set()) & set(candidates.get(i, ()))) for i in ids)
    return hit / max(tot, 1)


def oracle_f05(candidates, gold, s1_ids=None):
    """Macro F0.5 of a perfect matcher restricted to the candidates (best achievable)."""
    ids = list(gold) if s1_ids is None else s1_ids
    return macro_f05({i: set(candidates.get(i, ())) & gold.get(i, set()) for i in ids}, gold, ids)


def is_val(s1_id, n_folds=5):
    """Deterministic validation split by S1 id (fold 0 of a crc32 hash): ~20% of S1 entities."""
    import zlib
    return zlib.crc32(s1_id.encode()) % n_folds == 0


def fold_of(s1_id, n_folds=5):
    """Deterministic fold number of an S1 id, for grouped cross-validation."""
    import zlib
    return zlib.crc32(s1_id.encode()) % n_folds
