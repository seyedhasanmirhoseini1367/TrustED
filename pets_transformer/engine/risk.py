"""Re-identification risk metrics (spec §5.4), computed on the QI columns the attacker knows.

The DataFrame index is the hidden internal row id of every original record.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MISSING = "<NA>"


def _keys(frame: pd.DataFrame, qis: list[str]) -> pd.Series:
    """One hashable key per row over the QI columns (missing values are a value of their own)."""
    if not qis:
        return pd.Series(0, index=frame.index)
    sub = frame[qis].astype("string").fillna(MISSING)
    return pd.Series(pd.util.hash_pandas_object(sub, index=False).to_numpy(), index=frame.index)


def class_sizes(frame: pd.DataFrame, qis: list[str]) -> pd.Series:
    """Equivalence-class size of every row (indexed like `frame`)."""
    keys = _keys(frame, qis)
    return keys.map(keys.value_counts())


def attacker_sample(index: pd.Index, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = max(1, int(round(fraction * len(index))))
    return np.sort(rng.choice(np.asarray(index), size=min(n, len(index)), replace=False))


def linkage_attack(attacker: pd.DataFrame, released: pd.DataFrame, qis: list[str]) -> dict:
    """Record-linkage attack.

    `attacker` holds the attacker's QI values for their targets, already passed through the same public,
    deterministic generalisation as the release (index = true row id). `released` is the protected table
    (index = row id, suppressed rows absent). A target is re-identified when exactly one released record
    matches on the QIs and it is the true record.
    """
    if len(attacker) == 0:
        return {"success_rate": 0.0, "successes": 0, "targets": 0, "unique_matches": 0, "no_match": 0}
    if not qis:
        n_rel = len(released)
        hit = attacker.index.isin(released.index) & (n_rel == 1)
        return {"success_rate": float(hit.mean()), "successes": int(hit.sum()), "targets": len(attacker),
                "unique_matches": int(hit.sum()), "no_match": int(n_rel == 0) * len(attacker)}
    rel_keys = _keys(released, qis)
    counts = rel_keys.value_counts()
    singles = rel_keys[rel_keys.map(counts) == 1]
    owner = pd.Series(singles.index, index=singles.to_numpy())          # key -> released row id
    att_keys = _keys(attacker, qis)
    n_match = att_keys.map(counts).fillna(0).astype(int)
    matched_id = att_keys.map(owner)
    success = (n_match == 1) & (matched_id.to_numpy() == attacker.index.to_numpy())
    return {
        "success_rate": float(success.mean()),
        "successes": int(success.sum()),
        "targets": int(len(attacker)),
        "unique_matches": int((n_match == 1).sum()),
        "no_match": int((n_match == 0).sum()),
    }


def k_metrics(released: pd.DataFrame, qis: list[str], k_target: int) -> dict:
    if len(released) == 0:
        return {"uniqueness": 0.0, "min_k": 0, "prosecutor_risk": 0.0, "below_k": 0.0, "k_distribution": {}, "classes": 0}
    sizes = class_sizes(released, qis)
    bins = {"1": (1, 1), "2": (2, 2), "3-4": (3, 4), "5-9": (5, 9), "10-19": (10, 19), "20+": (20, 10 ** 12)}
    dist = {lab: float(((sizes >= lo) & (sizes <= hi)).mean()) for lab, (lo, hi) in bins.items()}
    return {
        "uniqueness": float((sizes == 1).mean()),
        "min_k": int(sizes.min()),
        "prosecutor_risk": float((1.0 / sizes).mean()),
        "below_k": float((sizes < k_target).mean()),
        "k_distribution": dist,
        "classes": int(_keys(released, qis).nunique()),
    }


def l_diversity(released: pd.DataFrame, qis: list[str], sensitive: list[str]) -> dict:
    """Distinct sensitive values per equivalence class (distinct l-diversity)."""
    out = {}
    if not qis or len(released) == 0:
        return out
    keys = _keys(released, qis)
    for col in sensitive:
        if col not in released.columns:
            continue
        vals = released[col].astype("string").fillna(MISSING)
        per_class = vals.groupby(keys).nunique()
        out[col] = {
            "min_l": int(per_class.min()),
            "classes_l1": int((per_class == 1).sum()),
            "records_in_l1_classes": int(keys.isin(per_class[per_class == 1].index).sum()),
            "classes": int(len(per_class)),
        }
    return out


def marginal_entropy(s: pd.Series) -> float:
    p = s.astype("string").fillna(MISSING).value_counts(normalize=True).to_numpy()
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0
