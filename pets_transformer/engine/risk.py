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


def tolerant_linkage(att_exact: pd.DataFrame, att_val: pd.DataFrame, rel_exact: pd.DataFrame, rel_lo: pd.DataFrame,
                     rel_hi: pd.DataFrame, windows: dict[str, tuple[float, float]], max_cells: int = 4_000_000) -> dict:
    """Linkage with approximate background knowledge, as in the kit's attack specs (age ± years, time windows).

    att_exact / rel_exact: exactly matched QIs (attacker values already passed through the public generalisation).
    att_val: the attacker's approximate values for the tolerant QIs; rel_lo / rel_hi: the interval each released value
    stands for. A released record is a candidate when every exact QI is equal and, for every tolerant QI c,
    [v - before, v + after] overlaps [lo, hi]. Success = exactly one candidate and it is the true record.
    With exact knowledge of the same columns this attacker is never stronger than `linkage_attack` (its candidate set
    contains the target's whole equivalence class), so it is reported next to it, not instead of it.
    """
    n = len(att_exact) if len(att_exact.columns) else len(att_val)
    if n == 0:
        return {"success_rate": 0.0, "successes": 0, "targets": 0}
    exact_cols = list(att_exact.columns)
    att_key = _keys(att_exact, exact_cols) if exact_cols else pd.Series(0, index=att_val.index)
    rel_key = _keys(rel_exact, exact_cols) if exact_cols else pd.Series(0, index=rel_lo.index)
    tol = list(windows)
    rel_groups = {k: idx for k, idx in rel_key.groupby(rel_key).groups.items()}
    successes = 0
    for k, a_idx in att_key.groupby(att_key).groups.items():
        r_idx = rel_groups.get(k)
        if r_idx is None:
            continue
        r_ids = np.asarray(r_idx)
        lo = rel_lo.loc[r_idx, tol].to_numpy(float)
        hi = rel_hi.loc[r_idx, tol].to_numpy(float)
        a_ids = np.asarray(a_idx)
        v = att_val.loc[a_idx, tol].to_numpy(float)
        before = np.array([windows[c][0] for c in tol])
        after = np.array([windows[c][1] for c in tol])
        step = max(1, max_cells // max(len(r_ids), 1))
        for s0 in range(0, len(a_ids), step):
            vv = v[s0:s0 + step]
            ok = np.ones((len(vv), len(r_ids)), dtype=bool)
            for j in range(len(tol)):
                a_lo, a_hi = vv[:, j] - before[j], vv[:, j] + after[j]
                # unknown attacker value or unknown released interval: that QI cannot narrow the candidates
                known = ~np.isnan(a_lo)[:, None] & ~np.isnan(lo[:, j])[None, :]
                hit = (lo[None, :, j] <= a_hi[:, None]) & (hi[None, :, j] >= a_lo[:, None])
                ok &= hit | ~known
            counts = ok.sum(axis=1)
            first = ok.argmax(axis=1)
            successes += int(((counts == 1) & (r_ids[first] == a_ids[s0:s0 + step])).sum())
    return {"success_rate": successes / n, "successes": successes, "targets": int(n)}
