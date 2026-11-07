"""Differential privacy for aggregate queries (spec §5.9): Laplace mechanism with clipping.

Unit of privacy: one record (row). If a person can contribute several rows, bound the contribution first
(`max_rows_per_person`) so that the stated epsilon holds per person.
For aggregate sharing only, never for row-level releases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .profiling import to_number

EPS_GRID = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0)


def bound_contributions(df: pd.DataFrame, person_col: str | None, max_rows: int, seed: int) -> pd.DataFrame:
    if not person_col or person_col not in df.columns:
        return df
    return (df.sample(frac=1.0, random_state=seed).groupby(df[person_col].astype("string"), sort=False)
            .head(max_rows).sort_index())


def dp_query(df: pd.DataFrame, group_col: str, statistic: str, epsilon: float, value_col: str | None = None,
             lower: float | None = None, upper: float | None = None, seed: int = 42, max_rows: int = 1) -> dict:
    """Noisy count or mean per group. Groups come from the public column domain (no data-dependent selection
    beyond the listed values, which is a documented simplification)."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    rng = np.random.default_rng(seed)
    groups = df[group_col].astype("string").fillna("<NA>")
    domain = sorted(groups.unique())
    true_counts = groups.value_counts().reindex(domain, fill_value=0).astype(float)
    rows = []
    if statistic == "count":
        scale = max_rows / epsilon
        noisy = true_counts + rng.laplace(0, scale, len(domain))
        for g, t, n in zip(domain, true_counts, noisy):
            rows.append({"group": g, "true": float(t), "noisy": float(max(n, 0.0)), "abs_error": float(abs(max(n, 0) - t))})
        mech = {"mechanism": "Laplace", "sensitivity": max_rows, "scale": scale}
    elif statistic == "mean":
        if value_col is None:
            raise ValueError("value_col is required for mean")
        v = to_number(df[value_col])
        lo = float(v.quantile(0.01)) if lower is None else float(lower)
        hi = float(v.quantile(0.99)) if upper is None else float(upper)
        clipped = v.clip(lo, hi).fillna((lo + hi) / 2)
        sums = clipped.groupby(groups).sum().reindex(domain, fill_value=0.0)
        true_means = v.groupby(groups).mean().reindex(domain)
        eps_half = epsilon / 2
        sens_sum = max_rows * max(abs(lo), abs(hi))
        noisy_sum = sums + rng.laplace(0, sens_sum / eps_half, len(domain))
        noisy_cnt = np.maximum(true_counts + rng.laplace(0, max_rows / eps_half, len(domain)), 1.0)
        noisy_mean = np.clip(noisy_sum / noisy_cnt, lo, hi)
        for g, t, n in zip(domain, true_means, noisy_mean):
            rows.append({"group": g, "true": None if pd.isna(t) else float(t), "noisy": float(n),
                         "abs_error": None if pd.isna(t) else float(abs(n - t))})
        mech = {"mechanism": "Laplace (noisy sum / noisy count, ε split 50/50)", "clip": [lo, hi],
                "sensitivity_sum": sens_sum, "sensitivity_count": max_rows}
    else:
        raise ValueError("statistic must be 'count' or 'mean'")
    rel = [r["abs_error"] / max(abs(r["true"]), 1.0) for r in rows if r["true"] is not None]
    return {"group_col": group_col, "statistic": statistic, "value_col": value_col, "epsilon": epsilon,
            "delta": 0.0, "unit_of_privacy": f"one person (≤ {max_rows} row(s) each)" if max_rows > 1 else "one record",
            "rows": rows, "mean_rel_error": float(np.mean(rel)) if rel else None, **mech}


def error_curve(df: pd.DataFrame, group_col: str, statistic: str, value_col: str | None = None, seed: int = 42,
                repeats: int = 20, max_rows: int = 1) -> list[dict]:
    """Mean relative error vs. epsilon, averaged over seeded repeats."""
    out = []
    for eps in EPS_GRID:
        errs = [dp_query(df, group_col, statistic, eps, value_col, seed=seed + i, max_rows=max_rows)["mean_rel_error"]
                for i in range(repeats)]
        errs = [e for e in errs if e is not None]
        out.append({"epsilon": eps, "mean_rel_error": float(np.mean(errs)) if errs else None})
    return out
