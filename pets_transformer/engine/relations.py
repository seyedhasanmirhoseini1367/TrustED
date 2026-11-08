"""Relationships between tables uploaded together (a "project").

A link child.col -> parent.key is detected when parent.key is a unique key of its table and at least 90 % of the
distinct non-missing values of child.col occur in it (a foreign key). Columns of two tables that hold the same
identifier for the same people (e.g. a national id in two tables, both non-unique) are reported as a shared identifier.
Detection uses values only; names are just a tie-breaker. Links are proposals the user can untick.
"""
from __future__ import annotations

import pandas as pd

from .roles import ColumnProfile, Role

MIN_COVERAGE = 0.90
SHARED_MIN = 0.50
SAMPLE = 5000


def _values(s: pd.Series, sample: bool = True) -> set:
    """Distinct values; the child side may be sampled, a parent key never (coverage would be underestimated)."""
    v = s.dropna().astype(str).str.strip()
    v = v[v != ""].drop_duplicates()
    if sample and len(v) > SAMPLE:
        v = v.sample(SAMPLE, random_state=0)
    return set(v)


# contact details and names are dropped by default; they are never used as link keys (ids are)
NOT_LINK_SUBTYPES = {"person_name", "email", "phone", "address", "plate"}


def _candidate(p: ColumnProfile, s: pd.Series) -> bool:
    """Columns that can carry a link: identifiers and other many-valued, non-measurement columns."""
    if p.subtype in NOT_LINK_SUBTYPES:
        return False
    if p.role in (Role.DIRECT_ID, Role.TECHNICAL_ID):
        return True
    if p.role in (Role.FREE_TEXT, Role.NUMERIC, Role.SENSITIVE, Role.TARGET) or p.dtype == "numeric":
        return False
    return s.nunique(dropna=True) > 20 and p.subtype in ("record_key", "identifier", "technical_key", "hash")


def detect_links(tables: dict[str, pd.DataFrame], profiles: dict[str, list[ColumnProfile]]) -> list[dict]:
    keys = {}       # table -> [(col, values)] unique keys
    cands = {}      # table -> [(col, values, unique)]
    for name, df in tables.items():
        for p in profiles[name]:
            if p.name not in df.columns or not _candidate(p, df[p.name]):
                continue
            s = df[p.name]
            nn = s.dropna()
            unique = len(nn) > 0 and nn.is_unique and p.missing_ratio < 0.01
            vals = _values(s)
            cands.setdefault(name, []).append((p.name, vals, unique))
            if unique:
                keys.setdefault(name, []).append((p.name, _values(s, sample=False)))
    links = []
    for child, cols in cands.items():
        for col, vals, col_unique in cols:
            if not vals:
                continue
            best = None
            for parent, pkeys in keys.items():
                if parent == child:
                    continue
                for key, kvals in pkeys:
                    cov = len(vals & kvals) / len(vals)
                    score = (cov, col == key)
                    if cov >= MIN_COVERAGE and (best is None or score > best[0]):
                        best = (score, parent, key, cov)
            if best is not None and not (col_unique and any(  # 1:1 links are reported once, from the larger table
                    l["child"] == best[1] and l["parent"] == child for l in links)):
                links.append({"child": child, "child_col": col, "parent": best[1], "parent_col": best[2],
                              "coverage": round(best[3], 4), "kind": "foreign key"})
    # shared identifiers (both sides non-unique, e.g. the same person id in two event tables)
    seen = {(l["child"], l["child_col"]) for l in links} | {(l["parent"], l["parent_col"]) for l in links}
    names = sorted(cands)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for ca, va, ua in cands[a]:
                for cb, vb, ub in cands[b]:
                    if ua or ub or (a, ca) in seen or (b, cb) in seen or not va or not vb:
                        continue
                    ov = len(va & vb) / min(len(va), len(vb))
                    if ov >= SHARED_MIN:
                        links.append({"child": a, "child_col": ca, "parent": b, "parent_col": cb,
                                      "coverage": round(ov, 4), "kind": "shared identifier"})
    return links


def processing_order(names: list[str], links: list[dict]) -> list[str]:
    """Parents before children (foreign-key links); cycles are broken in upload order."""
    deps = {n: {l["parent"] for l in links if l["child"] == n and l["kind"] == "foreign key" and l["parent"] != n}
            for n in names}
    order, done = [], set()
    while len(order) < len(names):
        ready = [n for n in names if n not in done and deps[n] <= done]
        if not ready:                    # cycle: take the first remaining table
            ready = [next(n for n in names if n not in done)]
        for n in ready:
            order.append(n)
            done.add(n)
    return order
