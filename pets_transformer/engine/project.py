"""Protect several related tables together (a project).

1. Links between tables are detected (relations.detect_links) or given.
2. Tables are processed parents first. For a child table, the parent's already-protected identifying attributes are
   joined onto its rows through the link and count as known attributes in its re-identification test, so the joined
   view (child row + its parent's attributes) also meets k and the re-identification target.
3. Link columns are kept as keyed HMAC pseudonyms with one secret, so joins still work after protection.
4. Free-text redaction in every table knows the names and places of all tables.
5. Link integrity after protection is reported: the share of released child rows whose link value still finds its
   released parent row.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .pipeline import RunResult, person_names, place_names, run
from .profiling import profile_dataframe
from .relations import detect_links, processing_order
from .roles import ColumnProfile, Scenario
from .transforms import get_secret


@dataclass
class ProjectResult:
    results: dict[str, RunResult]
    order: list[str]
    links: list[dict]
    summary: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"order": self.order, "links": self.links, "tables": self.summary}


def _context(child: pd.DataFrame, col: str, parent_df: pd.DataFrame, parent_col: str, parent_res: RunResult,
             parent_name: str) -> pd.DataFrame:
    """The parent's released identifying attributes for every child row (missing where the parent row was suppressed)."""
    known = [c for c in parent_res.report["threat_model"]["attacker_knows"]
             if c in parent_res.protected.columns]
    key_to_idx = pd.Series(parent_df.index, index=parent_df[parent_col].astype("string").str.strip())
    key_to_idx = key_to_idx[~key_to_idx.index.duplicated()]
    idx = child[col].astype("string").str.strip().map(key_to_idx)
    vals = parent_res.protected[known].reindex(idx.to_numpy())
    vals.index = child.index
    return vals.rename(columns={c: f"{parent_name}.{c}" for c in known})


def run_project(tables: dict[str, pd.DataFrame], scenarios: dict[str, Scenario] | Scenario,
                links: list[dict] | None = None, profiles: dict[str, list[ColumnProfile]] | None = None,
                secret: bytes | None = None, use_presidio: bool = True,
                progress: Callable[[str], None] | None = None) -> ProjectResult:
    say = progress or (lambda m: None)
    if secret is None:
        secret, _ = get_secret()
    tables = {n: df.reset_index(drop=True) for n, df in tables.items()}
    profiles = profiles or {}
    for n, df in tables.items():
        if n not in profiles:
            profiles[n] = profile_dataframe(df)
    if links is None:
        say("detecting links between tables")
        links = detect_links(tables, profiles)
    names, places = [], []
    for n, df in tables.items():
        names += person_names(df, profiles[n])
        places += place_names(df, profiles[n])
    order = processing_order(list(tables), links)
    results: dict[str, RunResult] = {}
    for n in order:
        df = tables[n]
        sc = copy.deepcopy(scenarios[n] if isinstance(scenarios, dict) else scenarios)
        # every link column stays a keyed pseudonym so the tables can still be joined
        link_cols = {l["child_col"] for l in links if l["child"] == n} | {l["parent_col"] for l in links if l["parent"] == n}
        sc.pseudonymise = sorted(set(sc.pseudonymise) | link_cols)
        ctx_parts = [_context(df, l["child_col"], tables[l["parent"]], l["parent_col"], results[l["parent"]], l["parent"])
                     for l in links if l["child"] == n and l["kind"] == "foreign key" and l["parent"] in results]
        context = pd.concat(ctx_parts, axis=1) if ctx_parts else None
        if context is not None:
            context = context.loc[:, ~context.columns.duplicated()]
        say(f"protecting {n}" + (f" (with {context.shape[1]} attributes joined from linked tables)" if context is not None else ""))
        results[n] = run(df, sc, profiles=profiles[n], secret=secret, use_presidio=use_presidio,
                         extra_names=names, extra_places=places, context=context,
                         progress=lambda m, n=n: say(f"{n}: {m}"))
    # link integrity after protection
    for l in links:
        c, p = results[l["child"]].protected, results[l["parent"]].protected
        if l["child_col"] not in c.columns or l["parent_col"] not in p.columns:
            l["integrity"] = 0.0
            l["integrity_note"] = "a link column is not released"
            continue
        cv = c[l["child_col"]].dropna().astype(str)
        pv = set(p[l["parent_col"]].dropna().astype(str))
        l["integrity"] = float(cv.isin(pv).mean()) if len(cv) else 0.0
        l["integrity_note"] = "share of released child rows whose link value finds a released parent row"
    summary = []
    for n in order:
        r = results[n].report
        m = r["metrics"]
        summary.append({"table": n, "rows": m["rows_original"], "rows_released": m["rows_released"],
                        "target_met": r["target_met"], "reid_before": m["before"]["linkage"], "reid_after": m["after"]["linkage"],
                        "min_k_after": m["after"]["min_k"], "utility": r["utility"]["score"],
                        "suppressed": m["rows_suppressed_frac"],
                        "joined_from": sorted({c.split(".")[0] for c in r["threat_model"].get("linked_context_columns", [])})})
    return ProjectResult(results=results, order=order, links=links, summary=summary)
