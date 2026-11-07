"""Greedy search over ladder levels (spec §5.6).

Starting from every yellow column at its floor level (red transforms applied), repeatedly apply the move
with the best risk reduction per unit of utility loss until the scenario's target is met:

    linkage attack success <= max_linkage   and   min equivalence-class size >= min_k

A move is one search unit (a column, or a group of columns that must share a level) going up one level,
or the global suppression move. The recorded path doubles as the privacy/utility trade-off curve.

Risk used for scoring is lexicographic: linkage success first (tie-break uniqueness, then prosecutor risk);
once linkage is on target, the share of records in classes smaller than k. When no move changes any of
those (a plateau: other QIs still make every record unique), the move that removes the most identifying
entropy from the attacker-known columns per unit of utility is chosen; the reason is recorded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .ladders import Ladder
from .risk import attacker_sample, class_sizes, k_metrics, linkage_attack, marginal_entropy
from .roles import ColumnProfile, Scenario
from .utility import ProxyUtility

EPS = 1e-9
SUPPRESS = "__suppress__"


@dataclass
class SearchResult:
    levels: dict[str, int]
    suppress: bool
    suppressed_rids: list[int]
    path: list[dict]
    target_met: bool
    contributions: dict[str, dict]
    final: dict
    baseline: dict
    units: list[list[str]]
    fixed: list[str] = field(default_factory=list)


class GreedySearch:
    def __init__(self, df: pd.DataFrame, profiles: list[ColumnProfile], ladders: dict[str, Ladder], scenario: Scenario,
                 known: list[str], groups: list[list[str]], proxy: ProxyUtility, caps: dict[str, int] | None = None):
        self.df, self.profiles, self.ladders, self.sc = df, profiles, ladders, scenario
        self.known = known
        self.proxy = proxy
        self.floors = {p.name: p.min_level for p in profiles if p.name in ladders}
        self.caps = dict(caps or {})  # utility requirements: coarsest allowed level per column
        known_set = set(known)
        self.units = [g for g in groups if any(c in known_set for c in g) and any(c in ladders for c in g)]
        self.fixed = [c for g in groups if g not in self.units for c in g]
        self.att_ids = attacker_sample(df.index, scenario.attacker_fraction, scenario.seed)
        self._cache: dict[tuple[str, int], pd.Series] = {}
        self._ent: dict[tuple[str, int], float] = {}

    # ------------------------------------------------------------ helpers
    def col(self, c: str, level: int) -> pd.Series:
        key = (c, level)
        if key not in self._cache:
            lad = self.ladders.get(c)
            self._cache[key] = self.df[c] if lad is None or level == 0 else lad.apply(self.df[c], level)
        return self._cache[key]

    def entropy(self, c: str, level: int) -> float:
        key = (c, level)
        if key not in self._ent:
            self._ent[key] = marginal_entropy(self.col(c, level))
        return self._ent[key]

    def distinct(self, c: str, level: int) -> int:
        key = (c, -level - 1)
        if key not in self._ent:
            self._ent[key] = float(self.col(c, level).nunique(dropna=False))
        return int(self._ent[key])

    def next_level(self, levels: dict[str, int], unit: list[str]) -> int | None:
        """Next level that actually merges values for some column of the unit (levels are coarsenings, so an
        unchanged number of distinct values means an unchanged partition). None when the unit is exhausted."""
        cur = self.unit_level(levels, unit)
        for lvl in range(cur + 1, self.unit_max(unit) + 1):
            for c in unit:
                if c not in self.ladders:
                    continue
                now, nxt = levels.get(c, 0), max(self.floors.get(c, 0), min(lvl, self.col_max(c)))
                if nxt > now and self.distinct(c, nxt) < self.distinct(c, now):
                    return lvl
        return None

    def unit_level(self, levels: dict[str, int], unit: list[str]) -> int:
        return max(levels.get(c, 0) for c in unit)

    def col_max(self, c: str) -> int:
        """Coarsest level allowed for a column: ladder end, the scenario's utility cap, never below the privacy floor."""
        return max(self.floors.get(c, 0), min(self.ladders[c].max_level, self.caps.get(c, self.ladders[c].max_level)))

    def unit_max(self, unit: list[str]) -> int:
        return max(self.col_max(c) for c in unit if c in self.ladders)

    def set_unit(self, levels: dict[str, int], unit: list[str], level: int) -> dict[str, int]:
        new = dict(levels)
        for c in unit:
            if c in self.ladders:
                new[c] = max(self.floors.get(c, 0), min(level, self.col_max(c)))
        return new

    # ------------------------------------------------------------ evaluation
    def evaluate(self, levels: dict[str, int], suppress: bool) -> dict | None:
        k = int(self.sc.min_k)
        qi = pd.DataFrame({c: self.col(c, levels.get(c, 0)) for c in self.known}, index=self.df.index)
        sizes = class_sizes(qi, self.known) if self.known else pd.Series(len(qi), index=qi.index)
        small = sizes < k
        need = float(small.mean())
        suppressed = []
        if suppress:
            if need > self.sc.suppression_cap + EPS:
                return None
            suppressed = qi.index[small].tolist()
        released = qi.drop(index=suppressed) if suppressed else qi
        attacker = qi.loc[self.att_ids]  # attacker's original QI values under the same public generalisation
        link = linkage_attack(attacker, released, self.known)
        km = k_metrics(released, self.known, k)
        supp_frac = len(suppressed) / len(self.df)
        util = self.proxy(levels, released.index, supp_frac)
        return {
            "linkage": link["success_rate"], "attack": link,
            "uniqueness": km["uniqueness"], "min_k": km["min_k"], "prosecutor_risk": km["prosecutor_risk"],
            "below_k": km["below_k"], "k_distribution": km["k_distribution"], "classes": km["classes"],
            "suppressed_frac": supp_frac, "suppressed": suppressed, "suppression_need": need,
            "entropy": sum(self.entropy(c, levels.get(c, 0)) for c in self.known),
            **util,
        }

    def met(self, m: dict) -> bool:
        return m["linkage"] <= self.sc.max_linkage + EPS and m["min_k"] >= int(self.sc.min_k)

    # ------------------------------------------------------------ search
    def run(self, max_steps: int = 200) -> SearchResult:
        levels = {c: self.floors.get(c, 0) for c in self.ladders}
        suppress = False
        cur = self.evaluate(levels, suppress)
        baseline = cur
        path = [self._point(0, "start (red transforms applied, yellow columns at floor level)", levels, suppress, cur, None)]
        contributions: dict[str, dict] = {}
        step = 0
        while not self.met(cur) and step < max_steps:
            candidates = []
            for unit in self.units:
                nxt = self.next_level(levels, unit)
                if nxt is None:
                    continue
                new_levels = self.set_unit(levels, unit, nxt)
                m = self.evaluate(new_levels, suppress)
                if m is not None:
                    candidates.append((unit, new_levels, suppress, m))
            if not suppress:
                m = self.evaluate(levels, True)
                if m is not None and m["suppressed_frac"] > 0:
                    candidates.append(([SUPPRESS], levels, True, m))
            if not candidates:
                break
            stage = "linkage" if cur["linkage"] > self.sc.max_linkage + EPS else "k"
            order = ["linkage", "uniqueness", "prosecutor_risk", "entropy"] if stage == "linkage" else \
                    ["below_k", "prosecutor_risk", "entropy"]
            best, metric = None, None
            for name in order:
                scored = []
                for cand in candidates:
                    d_risk = cur[name] - cand[3][name]
                    d_util = max(cur["proxy"] - cand[3]["proxy"], 0.0)
                    if d_risk > EPS:
                        scored.append((d_risk / (d_util + 1e-6), cand))
                if scored:
                    best = max(scored, key=lambda t: t[0])[1]
                    metric = name
                    break
            if best is None:  # nothing reduces any risk measure: take the cheapest remaining move
                best = min(candidates, key=lambda cand: cur["proxy"] - cand[3]["proxy"])
                metric = "none"
            unit, levels, suppress, new = best
            step += 1
            label = "suppress records in classes with k < %d" % self.sc.min_k if unit == [SUPPRESS] else \
                ", ".join(f"{c} → {self.ladders[c].level_names[levels[c]]}" for c in unit if c in self.ladders)
            key = "suppression" if unit == [SUPPRESS] else "+".join(unit)
            c = contributions.setdefault(key, {"columns": unit, "risk_pp": 0.0, "utility_cost": 0.0, "steps": 0, "metrics": []})
            c["risk_pp"] += (cur["linkage"] - new["linkage"]) * 100
            c["utility_cost"] += cur["proxy"] - new["proxy"]
            c["steps"] += 1
            c["metrics"].append(metric)
            path.append(self._point(step, label, levels, suppress, new, metric))
            cur = new
        return SearchResult(levels=levels, suppress=suppress, suppressed_rids=list(cur["suppressed"]), path=path,
                            target_met=self.met(cur), contributions=contributions, final=cur, baseline=baseline,
                            units=self.units, fixed=self.fixed)

    @staticmethod
    def _point(step: int, label: str, levels: dict, suppress: bool, m: dict, metric: str | None) -> dict:
        return {
            "step": step, "move": label, "chosen_by": metric, "levels": dict(levels), "suppress": suppress,
            "linkage": m["linkage"], "uniqueness": m["uniqueness"], "min_k": m["min_k"],
            "prosecutor_risk": m["prosecutor_risk"], "below_k": m["below_k"], "suppressed_frac": m["suppressed_frac"],
            "proxy": m["proxy"], "info_score": m["info_score"], "corr_diff": m["corr_diff"],
        }
