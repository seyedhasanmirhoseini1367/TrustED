"""Default gate assignment (spec §5.3) and threat-model helpers.

Red    = fixed transform, excluded from the search (direct and technical identifiers).
Yellow = search space (quasi-identifiers, dates, locations, rare sensitive categories, free text).
Green  = unchanged by default; promoted to yellow if it measurably helps the attacker.
"""
from __future__ import annotations

import pandas as pd

from .ladders import make_ladder
from .risk import attacker_sample, linkage_attack
from .roles import ColumnProfile, Gate, Kind, RedAction, Role, Scenario

DROP_SUBTYPES = {"person_name", "email", "phone", "address", "plate", "account", "credit_card", "iban", "identifier"}
PROMOTION_PP = 0.02


def assign_gates(profiles: list[ColumnProfile], df: pd.DataFrame, scenario: Scenario) -> list[ColumnProfile]:
    for p in profiles:
        if scenario.target and p.name == scenario.target:
            p.role, p.gate, p.red_action, p.min_level = Role.TARGET, Gate.GREEN, None, 0
            p.gate_reason = "target of the analytical task; released unchanged"
            continue
        if p.overridden and p.gate is not None:
            if p.gate == Gate.RED and p.red_action is None:
                p.red_action = _default_red(p)[0]
            if not p.gate_reason:
                p.gate_reason = "set by the user"
            continue
        p.min_level = 0
        if p.role in (Role.DIRECT_ID, Role.TECHNICAL_ID):
            p.gate = Gate.RED
            p.red_action, p.gate_reason = _default_red(p)
        elif p.role == Role.FREE_TEXT:
            p.gate, p.red_action = Gate.YELLOW, None
            p.gate_reason = "free text: PII redaction always applied (not searched)"
        elif p.role in (Role.QUASI_ID, Role.DATE, Role.LOCATION):
            p.gate, p.red_action = Gate.YELLOW, None
            p.gate_reason = "quasi-identifier: generalisation level chosen by the search"
        elif p.role == Role.SENSITIVE:
            k = int(scenario.min_k)
            rare = _rare_share(df[p.name], k) if p.kind == Kind.CATEGORICAL else 0.0
            if rare > 0:
                p.gate, p.red_action = Gate.YELLOW, None
                p.min_level = _floor_level(p, df, scenario)
                p.gate_reason = (f"sensitive attribute with rare categories ({rare:.1%} of records in categories with "
                                 f"< {k} records); floor level {p.min_level} so a rare value cannot single anyone out")
            else:
                p.gate, p.red_action = Gate.GREEN, None
                p.gate_reason = "sensitive attribute kept for analysis; attribute disclosure checked with l-diversity"
        else:
            p.gate, p.red_action = Gate.GREEN, None
            p.gate_reason = "not identifying on its own; unchanged"
    return profiles


def _default_red(p: ColumnProfile) -> tuple[RedAction, str]:
    if p.subtype == "ip":
        return RedAction.TRUNCATE_IP, "IP address: truncated to /24 (IPv4) or /48 (IPv6)"
    if p.role == Role.TECHNICAL_ID:
        return RedAction.PSEUDONYMISE, "technical identifier: keyed HMAC pseudonym keeps joins"
    if p.subtype in DROP_SUBTYPES:
        return RedAction.DROP, f"direct identifier ({p.subtype}): dropped, not needed for the analysis"
    return RedAction.PSEUDONYMISE, f"direct identifier ({p.subtype}): keyed HMAC pseudonym keeps a stable person key"


def _rare_share(s: pd.Series, k: int) -> float:
    """Share of records whose category occurs fewer than k times."""
    counts = s.astype("string").value_counts()
    return float(counts[counts < k].sum() / max(len(s), 1))


def _floor_level(p: ColumnProfile, df: pd.DataFrame, scenario: Scenario) -> int:
    """Lowest ladder level at which every category holds at least k records."""
    lad = make_ladder(p.name, p.ladder, scenario.hierarchies.get(p.name), int(scenario.min_k))
    if lad is None:
        return 0
    lad.fit(df[p.name])
    for level in range(1, lad.max_level + 1):
        out = lad.apply(df[p.name], level)
        if _rare_share(out[out != "Other"], int(scenario.min_k)) == 0:  # a small 'Other' bucket names no value
            return level
    return lad.max_level


def attacker_known(profiles: list[ColumnProfile], scenario: Scenario) -> list[str]:
    """Attacker-known QIs: the scenario's list, or every yellow quasi-identifier when none is given."""
    names = {p.name: p for p in profiles}
    if scenario.attacker_qis:
        return [c for c in scenario.attacker_qis if c in names and names[c].gate != Gate.RED]
    return [p.name for p in profiles if p.gate == Gate.YELLOW and p.role in (Role.QUASI_ID, Role.DATE, Role.LOCATION)]


def derive_knowledge(df: pd.DataFrame, profiles: list[ColumnProfile], known: list[str]) -> list[dict]:
    """Columns the attacker can derive from what they know (functional dependency in the original data).

    Example: an attacker who knows the postcode also knows the city, so releasing an exact city would undo
    a generalised postcode. Such columns are added to the attacker's knowledge.
    """
    derived = []
    for p in profiles:
        if p.name in known or p.gate != Gate.YELLOW or p.role == Role.FREE_TEXT:
            continue
        for k in known:
            a = df[k].astype("string").fillna("<NA>")
            b = df[p.name].astype("string").fillna("<NA>")
            if b.nunique() < 2 or a.nunique() >= 0.95 * len(df):
                continue
            per = pd.DataFrame({"a": a, "b": b}).groupby("a")["b"].nunique()
            share = float(a.map(per).eq(1).mean())
            if share >= 0.99:
                derived.append({"column": p.name, "from": k, "share": round(share, 4)})
                break
    return derived


def column_groups(profiles: list[ColumnProfile]) -> list[list[str]]:
    """Columns that must share one generalisation level.

    - event dates of the same table (admission/discharge...): otherwise one exact date plus a duration
      reconstructs the other;
    - latitude/longitude pairs.
    """
    groups, seen = [], set()
    event_dates = [p.name for p in profiles if p.gate == Gate.YELLOW and p.ladder in ("date", "timestamp")
                   and p.subtype != "birth_date"]
    if len(event_dates) > 1:
        groups.append(event_dates)
        seen.update(event_dates)
    coords = [p.name for p in profiles if p.gate == Gate.YELLOW and p.ladder == "latlon"]
    if len(coords) > 1:
        groups.append(coords)
        seen.update(coords)
    for p in profiles:
        if p.gate == Gate.YELLOW and p.name not in seen and p.role != Role.FREE_TEXT:
            groups.append([p.name])
    return groups


def promote_green(df: pd.DataFrame, profiles: list[ColumnProfile], known: list[str], scenario: Scenario) -> list[dict]:
    """Promote a green column to yellow if adding it to the attacker's known set raises linkage success by > 2 pp."""
    if not scenario.promote_green:
        return []
    att_ids = attacker_sample(df.index, scenario.attacker_fraction, scenario.seed)
    base = linkage_attack(df.loc[att_ids], df, known)["success_rate"]
    promoted = []
    for p in profiles:
        if p.gate != Gate.GREEN or p.overridden or p.role in (Role.TARGET,) or p.ladder is None:
            continue
        if p.role not in (Role.NUMERIC, Role.CATEGORICAL, Role.OTHER, Role.SENSITIVE):
            continue
        rate = linkage_attack(df.loc[att_ids], df, known + [p.name])["success_rate"]
        if rate - base > PROMOTION_PP:
            p.gate = Gate.YELLOW
            p.gate_reason = (f"promoted from green: adding it to the threat model's known attributes raises the "
                             f"re-identification rate "
                             f"from {base:.1%} to {rate:.1%}")
            promoted.append({"column": p.name, "before": base, "after": rate})
    return promoted
