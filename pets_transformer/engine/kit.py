"""Starter-kit integration: official sharing scenarios, table recognition and the kit's output conventions.

The kit (TrustED PETs Proving Ground 2026) scores protected tables with `pets-eval`. This module turns an official
scenario (e.g. HEALTH-S1) into our Scenario: recipient, attacker knowledge (every catalogued quasi-identifier, because
the kit counts k over all of them) and utility requirements (the coarsest level each column may reach so that the
scenario's utility tasks stay answerable). It also maps our roles to the kit's label vocabulary.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .roles import ColumnProfile, Role, Scenario

DATA = Path(__file__).resolve().parent / "kit_data"
TRUST_TO_RECIPIENT = {"public": "public", "contractual": "partner", "controlled_research": "partner", "internal": "internal"}

# Utility tasks -> {column: coarsest level that keeps the task answerable} (level names of our ladders).
# `conflict` marks tasks that need a granularity which normally cannot reach k >= 5; they start unchecked
# (sacrificed) and the user can opt in.
TASK_NEEDS: dict[str, dict] = {
    "H-U1": {"desc": "Weekly influenza-like-illness visits by 10-year age band",
             "caps": {"arrival_ts": "week", "date_of_birth": "age, 10-year bands", "icd10_primary": "3-character category"}},
    "H-U2": {"desc": "Median length of stay by triage level", "caps": {}, "keep": ["triage", "los_minutes"]},
    "H-U3": {"desc": "Admission prediction (AUC ratio)", "conflict": "needs the hour of arrival",
             "caps": {"arrival_ts": "hour", "date_of_birth": "age, 10-year bands", "icd10_primary": "3-character category"}},
    "H-U4": {"desc": "Diagnosis-group mix in the eight largest municipalities",
             "caps": {"municipality": "merge categories < 1 %", "icd10_primary": "3-character category"}},
    "H-U5": {"desc": "Distributions of vital signs", "caps": {}, "keep": ["systolic_bp", "heart_rate", "temp_c", "spo2"]},
    "H-U6": {"desc": "30-day revisit rate by age band", "caps": {"date_of_birth": "age, 10-year bands"},
             "keep": ["readmit_30d"]},
    "H-U7": {"desc": "CRP vs temperature (needs lab_results released too, joined on the visit pseudonym)",
             "caps": {}, "keep": ["temp_c"], "tables": ["lab_results"]},
    "H-U8": {"desc": "Hour-of-day arrival curve", "conflict": "needs the hour of arrival", "caps": {"arrival_ts": "hour"}},
}

ROLE_TO_KIT = {"DIRECT_ID": "direct_identifier", "QUASI_ID": "quasi_identifier", "DATE": "quasi_identifier",
               "LOCATION": "quasi_identifier", "SENSITIVE": "sensitive_attribute", "FREE_TEXT": "free_text",
               "NUMERIC": "non_sensitive", "CATEGORICAL": "non_sensitive", "OTHER": "non_sensitive",
               "TARGET": "non_sensitive"}
SUBTYPE_TO_KIT = {"person_name": "PERSON_NAME", "email": "EMAIL", "phone": "PHONE", "national_id": "NATIONAL_ID",
                  "address": "STREET_ADDRESS", "iban": "IBAN", "credit_card": "PAYMENT_CARD", "account": "ACCOUNT_ID",
                  "plate": "LICENSE_PLATE", "ip": "IP_ADDRESS", "mac": "MAC_ADDRESS", "url": "URL", "device": "DEVICE_ID",
                  "postcode": "POSTCODE", "area": "MUNICIPALITY", "birth_date": "BIRTH_DATE", "birth_year": "BIRTH_DATE",
                  "age": "AGE", "gender": "SEX", "event_date": "TIMESTAMP", "latitude": "GPS_COORDINATE",
                  "longitude": "GPS_COORDINATE", "latlon_pair": "GPS_COORDINATE", "diagnosis_code": "DIAGNOSIS_CODE",
                  "free_text": "FREE_TEXT", "record_key": "RECORD_ID", "measurement": "MEASUREMENT",
                  "category": "CATEGORY", "occupation": "OCCUPATION", "origin": "NATIONALITY",
                  "sensitive_amount": "FINANCIAL_DATA", "sensitive_category": "OTHER"}


@lru_cache(maxsize=1)
def scenarios() -> list[dict]:
    return json.loads((DATA / "scenarios.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def catalog() -> pd.DataFrame:
    return pd.read_csv(DATA / "field_catalog.csv", dtype=str)


def match_table(columns: list[str]) -> tuple[str, str] | None:
    """(dataset, table) of the kit table whose catalogued columns best match these columns (>= 80 % overlap)."""
    cols = set(columns)
    best, score = None, 0.0
    for (ds, tb), g in catalog().groupby(["dataset", "table"]):
        cat_cols = set(g["column"])
        overlap = len(cols & cat_cols) / max(len(cat_cols | cols), 1)
        if overlap > score:
            best, score = (ds, tb), overlap
    return best if score >= 0.8 else None


def table_qis(dataset: str, table: str, columns: list[str]) -> list[str]:
    c = catalog()
    q = c[(c.dataset == dataset) & (c.table == table) & (c.privacy_role == "quasi_identifier")]["column"]
    return [x for x in q if x in columns]


def scenarios_for(columns: list[str]) -> list[dict]:
    """Official scenarios that apply to an uploaded table, with their tasks and default task selection."""
    m = match_table(columns)
    if m is None:
        return []
    out = []
    for s in scenarios():
        if s["dataset"] != m[0]:
            continue
        tasks = []
        for t in s.get("utility_requirements", []):
            need = TASK_NEEDS.get(t, {})
            tasks.append({"id": t, "desc": need.get("desc", "task defined by the kit (no column requirement mapped)"),
                          "conflict": need.get("conflict"), "default": not need.get("conflict"),
                          "caps": need.get("caps", {}), "tables": need.get("tables", [])})
        out.append({**s, "table": m[1], "tasks": tasks})
    return out


def scenario_from_kit(sid: str, columns: list[str], tasks: list[str] | None = None, seed: int = 42) -> Scenario:
    """Build our Scenario from an official kit scenario. `tasks` = utility tasks to keep (default: non-conflicting)."""
    s = next((x for x in scenarios_for(columns) if x["id"] == sid), None)
    if s is None:
        raise ValueError(f"Scenario {sid} does not apply to this table.")
    chosen = [t["id"] for t in s["tasks"] if (t["id"] in tasks if tasks is not None else t["default"])]
    caps: dict[str, str] = {}
    keep: list[str] = []
    for t in chosen:
        need = TASK_NEEDS.get(t, {})
        for col, lvl in need.get("caps", {}).items():
            if col in columns:
                caps.setdefault(col, lvl)  # first requirement wins; tasks here never disagree on a column's cap
        keep += [c for c in need.get("keep", []) if c in columns]
    target = "readmit_30d" if "readmit_30d" in columns and s["dataset"] == "health" else None
    return Scenario(
        recipient=TRUST_TO_RECIPIENT.get(s["trust_level"], "partner"),
        attacker_qis=table_qis(s["dataset"], s["table"], columns),
        task_type="classification" if target else "descriptive", target=target,
        key_columns=sorted(set(keep) - {target}),
        description=(f"{s['id']} — {s['title']}. Recipient: {s['recipient']}. Purpose: {s['purpose']}. "
                     f"Threat model: {s['threat_model']} Suggested targets: {s['suggested_privacy_targets']}"),
        seed=seed, max_levels=caps, kit_scenario=sid, kit_tasks=chosen,
        sacrificed_tasks=[t["id"] for t in s["tasks"] if t["id"] not in chosen],
    )


def kit_role(p: ColumnProfile) -> str:
    if p.role == Role.TECHNICAL_ID:
        return "record_key" if p.subtype == "record_key" and p.unique_ratio > 0.99 else "technical_identifier"
    return ROLE_TO_KIT[p.role.value]


def field_labels(profiles: list[ColumnProfile], dataset: str, table: str) -> pd.DataFrame:
    """Our detection in the kit's format `dataset,table,column,privacy_role,subtype` (for `pets-eval detect` and
    the organisers' unseen-dataset scoring)."""
    return pd.DataFrame([{"dataset": dataset, "table": table, "column": p.name, "privacy_role": kit_role(p),
                          "subtype": SUBTYPE_TO_KIT.get(p.subtype or "", "OTHER")} for p in profiles])
