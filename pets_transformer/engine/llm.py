"""Optional, isolated LLM helper (spec §2, §5.2, §7). Off by default; the tool fully works without it.

Enabled only when PETS_LLM=1 and the `anthropic` package can resolve credentials (e.g. ANTHROPIC_API_KEY).
It is used for exactly two things and never for a transformation decision:
  (a) a role *suggestion* for low-confidence columns: only the column name + at most 5 sample values are sent;
  (b) a plain-language rewrite of the JSON report: the report itself (no row-level data) is sent and every
      number must be copied from it.
"""
from __future__ import annotations

import json
import os

MODEL = "claude-opus-5-5"
ROLES = ["DIRECT_ID", "QUASI_ID", "SENSITIVE", "DATE", "LOCATION", "FREE_TEXT", "TECHNICAL_ID", "NUMERIC",
         "CATEGORICAL", "OTHER"]
MAX_SAMPLES = 5


def enabled() -> bool:
    if os.environ.get("PETS_LLM", "0") != "1":
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _client():
    import anthropic
    return anthropic.Anthropic()


def _call(system: str, user: str, max_tokens: int, effort: str, schema: dict | None = None) -> str | None:
    """One request with server-side refusal fallback. Returns the text, or None on refusal/error."""
    import anthropic
    output_config: dict = {"effort": effort}
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    try:
        resp = _client().beta.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config=output_config,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.APIStatusError):
        return None
    if resp.stop_reason == "refusal":
        return None
    return "".join(b.text for b in resp.content if b.type == "text") or None


ROLE_SCHEMA = {
    "type": "object",
    "properties": {"role": {"type": "string", "enum": ROLES}, "reason": {"type": "string"}},
    "required": ["role", "reason"],
    "additionalProperties": False,
}


def suggest_role(column: str, samples: list[str]) -> dict | None:
    """Role suggestion for an ambiguous column. Shown in the UI as a suggestion only."""
    if not enabled():
        return None
    samples = [str(s)[:80] for s in samples[:MAX_SAMPLES]]
    text = _call(
        system=("You classify dataset columns for privacy protection. Roles: DIRECT_ID identifies a person on its "
                "own (name, email, phone, national id); QUASI_ID identifies in combination (age, sex, postcode, "
                "event dates); SENSITIVE is special-category or financial information; TECHNICAL_ID is a device/"
                "network/record key; FREE_TEXT is prose. Answer with the single best role and a one-sentence reason."),
        user=f"Column name: {column}\nSample values (synthetic data): {json.dumps(samples, ensure_ascii=False)}",
        max_tokens=1024, effort="low", schema=ROLE_SCHEMA,
    )
    if not text:
        return None
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        return None
    return out if out.get("role") in ROLES else None


def plain_language(report: dict) -> str | None:
    """Plain-language summary written from the JSON report only."""
    if not enabled():
        return None
    slim = {k: report.get(k) for k in ("scenario", "threat_model", "decisions", "metrics", "utility", "target_met",
                                       "residual_risks", "text")}
    if slim.get("utility"):
        slim["utility"] = {k: v for k, v in slim["utility"].items() if k != "descriptive"} | {
            "descriptive_score": report["utility"]["descriptive"]["score"]}
    return _call(
        system=("You explain a privacy-protection report to a non-technical data owner in plain English, in at most "
                "250 words: what was shared and with whom, what was changed and why, how much re-identification risk "
                "was removed, how much analytical value was kept, and the residual risks. Copy every number exactly "
                "from the JSON; never compute, round differently or invent numbers. No headings."),
        user=json.dumps(slim, ensure_ascii=False, default=str),
        max_tokens=4000, effort="medium",
    )
