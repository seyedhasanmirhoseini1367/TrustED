"""Apply a level plan (red fixed transforms + yellow ladder levels + suppression) to a DataFrame."""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .ladders import Ladder
from .roles import ColumnProfile, Gate, RedAction, Role

RID = "__rid"
SECRET_FILE = Path(__file__).resolve().parent.parent / ".pets_secret"


def get_secret() -> tuple[bytes, str]:
    """Pseudonymisation key: env var PETS_SECRET, else a local key file created once (never exported)."""
    env = os.environ.get("PETS_SECRET")
    if env:
        return env.encode(), "environment variable PETS_SECRET"
    if not SECRET_FILE.exists():
        SECRET_FILE.write_text(secrets.token_hex(32))
    return SECRET_FILE.read_text().strip().encode(), "local key file (.pets_secret, not exported)"


def pseudonymise(s: pd.Series, secret: bytes) -> pd.Series:
    """HMAC-SHA256(secret, value) truncated to 16 hex chars. Same value -> same token across tables."""
    cache: dict[str, str] = {}

    def tok(v):
        if pd.isna(v):
            return pd.NA
        key = str(v).strip()
        if key not in cache:
            cache[key] = hmac.new(secret, key.encode("utf-8"), hashlib.sha256).hexdigest()[:16]
        return cache[key]

    return s.map(tok).astype("string")


def truncate_ip(s: pd.Series) -> pd.Series:
    """IPv4 -> /24, IPv6 -> /48. Non-IP values become missing."""
    def trunc(v):
        if pd.isna(v):
            return pd.NA
        try:
            ip = ipaddress.ip_address(str(v).strip())
        except ValueError:
            return pd.NA
        prefix = 24 if ip.version == 4 else 48
        return str(ipaddress.ip_network(f"{ip}/{prefix}", strict=False))

    return s.map(trunc).astype("string")


@dataclass
class Plan:
    levels: dict[str, int] = field(default_factory=dict)        # yellow column -> ladder level
    suppress: bool = False
    suppressed_rids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"levels": dict(self.levels), "suppress": self.suppress, "suppressed_rows": len(self.suppressed_rids)}


def apply_red(s: pd.Series, profile: ColumnProfile, secret: bytes) -> pd.Series | None:
    action = profile.red_action or RedAction.DROP
    if action == RedAction.DROP:
        return None
    if action == RedAction.PSEUDONYMISE:
        return pseudonymise(s, secret)
    if action == RedAction.TRUNCATE_IP:
        return truncate_ip(s)
    return s


def apply_plan(df: pd.DataFrame, profiles: list[ColumnProfile], ladders: dict[str, Ladder], plan: Plan,
               secret: bytes, text: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Return the protected view. `df` may be any subset of the original rows (index = original row id).

    Suppressed rows are removed; dropped columns are removed; column order is kept.
    """
    out = {}
    text = text or {}
    for p in profiles:
        if p.name not in df.columns:
            continue
        s = df[p.name]
        if p.gate == Gate.RED:
            r = apply_red(s, p, secret)
            if r is not None:
                out[p.name] = r
        elif p.role == Role.FREE_TEXT:
            if p.name in text:
                out[p.name] = text[p.name].reindex(df.index)
            else:
                out[p.name] = s
        elif p.gate == Gate.YELLOW and p.name in ladders:
            out[p.name] = ladders[p.name].apply(s, plan.levels.get(p.name, 0))
        else:
            out[p.name] = s
    result = pd.DataFrame(out, index=df.index)
    if plan.suppress and plan.suppressed_rids:
        result = result.drop(index=[r for r in plan.suppressed_rids if r in result.index])
    return result
