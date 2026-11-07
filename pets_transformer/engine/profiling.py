"""Rule-based column role detection (spec §5.2).

Order: (1) value regexes, (2) column-name keywords, (3) dtype + cardinality,
(4) default quasi-identifier set, (5) sensitive keywords.
"""
from __future__ import annotations

import re
import warnings

import pandas as pd

from .roles import ColumnProfile, Kind, Role

SAMPLE = 500
MATCH = 0.80

RE = {
    "email": re.compile(r"^[\w.+'-]+@[\w-]+(\.[\w-]+)+$"),
    "national_id": re.compile(r"^(\d{6}[-+A-FU-Y]\d{3}[0-9A-FHJ-NPR-Y]|\d{3}-\d{2}-\d{4})$", re.I),
    "iban": re.compile(r"^[A-Z]{2}\d{2}(?: ?[A-Z0-9]){10,30}$"),
    "ipv4": re.compile(r"^(25[0-5]|2[0-4]\d|1?\d?\d)(\.(25[0-5]|2[0-4]\d|1?\d?\d)){3}$"),
    "ipv6": re.compile(r"^[0-9a-f]{0,4}(:[0-9a-f]{0,4}){2,7}$", re.I),
    "mac": re.compile(r"^[0-9a-f]{2}([:-][0-9a-f]{2}){5}$", re.I),
    "url": re.compile(r"^(https?://|www\.)\S+$", re.I),
    "latlon_pair": re.compile(r"^-?\d{1,3}\.\d+\s*,\s*-?\d{1,3}\.\d+$"),
    "postcode": re.compile(r"^\d{4,5}$|^[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}$", re.I),
    "icd": re.compile(r"^[A-TV-Z]\d{2}(\.\d{1,2}[A-Z]?)?$", re.I),
}

# keywords -> (role, kind, ladder, subtype). Matched on name tokens (split on non-alphanumerics / camelCase).
NAME_RULES: list[tuple[tuple[str, ...], Role, Kind, str | None, str]] = [
    (("name", "surname", "firstname", "lastname", "fullname", "forename"), Role.DIRECT_ID, Kind.ID, None, "person_name"),
    (("email", "mail"), Role.DIRECT_ID, Kind.ID, None, "email"),
    (("phone", "mobile", "tel", "telephone", "msisdn"), Role.DIRECT_ID, Kind.ID, None, "phone"),
    (("ssn", "hetu", "passport", "nationalid", "personalid", "henkilotunnus"), Role.DIRECT_ID, Kind.ID, None, "national_id"),
    (("address", "street", "streetaddress"), Role.DIRECT_ID, Kind.ID, None, "address"),
    (("iban", "cardnumber", "pan", "accountnumber"), Role.DIRECT_ID, Kind.ID, None, "account"),
    (("plate", "licenseplate"), Role.DIRECT_ID, Kind.ID, None, "plate"),
    (("ip", "ipaddress", "srcip", "dstip", "clientip"), Role.TECHNICAL_ID, Kind.ID, None, "ip"),
    (("mac", "macaddress", "imei", "device", "deviceid", "hostname", "useragent", "cookie", "session", "sessionid"),
     Role.TECHNICAL_ID, Kind.ID, None, "device"),
    (("lat", "latitude"), Role.LOCATION, Kind.LOCATION, "latlon", "latitude"),
    (("lon", "lng", "long", "longitude"), Role.LOCATION, Kind.LOCATION, "latlon", "longitude"),
    (("zip", "zipcode", "postcode", "postal", "postalcode", "postinumero"), Role.QUASI_ID, Kind.LOCATION, "postcode", "postcode"),
    (("dob", "birth", "birthdate", "birthday", "dateofbirth"), Role.QUASI_ID, Kind.DATE, "birthdate", "birth_date"),
    (("age",), Role.QUASI_ID, Kind.NUMERIC, "age", "age"),
    (("gender", "sex"), Role.QUASI_ID, Kind.CATEGORICAL, "categorical", "gender"),
    (("city", "municipality", "town", "region", "county", "district", "kunta", "country", "neighbourhood", "neighborhood"),
     Role.QUASI_ID, Kind.LOCATION, "categorical", "area"),
    (("ethnicity", "ethnic", "race", "nationality", "citizenship", "language", "mothertongue"),
     Role.QUASI_ID, Kind.CATEGORICAL, "categorical", "origin"),
    (("occupation", "job", "profession", "jobtitle", "employer"), Role.QUASI_ID, Kind.CATEGORICAL, "categorical", "occupation"),
    (("marital", "maritalstatus"), Role.QUASI_ID, Kind.CATEGORICAL, "categorical", "marital"),
    (("comment", "comments", "notes", "note", "review", "description", "text", "message", "feedback", "narrative", "body", "summary"),
     Role.FREE_TEXT, Kind.TEXT, None, "free_text"),
]

SENSITIVE_WORDS = ("diagnosis", "icd", "disease", "condition", "medication", "drug", "treatment", "salary", "income",
                   "wage", "religion", "religious", "sexual", "orientation", "hiv", "pregnan", "disability",
                   "health", "mental", "psych", "union", "political", "criminal", "conviction", "debt", "balance",
                   "benefit")
ID_TOKENS = ("id", "uuid", "guid", "key", "number", "no", "nr", "ref", "token")
DOMAIN_HINTS = {
    "healthcare": ("patient", "diagnosis", "icd", "admission", "discharge", "lab", "clinic", "clinician", "triage",
                   "hospital", "ward", "medication", "visit"),
    "finance": ("iban", "transaction", "amount", "merchant", "card", "account", "balance", "transfer", "loan"),
    "mobility": ("trip", "stop", "route", "tap", "vehicle", "bike", "journey", "station"),
    "cybersecurity": ("dns", "url", "host", "proxy", "signin", "alert", "domain", "useragent", "port"),
    "education": ("student", "grade", "course", "school", "teacher", "absence", "exam"),
    "energy": ("meter", "kwh", "consumption", "load", "tariff"),
    "smart city": ("sensor", "probe", "footfall", "parking", "anpr"),
    "customer service": ("conversation", "turn", "agent", "ticket", "chat", "call"),
    "public administration": ("permit", "complaint", "request", "service", "application"),
    "HR / payroll": ("employee", "salary", "payroll", "hire", "contract"),
}


def tokens(name: str) -> list[str]:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if p]
    return parts + ["".join(parts)]


def luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def _share(values: pd.Series, pattern: re.Pattern) -> float:
    if values.empty:
        return 0.0
    return float(values.str.match(pattern).fillna(False).mean())


def _is_phone(v: str) -> bool:
    digits = re.sub(r"\D", "", v)
    if not 7 <= len(digits) <= 15 or re.search(r"[A-Za-z]", v):
        return False
    if re.fullmatch(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}.*", v):
        return False
    return v.startswith(("+", "0", "(")) or bool(re.search(r"[\s().-]", v))


def parse_dates(s: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pd.to_datetime(s, errors="coerce", format="mixed")
    try:
        if out.dt.tz is not None:
            out = out.dt.tz_localize(None)
    except (AttributeError, TypeError):
        out = pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None)
    return out


def _date_share(values: pd.Series) -> tuple[float, bool]:
    """Share of values parseable as dates and whether they carry a time-of-day component."""
    looks = values.str.match(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|^\d{1,2}[-/.]\d{1,2}[-/.]\d{4}").fillna(False)
    if looks.mean() < MATCH:
        return 0.0, False
    parsed = parse_dates(values)
    ok = parsed.notna()
    has_time = bool(ok.any() and ((parsed[ok].dt.hour != 0) | (parsed[ok].dt.minute != 0)).mean() > 0.2)
    return float(ok.mean()), has_time


def to_number(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype("string").str.strip().str.replace(",", ".", regex=False), errors="coerce").astype(float)


def _integer_like(sample: pd.Series) -> bool:
    v = to_number(sample).dropna()
    return bool(len(v)) and bool((v == v.round()).all())


def profile_column(name: str, s: pd.Series, seed: int = 0) -> ColumnProfile:
    raw = s.astype("string")
    non_null = raw.dropna()
    non_null = non_null[non_null.str.strip() != ""]
    n = len(s)
    unique_ratio = float(non_null.nunique() / max(len(non_null), 1))
    missing_ratio = float(1 - len(non_null) / max(n, 1))
    sample = non_null.sample(min(SAMPLE, len(non_null)), random_state=seed).str.strip() if len(non_null) else non_null
    examples = [str(x)[:60] for x in non_null.drop_duplicates().head(3)]
    toks = tokens(name)
    num_share = float(to_number(sample).notna().mean()) if len(sample) else 0.0
    is_numeric = num_share >= 0.95
    mean_len = float(sample.str.len().mean()) if len(sample) else 0.0
    mean_tokens = float(sample.str.split().str.len().mean()) if len(sample) else 0.0
    dtype = "numeric" if is_numeric else "string"

    def prof(role, kind, ladder, conf, reason, subtype=None):
        return ColumnProfile(name=name, dtype=dtype, role=role, kind=kind, ladder=ladder, confidence=round(conf, 2),
                             reason=reason, unique_ratio=round(unique_ratio, 4), missing_ratio=round(missing_ratio, 4),
                             examples=examples, subtype=subtype)

    if len(sample) == 0:
        return prof(Role.OTHER, Kind.CATEGORICAL, None, 0.3, "column is empty")

    # 1. value regexes (>= 80 % of sampled values must match)
    checks = [
        ("email", _share(sample, RE["email"]), Role.DIRECT_ID, Kind.ID),
        ("national_id", _share(sample, RE["national_id"]), Role.DIRECT_ID, Kind.ID),
        ("iban", _share(sample.str.upper(), RE["iban"]), Role.DIRECT_ID, Kind.ID),
        ("ip", max(_share(sample, RE["ipv4"]), _share(sample, RE["ipv6"]) if sample.str.contains(":").mean() > .5 else 0),
         Role.TECHNICAL_ID, Kind.ID),
        ("mac", _share(sample, RE["mac"]), Role.TECHNICAL_ID, Kind.ID),
        ("url", _share(sample, RE["url"]), Role.TECHNICAL_ID, Kind.ID),
        ("latlon_pair", _share(sample, RE["latlon_pair"]), Role.LOCATION, Kind.LOCATION),
    ]
    digits = sample.str.replace(r"[\s-]", "", regex=True)
    card_like = digits.str.fullmatch(r"\d{13,19}").fillna(False)
    checks.append(("credit_card", float((card_like & digits.map(lambda d: d.isdigit() and luhn_ok(d))).mean()),
                   Role.DIRECT_ID, Kind.ID))
    if not is_numeric or sample.str.startswith(("+", "0")).mean() > 0.5:
        checks.append(("phone", float(sample.map(_is_phone).mean()), Role.DIRECT_ID, Kind.ID))
    for subtype, share, role, kind in checks:
        if share >= MATCH:
            return prof(role, kind, None, 0.6 + 0.4 * share, f"{share:.0%} of values match the {subtype} pattern", subtype)

    name_rule = next(((r, k, lad, st) for words, r, k, lad, st in NAME_RULES if any(t in words for t in toks)), None)

    # lat/lon numeric columns need a name hint because their ranges overlap with ordinary numbers
    if name_rule and name_rule[2] == "latlon" and is_numeric:
        vals = to_number(sample)
        limit = 90 if name_rule[3] == "latitude" else 180
        if vals.abs().max() <= limit:
            return prof(Role.LOCATION, Kind.LOCATION, "latlon", 0.95,
                        f"name suggests {name_rule[3]} and values are within ±{limit}°", name_rule[3])
        name_rule = None

    date_share, has_time = (0.0, False) if is_numeric else _date_share(sample)

    # 2. column-name keywords (validated against the values where cheap)
    if name_rule:
        role, kind, ladder, subtype = name_rule
        if subtype == "free_text" and mean_len < 15:
            name_rule = None
        elif subtype == "age" and not is_numeric:
            name_rule = None
        elif subtype == "birth_date" and date_share < MATCH:
            if is_numeric:
                return prof(Role.QUASI_ID, Kind.NUMERIC, "age", 0.8, "name suggests birth year and values are numeric", "birth_year")
            name_rule = None
        elif kind == Kind.ID and role == Role.DIRECT_ID and is_numeric and subtype in ("person_name", "address"):
            name_rule = None
    if name_rule:
        role, kind, ladder, subtype = name_rule
        if subtype == "birth_date":
            ladder = "birthdate"
        if kind == Kind.CATEGORICAL and is_numeric and unique_ratio > 0.5:
            ladder = "numeric"
        if subtype == "postcode" and not is_numeric and _share(sample, RE["postcode"]) < 0.5 and unique_ratio > 0.9:
            ladder = "categorical"
        return prof(role, kind, ladder, 0.85, f"column name contains '{_hit(toks, name)}'", subtype)

    # 3. dtype + cardinality
    if date_share >= MATCH:
        ladder = "timestamp" if has_time else "date"
        return prof(Role.QUASI_ID, Kind.DATE, ladder, 0.8,
                    f"{date_share:.0%} of values parse as {'timestamps' if has_time else 'dates'}; event dates are quasi-identifiers",
                    "event_date")
    if not is_numeric and (mean_len > 40 or mean_tokens >= 6):
        return prof(Role.FREE_TEXT, Kind.TEXT, None, 0.8, f"long strings (mean {mean_len:.0f} chars, {mean_tokens:.0f} tokens)",
                    "free_text")
    is_id_name = any(t in ID_TOKENS for t in toks[:-1]) or name.lower().endswith("id")
    if unique_ratio > 0.95 and (not is_numeric or is_id_name):
        if is_id_name:
            return prof(Role.TECHNICAL_ID, Kind.ID, None, 0.8, f"{unique_ratio:.0%} unique and name looks like a key", "record_key")
        return prof(Role.DIRECT_ID, Kind.ID, None, 0.6, f"{unique_ratio:.0%} unique strings: likely an identifier", "identifier")
    if is_id_name and ((not is_numeric and sample.str.contains(r"\d").mean() > 0.8)
                       or (unique_ratio > 0.2 and _integer_like(sample))):
        return prof(Role.TECHNICAL_ID, Kind.ID, None, 0.65, "name looks like a key / foreign key", "record_key")

    code_share = _share(sample, RE["icd"])
    if code_share >= MATCH:
        return prof(Role.SENSITIVE, Kind.CATEGORICAL, "code", 0.8,
                    f"{code_share:.0%} of values look like ICD-10 codes (health data, GDPR Art. 9)", "diagnosis_code")

    # 5. sensitive keywords
    hit = next((w for w in SENSITIVE_WORDS if w in name.lower()), None)
    if hit:
        if is_numeric:
            return prof(Role.SENSITIVE, Kind.NUMERIC, "numeric", 0.75, f"name contains sensitive term '{hit}'", "sensitive_amount")
        return prof(Role.SENSITIVE, Kind.CATEGORICAL, "categorical", 0.75, f"name contains sensitive term '{hit}'",
                    "sensitive_category")

    if is_numeric:
        return prof(Role.NUMERIC, Kind.NUMERIC, "numeric", 0.7, "numeric measurement", "measurement")
    n_distinct = int(non_null.nunique())
    if n_distinct <= max(50, int(0.05 * len(non_null))):
        return prof(Role.CATEGORICAL, Kind.CATEGORICAL, "categorical", 0.7, f"{n_distinct} distinct values", "category")
    return prof(Role.OTHER, Kind.CATEGORICAL, "categorical", 0.4,
                f"high-cardinality strings ({n_distinct} distinct); review manually", "unknown")


def _hit(toks: list[str], name: str) -> str:
    for words, *_ in NAME_RULES:
        for t in toks:
            if t in words:
                return t
    return name


def detect_domain(columns: list[str]) -> tuple[str | None, int]:
    toks = [t for c in columns for t in tokens(c)[:-1]]
    scores = {d: sum(any(t.startswith(w) for w in words) for t in toks) for d, words in DOMAIN_HINTS.items()}
    best = max(scores, key=scores.get)
    return (best, scores[best]) if scores[best] >= 2 else (None, 0)


def profile_dataframe(df: pd.DataFrame, seed: int = 0) -> list[ColumnProfile]:
    return [profile_column(c, df[c], seed) for c in df.columns]
