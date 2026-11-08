"""Transformation ladders (spec §5.5). Level 0 = unchanged; each level is deterministic.

A ladder is fitted once on the original column (frequencies, quantile edges) and can then
be applied to any series with the same meaning, e.g. a train or test split.

Every label uses a notation the starter kit's `pets-eval` parses: ISO dates, `2025-03`, `2025-W12`, `2025`,
`2025-03-12 14:00`, ranges `30-39`, `80+`, `<20`, masked postcodes `201**`. Quarters are therefore not a level.
Birth dates have their own ladder that ends in age bands and is released as an `age` column (as in the kit's
quickstart), so a birth date can be coarsened far enough to reach k >= 5 while age analyses keep working.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .profiling import parse_dates, to_number

NA = pd.NA


def _mask(m) -> np.ndarray:
    """Plain numpy bool mask (NA -> False). Boolean-Series indexing with list assignment differs across pandas/numpy
    versions; .loc with a numpy mask behaves the same everywhere."""
    if isinstance(m, pd.Series):
        m = m.fillna(False) if m.dtype == object or str(m.dtype) == "boolean" else m
        return m.to_numpy(dtype=bool)
    return np.asarray(m, dtype=bool)


def _entropy(s: pd.Series) -> float:
    p = s.astype("string").fillna("<NA>").value_counts(normalize=True).to_numpy()
    return float(-(p * np.log(p)).sum()) if len(p) else 0.0


class Ladder:
    name = "base"
    level_names: list[str] = ["exact"]
    numeric_capable = False      # can map generalised values back to numbers (for correlations / models)
    entropy_loss = False         # information loss measured by entropy instead of level / max_level

    def __init__(self, column: str):
        self.column = column
        self._h0: float | None = None
        self._loss: dict[int, float] = {}

    @property
    def max_level(self) -> int:
        return len(self.level_names) - 1

    def fit(self, s: pd.Series) -> "Ladder":
        self._fit_series = s
        return self

    def apply(self, s: pd.Series, level: int) -> pd.Series:
        raise NotImplementedError

    def numeric(self, s: pd.Series, level: int) -> pd.Series | None:
        """Numeric representation of the column at a given level (bin centres, date midpoints)."""
        return None

    # Tolerant matching (risk.tolerant_linkage): an attacker's approximate value is compared with the interval a
    # released value stands for. Ladders that do not support it return None.
    def tolerance_value(self, s: pd.Series) -> pd.Series | None:
        """The attacker's exact value on the tolerance scale (hours for times, years for ages)."""
        return None

    def interval(self, s: pd.Series, level: int) -> tuple[pd.Series, pd.Series] | None:
        """[lo, hi] on the tolerance scale that each released value (original value at `level`) stands for."""
        return None

    def describe(self, level: int) -> str:
        if level <= 0:
            return "unchanged"
        return f"{self.name} → {self.level_names[min(level, self.max_level)]}"

    def output_name(self, level: int) -> str:
        """Column name in the released table (a ladder may release a derived column, e.g. age instead of DOB)."""
        return self.column

    def drops(self, level: int) -> bool:
        """True when this level removes the column from the release altogether."""
        return False

    def level_index(self, name: str | int) -> int:
        """Level from its index or (case-insensitive prefix of) its name."""
        if isinstance(name, int) or str(name).isdigit():
            return min(int(name), self.max_level)
        key = str(name).strip().lower()
        for i, lab in enumerate(self.level_names):
            if lab.lower() == key or lab.lower().startswith(key):
                return i
        raise ValueError(f"{self.column}: unknown level {name!r}; levels are {self.level_names}")

    def info_loss(self, level: int) -> float:
        if level <= 0:
            return 0.0
        if not self.entropy_loss:
            return min(level, self.max_level) / self.max_level
        if level not in self._loss:
            s = self._fit_series
            if self._h0 is None:
                self._h0 = _entropy(s)
            h = _entropy(self.apply(s, level))
            self._loss[level] = 0.0 if self._h0 <= 0 else float(np.clip(1 - h / self._h0, 0, 1))
        return self._loss[level]


# ---------------------------------------------------------------- dates / timestamps
class DateLadder(Ladder):
    name = "date"
    level_names = ["exact", "week", "month", "year"]
    numeric_capable = True
    _steps = [None, "W", "M", "Y"]

    def apply(self, s: pd.Series, level: int) -> pd.Series:
        if level <= 0:
            return s
        d = parse_dates(s)
        step = self._steps[min(level, self.max_level)]
        out = pd.Series(pd.NA, index=s.index, dtype="string")
        ok = _mask(d.notna())
        if not ok.any():
            return out
        dd = d[ok]
        if step == "h":
            lab = dd.dt.strftime("%Y-%m-%d %H:00")
        elif step == "D":
            lab = dd.dt.strftime("%Y-%m-%d")
        elif step == "W":
            iso = dd.dt.isocalendar()
            lab = iso["year"].astype(str) + "-W" + iso["week"].astype(int).map("{:02d}".format)
        elif step == "M":
            lab = dd.dt.strftime("%Y-%m")
        elif step == "Q":
            lab = dd.dt.year.astype(str) + "-Q" + dd.dt.quarter.astype(str)
        elif step == "Y":
            lab = dd.dt.year.astype(str)
        else:
            width = 5 if step == "5Y" else 10
            lo = (dd.dt.year // width) * width
            lab = lo.astype(str) + "-" + (lo + width - 1).astype(str)
        out.loc[_mask(ok)] = lab.astype("string").to_numpy()
        return out

    def numeric(self, s: pd.Series, level: int) -> pd.Series:
        """Days since 1970-01-01 of the period midpoint (exact date at level 0)."""
        d = parse_dates(s) if level <= 0 else None
        if level <= 0:
            return (d - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0
        labels = self.apply(s, level)
        uniq = labels.dropna().unique()
        mids = {lab: self._midpoint(lab, self._steps[min(level, self.max_level)]) for lab in uniq}
        return labels.map(mids).astype(float)

    def tolerance_value(self, s: pd.Series) -> pd.Series:
        return (parse_dates(s) - pd.Timestamp("1970-01-01")).dt.total_seconds() / 3600.0

    def interval(self, s: pd.Series, level: int) -> tuple[pd.Series, pd.Series]:
        if level <= 0:
            v = self.tolerance_value(s)
            return v, v
        step = self._steps[min(level, self.max_level)]
        labels = self.apply(s, level)
        bounds = {lab: self._period(lab, step) for lab in labels.dropna().unique()}
        epoch = pd.Timestamp("1970-01-01")
        lo = labels.map({k: (a - epoch).total_seconds() / 3600.0 for k, (a, b) in bounds.items()}).astype(float)
        hi = labels.map({k: (b - epoch).total_seconds() / 3600.0 for k, (a, b) in bounds.items()}).astype(float)
        return lo, hi

    @staticmethod
    def _midpoint(label: str, step: str) -> float:
        start, end = DateLadder._period(label, step)
        return ((start - pd.Timestamp("1970-01-01")) + (end - start) / 2).total_seconds() / 86400.0

    @staticmethod
    def _period(label: str, step: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        if step in ("h",):
            start = pd.Timestamp(label)
            end = start + pd.Timedelta(hours=1)
        elif step == "D":
            start = pd.Timestamp(label)
            end = start + pd.Timedelta(days=1)
        elif step == "W":
            y, w = label.split("-W")
            start = pd.Timestamp.fromisocalendar(int(y), int(w), 1)
            end = start + pd.Timedelta(days=7)
        elif step == "M":
            p = pd.Period(label, "M")
            start, end = p.start_time, p.end_time
        elif step == "Q":
            p = pd.Period(label.replace("-", ""), "Q")
            start, end = p.start_time, p.end_time
        elif step == "Y":
            start = pd.Timestamp(f"{label}-01-01")
            end = pd.Timestamp(f"{int(label) + 1}-01-01")
        else:
            lo, hi = label.split("-")
            start = pd.Timestamp(f"{lo}-01-01")
            end = pd.Timestamp(f"{int(hi) + 1}-01-01")
        return start, end


class TimestampLadder(DateLadder):
    name = "timestamp"
    level_names = ["exact", "hour", "day", "week", "month", "year"]
    _steps = [None, "h", "D", "W", "M", "Y"]


class BirthDateLadder(Ladder):
    """Birth date: month → year → age bands. From the first age level on, the released column is `age`
    (age at the reference date: the latest event date in the table, or today)."""
    name = "birth date"
    level_names = ["exact", "month", "year", "age, 5-year bands", "age, 10-year bands",
                   "age, 10-year bands, 80+ top-coded", "age, 20-year bands"]
    numeric_capable = True
    AGE_FROM = 3

    def __init__(self, column: str, reference=None, age_name: str = "age"):
        super().__init__(column)
        self.reference = reference
        self.age_name = age_name

    def fit(self, s: pd.Series) -> "Ladder":
        super().fit(s)
        self.ref = pd.Timestamp(self.reference) if self.reference is not None else pd.Timestamp.today().normalize()
        return self

    def output_name(self, level: int) -> str:
        return self.age_name if level >= self.AGE_FROM else self.column

    def _age(self, s: pd.Series) -> pd.Series:
        return ((self.ref - parse_dates(s)).dt.days / 365.25).clip(lower=0)

    def tolerance_value(self, s: pd.Series) -> pd.Series:
        return self._age(s)

    def interval(self, s: pd.Series, level: int) -> tuple[pd.Series, pd.Series]:
        level = min(level, self.max_level)
        if level <= 0:
            a = self._age(s)
            return a, a
        if level <= 2:  # month / year of birth -> range of possible ages
            d = parse_dates(s)
            per = d.dt.to_period("M" if level == 1 else "Y")
            start, end = per.dt.start_time, per.dt.end_time
            return ((self.ref - end).dt.days / 365.25).clip(lower=0), ((self.ref - start).dt.days / 365.25).clip(lower=0)
        lo, hi = self._band(self._age(s), level)
        return lo.astype(float), (hi.fillna(lo + 29) + 1).astype(float)

    def _band(self, age: pd.Series, level: int) -> tuple[pd.Series, pd.Series]:
        width = {3: 5, 4: 10, 5: 10, 6: 20}[min(level, self.max_level)]
        lo = np.floor(age / width) * width
        hi = lo + width - 1
        if level == 5:
            top = lo >= 80
            lo, hi = lo.mask(top, 80), hi.mask(top, np.nan)
        return lo, hi

    def apply(self, s: pd.Series, level: int) -> pd.Series:
        if level <= 0:
            return s
        level = min(level, self.max_level)
        if level == 1:
            return parse_dates(s).dt.strftime("%Y-%m").astype("string")
        if level == 2:
            return parse_dates(s).dt.year.astype("Int64").astype("string")
        lo, hi = self._band(self._age(s), level)
        lab = pd.Series(pd.NA, index=s.index, dtype="string")
        ok = _mask(lo.notna())
        lab.loc[ok] = [f"{int(a)}+" if pd.isna(b) else f"{int(a)}-{int(b)}" for a, b in zip(lo[ok], hi[ok])]
        return lab

    def numeric(self, s: pd.Series, level: int) -> pd.Series:
        """Days since 1970-01-01 of the birth-date midpoint implied by the released value."""
        d = parse_dates(s)
        epoch = pd.Timestamp("1970-01-01")
        if level <= 0:
            return (d - epoch).dt.total_seconds() / 86400.0
        if level == 1:
            mid = d.dt.to_period("M").dt.start_time + pd.Timedelta(days=15)
            return (mid - epoch).dt.total_seconds() / 86400.0
        if level == 2:
            mid = pd.to_datetime(d.dt.year.astype("Int64").astype("string") + "-07-02", errors="coerce")
            return (mid - epoch).dt.total_seconds() / 86400.0
        lo, hi = self._band(self._age(s), level)
        mid_age = (lo + hi.fillna(lo + 10) + 1) / 2
        return ((self.ref - epoch).days - mid_age * 365.25).astype(float)


# ---------------------------------------------------------------- numbers
def _fmt(x: float) -> str:
    if x == int(x):
        return str(int(x))
    return f"{x:.3g}"


class AgeLadder(Ladder):
    """Age or other numeric quasi-identifier: fixed-width bins, then top/bottom-coded wide bins."""
    name = "age"
    level_names = ["exact", "bins of 5", "bins of 10", "bins of 20", "wide bins, top/bottom-coded"]
    numeric_capable = True
    widths = [None, 5, 10, 20, 20]

    def fit(self, s: pd.Series) -> "Ladder":
        super().fit(s)
        v = to_number(s).dropna()
        rng = float(v.max() - v.min()) if len(v) else 1.0
        if 20 <= rng <= 200:
            self.unit = 1.0
        elif rng > 200:
            self.unit = 10.0 ** math.floor(math.log10(rng / 100))
        else:
            self.unit = 10.0 ** math.floor(math.log10(max(rng, 1e-9) / 20))
        self.integer = bool(len(v)) and bool((v == v.round()).all())
        w = self.widths[-1] * self.unit
        # tails beyond the 5th / 95th percentile (snapped inwards to the bin grid) are coded "<lo" and "hi+"
        lo = math.ceil(float(v.quantile(0.05)) / w) * w if len(v) else 0.0
        hi = math.floor(float(v.quantile(0.95)) / w) * w if len(v) else 0.0
        self.bottom, self.top = lo, max(hi, lo + w)
        self.bottom_mid = float(v[v < self.bottom].median()) if (v < self.bottom).any() else self.bottom
        self.top_mid = float(v[v >= self.top].median()) if (v >= self.top).any() else self.top
        return self

    def _bin(self, v: pd.Series, level: int):
        w = self.widths[min(level, self.max_level)] * self.unit
        lo = np.floor(v / w) * w
        return lo, w

    def apply(self, s: pd.Series, level: int) -> pd.Series:
        if level <= 0:
            return s
        v = to_number(s)
        lo, w = self._bin(v, level)
        step = self.unit if self.integer else 0
        lab = pd.Series(pd.NA, index=s.index, dtype="string")
        ok = _mask(v.notna())
        lab.loc[ok] = [f"{_fmt(a)}-{_fmt(a + w - step)}" if self.integer else f"[{_fmt(a)}, {_fmt(a + w)})"
                       for a in lo[ok]]
        if level >= 4:
            lab.loc[ok & _mask(v < self.bottom)] = f"<{_fmt(self.bottom)}"
            lab.loc[ok & _mask(v >= self.top)] = f"{_fmt(self.top)}+"
        return lab

    def tolerance_value(self, s: pd.Series) -> pd.Series:
        return to_number(s)

    def interval(self, s: pd.Series, level: int) -> tuple[pd.Series, pd.Series]:
        v = to_number(s)
        if level <= 0:
            return v, v
        lo, w = self._bin(v, level)
        hi = lo + w
        if level >= 4:
            lo = lo.mask(v < self.bottom, -np.inf)
            hi = hi.mask(v < self.bottom, self.bottom)
            lo = lo.mask(v >= self.top, self.top)
            hi = hi.mask(v >= self.top, np.inf)
        return lo.astype(float), hi.astype(float)

    def numeric(self, s: pd.Series, level: int) -> pd.Series:
        v = to_number(s)
        if level <= 0:
            return v
        lo, w = self._bin(v, level)
        mid = lo + (w - (self.unit if self.integer else 0)) / 2
        if level >= 4:
            mid = mid.mask(v < self.bottom, self.bottom_mid).mask(v >= self.top, self.top_mid)
        return mid.astype(float)


class NumericLadder(Ladder):
    """Other numeric: 2 significant figures, then 10 and 4 quantile bins."""
    name = "numeric"
    level_names = ["exact", "2 significant figures", "10 quantile bins", "4 quantile bins"]
    numeric_capable = True

    def fit(self, s: pd.Series) -> "Ladder":
        super().fit(s)
        v = to_number(s).dropna()
        self.edges, self.centres = {}, {}
        for level, q in ((2, 10), (3, 4)):
            edges = np.unique(np.nanquantile(v, np.linspace(0, 1, q + 1))) if len(v) else np.array([0.0, 1.0])
            if len(edges) < 2:
                edges = np.array([edges[0], edges[0] + 1])
            self.edges[level] = edges
            idx = np.clip(np.searchsorted(edges[1:-1], v.to_numpy(), side="right"), 0, len(edges) - 2)
            self.centres[level] = {i: float(np.median(v.to_numpy()[idx == i])) if (idx == i).any() else
                                   float((edges[i] + edges[i + 1]) / 2) for i in range(len(edges) - 1)}
        return self

    @staticmethod
    def _sig2(v: pd.Series) -> pd.Series:
        a = v.abs().replace(0, np.nan)
        digits = (1 - np.floor(np.log10(a))).fillna(0).astype(int)
        return pd.Series([round(x, d) if pd.notna(x) else np.nan for x, d in zip(v, digits)], index=v.index)

    def _idx(self, v: pd.Series, level: int) -> np.ndarray:
        edges = self.edges[level]
        return np.clip(np.searchsorted(edges[1:-1], v.to_numpy(), side="right"), 0, len(edges) - 2)

    def apply(self, s: pd.Series, level: int) -> pd.Series:
        if level <= 0:
            return s
        v = to_number(s)
        if level == 1:
            return self._sig2(v).map(lambda x: _fmt(x) if pd.notna(x) else pd.NA).astype("string")
        edges = self.edges[min(level, 3)]
        idx = self._idx(v, min(level, 3))
        labels = [f"[{_fmt(edges[i])}, {_fmt(edges[i + 1])}{']' if i == len(edges) - 2 else ')'}" for i in range(len(edges) - 1)]
        out = pd.Series([labels[i] for i in idx], index=s.index, dtype="string")
        out.loc[_mask(v.isna())] = pd.NA
        return out

    def numeric(self, s: pd.Series, level: int) -> pd.Series:
        v = to_number(s)
        if level <= 0:
            return v
        if level == 1:
            return self._sig2(v).astype(float)
        lvl = min(level, 3)
        idx = self._idx(v, lvl)
        out = pd.Series([self.centres[lvl][i] for i in idx], index=s.index, dtype=float)
        out.loc[_mask(v.isna())] = np.nan
        return out


class LatLonLadder(Ladder):
    name = "coordinates"
    level_names = ["exact", "3 decimals (~100 m)", "2 decimals (~1 km)", "1 decimal (~10 km)"]
    numeric_capable = True

    def apply(self, s: pd.Series, level: int) -> pd.Series:
        if level <= 0:
            return s
        dec = 4 - min(level, self.max_level)
        v = to_number(s).round(dec)
        return v.map(lambda x: f"{x:.{dec}f}" if pd.notna(x) else pd.NA).astype("string")

    def numeric(self, s: pd.Series, level: int) -> pd.Series:
        v = to_number(s)
        return v if level <= 0 else v.round(4 - min(level, self.max_level))


# ---------------------------------------------------------------- categorical
class PostcodeLadder(Ladder):
    name = "postcode"
    level_names = ["exact", "drop last character", "drop last 2 characters", "region prefix only"]
    entropy_loss = True

    def apply(self, s: pd.Series, level: int) -> pd.Series:
        if level <= 0:
            return s
        level = min(level, self.max_level)

        def mask(v):
            if pd.isna(v):
                return pd.NA
            v = str(v).strip()
            n = len(v)
            keep = {1: n - 1, 2: n - 2, 3: n - 3 if n >= 5 else 1}[level]
            keep = max(1, keep)
            return v[:keep] + "*" * (n - keep)

        return s.map(mask).astype("string")


class CategoricalLadder(Ladder):
    """Categories: merge rare ones into 'Other' (< 1 %, then < 5 %), an optional user hierarchy, then for
    many-valued columns (employer, occupation) keep only the top 20 / top 5 categories, then suppress every value
    ('*'), and finally drop the column. Levels that change nothing for a column are skipped by the search."""
    name = "categories"
    entropy_loss = True

    def __init__(self, column: str, hierarchy: dict[str, str] | None = None):
        super().__init__(column)
        self.hierarchy = hierarchy or {}
        self.steps: list[tuple] = [("exact",), ("rare", 0.01), ("rare", 0.05)]
        self.level_names = ["exact", "merge categories < 1 % into Other", "merge categories < 5 % into Other"]
        if self.hierarchy:
            self.steps.append(("hierarchy",))
            self.level_names.append("user-provided hierarchy")
        self.steps += [("top", 20), ("top", 5), ("star",), ("drop",)]
        self.level_names += ["top 20 categories + Other", "top 5 categories + Other", "* (all values suppressed)",
                             "column dropped"]

    def fit(self, s: pd.Series) -> "Ladder":
        super().fit(s)
        self.freq = s.astype("string").value_counts(normalize=True, dropna=True)
        return self

    def drops(self, level: int) -> bool:
        return self.steps[min(level, self.max_level)][0] == "drop"

    def apply(self, s: pd.Series, level: int) -> pd.Series:
        if level <= 0:
            return s
        step = self.steps[min(level, self.max_level)]
        v = s.astype("string")
        if step[0] == "hierarchy":
            return v.map(lambda x: self.hierarchy.get(x, "Other") if pd.notna(x) else pd.NA).astype("string")
        if step[0] in ("star", "drop"):   # a dropped column carries no information either: '*' internally
            return v.where(v.isna(), "*")
        keep = set(self.freq[self.freq >= step[1]].index) if step[0] == "rare" else set(self.freq.index[:step[1]])
        return v.where(v.isin(keep) | v.isna(), "Other")


class CodeLadder(Ladder):
    """Hierarchical codes such as ICD-10.

    1: only rare codes (< k records) are recoded up the hierarchy (J45.9 → J45 → J → Other) until they are common;
    2: every code to its 3-character category; 3: chapter letter; 4: chapter, rare chapters (< 5 %) into Other.
    """
    name = "code"
    level_names = ["exact", "rare codes recoded up the hierarchy", "3-character category", "chapter letter",
                   "chapter, rare (< 5 %) merged into Other"]
    entropy_loss = True

    def __init__(self, column: str, k: int = 5):
        super().__init__(column)
        self.k = k

    def fit(self, s: pd.Series) -> "Ladder":
        super().fit(s)
        v = s.astype("string").str.strip().str.upper()
        ch = v.str[:1]
        self.chapter_freq = ch.value_counts(normalize=True)
        # walk rare values up the hierarchy (code -> 3-char -> chapter -> Other) until every group has >= k records
        mapping = {c: c for c in v.dropna().unique()}
        for _ in range(4):
            counts = v.map(mapping).value_counts()
            rare = {g for g, n in counts.items() if n < self.k and g != "Other"}
            if not rare:
                break
            for code, g in mapping.items():
                if g in rare:
                    mapping[code] = g[:3] if len(g) > 3 else (g[:1] if len(g) > 1 else "Other")
        self.recode = {c: g for c, g in mapping.items() if g != c}
        return self

    def apply(self, s: pd.Series, level: int) -> pd.Series:
        if level <= 0:
            return s
        v = s.astype("string").str.strip().str.upper()
        if level == 1:
            return v.map(lambda x: self.recode.get(x, x) if pd.notna(x) else pd.NA).astype("string")
        if level == 2:
            return v.str[:3]
        ch = v.str[:1]
        if level == 3:
            return ch
        keep = set(self.chapter_freq[self.chapter_freq >= 0.05].index)
        return ch.where(ch.isin(keep) | ch.isna(), "Other")


def make_ladder(column: str, ladder: str | None, hierarchy: dict[str, str] | None = None, k: int = 5,
                reference=None, age_name: str = "age") -> Ladder | None:
    if hierarchy:
        return CategoricalLadder(column, hierarchy)
    if ladder == "code":
        return CodeLadder(column, k)
    if ladder == "birthdate":
        return BirthDateLadder(column, reference, age_name)
    return {
        "date": lambda: DateLadder(column),
        "timestamp": lambda: TimestampLadder(column),
        "age": lambda: AgeLadder(column),
        "numeric": lambda: NumericLadder(column),
        "latlon": lambda: LatLonLadder(column),
        "postcode": lambda: PostcodeLadder(column),
        "categorical": lambda: CategoricalLadder(column),
        "code": lambda: CodeLadder(column),
    }.get(ladder or "", lambda: None)()


class DroppableLadder(Ladder):
    """Any ladder plus a final 'column dropped' level, for quasi-identifiers: when no generalisation is enough, the
    column is left out of the release (full information loss in the utility proxy; '*' for risk computations)."""
    entropy_loss = False

    def __init__(self, inner: Ladder):
        super().__init__(inner.column)
        self.inner = inner
        self.name = inner.name
        self.level_names = list(inner.level_names) + ["column dropped"]
        self.numeric_capable = inner.numeric_capable

    @property
    def _drop(self) -> int:
        return len(self.level_names) - 1

    def fit(self, s: pd.Series) -> "Ladder":
        self.inner.fit(s)
        self._fit_series = s
        return self

    def drops(self, level: int) -> bool:
        return level >= self._drop

    def apply(self, s: pd.Series, level: int) -> pd.Series:
        return s.astype("string").where(s.isna(), "*") if self.drops(level) else self.inner.apply(s, level)

    def numeric(self, s: pd.Series, level: int) -> pd.Series | None:
        if self.drops(level):
            return pd.Series(np.nan, index=s.index, dtype=float)
        return self.inner.numeric(s, level)

    def tolerance_value(self, s: pd.Series):
        return self.inner.tolerance_value(s)

    def interval(self, s: pd.Series, level: int):
        if self.drops(level):
            return pd.Series(-np.inf, index=s.index), pd.Series(np.inf, index=s.index)
        return self.inner.interval(s, level)

    def info_loss(self, level: int) -> float:
        return 1.0 if self.drops(level) else self.inner.info_loss(level)

    def output_name(self, level: int) -> str:
        return self.inner.output_name(min(level, self._drop - 1))

    def describe(self, level: int) -> str:
        return f"{self.name} → column dropped" if self.drops(level) else self.inner.describe(level)
