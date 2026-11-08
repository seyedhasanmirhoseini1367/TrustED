"""Role, kind and gate enums plus the dataclasses shared by the whole engine."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    DIRECT_ID = "DIRECT_ID"
    QUASI_ID = "QUASI_ID"
    SENSITIVE = "SENSITIVE"
    DATE = "DATE"
    LOCATION = "LOCATION"
    FREE_TEXT = "FREE_TEXT"
    TECHNICAL_ID = "TECHNICAL_ID"
    NUMERIC = "NUMERIC"
    CATEGORICAL = "CATEGORICAL"
    TARGET = "TARGET"
    OTHER = "OTHER"


class Kind(str, Enum):
    DATE = "date"
    LOCATION = "location"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    ID = "id"


class Gate(str, Enum):
    RED = "red"        # fixed transform, excluded from search
    YELLOW = "yellow"  # search space
    GREEN = "green"    # unchanged by default


class RedAction(str, Enum):
    DROP = "drop"
    PSEUDONYMISE = "pseudonymise"
    TRUNCATE_IP = "truncate_ip"
    KEEP = "keep"


# Ladder names (see ladders.py). None means the column has no generalisation ladder.
LADDERS = ("date", "timestamp", "birthdate", "age", "numeric", "latlon", "postcode", "categorical", "code")

RECIPIENTS = {
    "public": {"label": "Public release", "max_linkage": 0.01, "min_k": 10},
    "partner": {"label": "Trusted research partner", "max_linkage": 0.05, "min_k": 5},
    "internal": {"label": "Internal analytics team", "max_linkage": 0.10, "min_k": 3},
}

TASK_TYPES = ("classification", "regression", "descriptive", "visualization")
PREDICTION = ("classification", "regression")


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    role: Role
    kind: Kind
    ladder: str | None
    confidence: float
    reason: str
    unique_ratio: float = 0.0
    missing_ratio: float = 0.0
    examples: list[str] = field(default_factory=list)
    subtype: str | None = None          # e.g. email, phone, national_id, ip, postcode, birth_date
    gate: Gate | None = None
    red_action: RedAction | None = None
    gate_reason: str = ""
    min_level: int = 0                  # floor for yellow columns (e.g. rare sensitive categories)
    overridden: bool = False            # set when the user edits role/gate in the UI
    llm_suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("role", "kind", "gate", "red_action"):
            if d[key] is not None:
                d[key] = d[key].value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ColumnProfile":
        d = dict(d)
        d["role"] = Role(d["role"])
        d["kind"] = Kind(d["kind"])
        d["gate"] = Gate(d["gate"]) if d.get("gate") else None
        d["red_action"] = RedAction(d["red_action"]) if d.get("red_action") else None
        known = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Scenario:
    recipient: str = "partner"
    attacker_qis: list[str] = field(default_factory=list)
    task_type: str = "classification"
    target: str | None = None
    key_columns: list[str] = field(default_factory=list)
    max_linkage: float | None = None    # None -> recipient default
    min_k: int | None = None            # None -> recipient default
    description: str = ""
    seed: int = 42
    attacker_fraction: float = 0.30
    suppression_cap: float = 0.05
    promote_green: bool = True
    derive_knowledge: bool = True
    hierarchies: dict[str, dict[str, str]] = field(default_factory=dict)
    # utility requirements: the coarsest level each column may reach (level index or level name, e.g. "week")
    max_levels: dict[str, int | str] = field(default_factory=dict)
    kit_scenario: str | None = None                       # official starter-kit scenario id, e.g. HEALTH-S1
    kit_tasks: list[str] = field(default_factory=list)    # kit utility tasks this release must keep answerable
    sacrificed_tasks: list[str] = field(default_factory=list)
    pseudonymise: list[str] = field(default_factory=list)  # red columns that must stay linkable (keyed pseudonym)
    # several purposes at once (e.g. a prediction model AND charts); task_type/target stay the primary purpose
    purposes: list[str] = field(default_factory=list)
    targets: dict[str, list[str]] = field(default_factory=dict)  # prediction purpose -> its target columns

    def __post_init__(self) -> None:
        if self.recipient not in RECIPIENTS:
            raise ValueError(f"unknown recipient {self.recipient!r}; choose one of {list(RECIPIENTS)}")
        if not self.purposes:
            self.purposes = [self.task_type]
        for t in self.purposes:
            if t not in TASK_TYPES:
                raise ValueError(f"unknown purpose {t!r}; choose from {TASK_TYPES}")
        if not self.targets and self.target and self.task_type in PREDICTION:
            self.targets = {self.task_type: [self.target]}
        # one or several target columns per prediction purpose (a single string is accepted too)
        norm = {}
        for k, v in self.targets.items():
            cols = [v] if isinstance(v, str) else list(v or [])
            cols = list(dict.fromkeys(c for c in cols if c))
            if k in self.purposes and k in PREDICTION and cols:
                norm[k] = cols
        self.targets = norm
        # primary purpose / target: the first prediction purpose with a target, else the first purpose
        primary = next((t for t in self.purposes if t in self.targets), self.purposes[0])
        self.task_type = primary
        self.target = self.targets[primary][0] if primary in self.targets else None
        defaults = RECIPIENTS[self.recipient]
        if self.max_linkage is None:
            self.max_linkage = defaults["max_linkage"]
        if self.min_k is None:
            self.min_k = defaults["min_k"]

    @property
    def prediction_tasks(self) -> list[tuple[str, str]]:
        """(purpose, target) for every selected prediction purpose that has a target."""
        return [(t, col) for t in self.purposes for col in self.targets.get(t, [])]

    @property
    def all_targets(self) -> set[str]:
        return {c for cols in self.targets.values() for c in cols}

    @property
    def recipient_label(self) -> str:
        return RECIPIENTS[self.recipient]["label"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Scenario":
        known = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in known})
