"""End-to-end run: gates → threat model → ladders → text redaction → greedy search → metrics → report."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import __version__
from .gates import assign_gates, attacker_known, column_groups, derive_knowledge, promote_green
from .ladders import DroppableLadder, Ladder, make_ladder
from .profiling import detect_domain, parse_dates, profile_dataframe
from .risk import k_metrics, l_diversity
from .roles import ColumnProfile, Gate, RedAction, Role, Scenario
from .search import GreedySearch, SUPPRESS
from .text_pii import Redactor, evaluate as evaluate_text
from .transforms import Plan, apply_plan, get_secret
from .utility import ProxyUtility, descriptive_utility, task_utility

PET_NAMES = {
    "date": "Date generalisation", "timestamp": "Timestamp rounding", "age": "Binning",
    "numeric": "Rounding / quantile binning", "latlon": "Location precision reduction",
    "postcode": "Postcode truncation", "categorical": "Category merging", "code": "Code hierarchy generalisation",
}
RED_NAMES = {RedAction.DROP: "Suppression (column dropped)", RedAction.PSEUDONYMISE: "Pseudonymisation (HMAC-SHA256)",
             RedAction.TRUNCATE_IP: "IP truncation (/24, /48)", RedAction.KEEP: "Kept (user override)"}


@dataclass
class RunResult:
    protected: pd.DataFrame
    report: dict
    preview: dict
    text_examples: list[dict] = field(default_factory=list)
    rowmap: pd.DataFrame | None = None            # kit scoring aid: protected_row -> original_record_id (never publish)
    renames: dict[str, str] = field(default_factory=dict)

    def export_table(self) -> pd.DataFrame:
        """The release as written to disk: derived columns renamed (e.g. date_of_birth -> age)."""
        return self.protected.rename(columns=self.renames)


def load_table(path, sheet=None) -> pd.DataFrame:
    """Read every column as string (phone numbers, postcodes and ids break when parsed as numbers)."""
    p = str(path).lower()
    if p.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, dtype=str, sheet_name=sheet or 0)
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=True, encoding="utf-8", encoding_errors="replace")
    return df.reset_index(drop=True)


def reference_date(df: pd.DataFrame, profiles: list[ColumnProfile]) -> pd.Timestamp | None:
    """Latest event date in the table: ages are computed at this date (None -> today)."""
    latest = None
    for p in profiles:
        if p.ladder in ("date", "timestamp") and p.subtype != "birth_date" and p.name in df.columns:
            d = parse_dates(df[p.name]).max()
            if pd.notna(d) and (latest is None or d > latest):
                latest = d
    return latest.normalize() if latest is not None else None


def fit_ladders(df: pd.DataFrame, profiles: list[ColumnProfile], scenario: Scenario) -> dict[str, Ladder]:
    ladders = {}
    ref = reference_date(df, profiles)
    age_name = "age_band" if "age" in df.columns else "age"
    for p in profiles:
        if p.gate == Gate.RED or p.role == Role.FREE_TEXT or p.ladder is None:
            continue
        lad = make_ladder(p.name, p.ladder, scenario.hierarchies.get(p.name), int(scenario.min_k), ref, age_name)
        if lad is not None:
            # quasi-identifiers may be dropped as a last resort (categorical ladders already end with that level)
            if p.role in (Role.QUASI_ID, Role.DATE, Role.LOCATION) and not lad.drops(lad.max_level):
                lad = DroppableLadder(lad)
            ladders[p.name] = lad.fit(df[p.name])
    return ladders


def person_names(df: pd.DataFrame, profiles: list[ColumnProfile]) -> list[str]:
    """Name tokens from direct-identifier name columns: a dataset dictionary for the free-text recognizer."""
    names = set()
    for p in profiles:
        if p.role == Role.DIRECT_ID and p.subtype == "person_name":
            vals = df[p.name].dropna().astype(str)
            names.update(vals.str.strip())
            names.update(t for v in vals for t in v.split() if len(t) >= 3)
    return sorted(names)


def place_names(df: pd.DataFrame, profiles: list[ColumnProfile]) -> list[str]:
    """Values of the dataset's own area columns (city, municipality...): a gazetteer for the free-text recognizer."""
    places = set()
    for p in profiles:
        if p.subtype == "area" and p.name in df.columns:
            vals = df[p.name].dropna().astype(str).str.strip()
            if vals.nunique() <= 5000:
                places.update(v for v in vals.unique() if not v.isdigit())
    return sorted(places)


def run(df: pd.DataFrame, scenario: Scenario, profiles: list[ColumnProfile] | None = None, gold: dict | None = None,
        gold_column: str | None = None, secret: bytes | None = None, use_presidio: bool = True,
        progress: Callable[[str], None] | None = None,
        extra_names: list[str] | None = None, extra_places: list[str] | None = None,
        context: pd.DataFrame | None = None) -> RunResult:
    """Protect one table. `context`: attributes of linked parent tables, already protected, joined onto this table's
    rows (columns named 'parent.column'). They count as known attributes in the re-identification test, so k and the
    re-identification rate hold on the joined view, but they are not part of this table's output."""
    t0 = time.time()
    say = progress or (lambda m: None)
    df = df.reset_index(drop=True)
    if secret is None:
        secret, secret_source = get_secret()
    else:
        secret_source = "provided by caller"
    if profiles is None:
        profiles = profile_dataframe(df, scenario.seed)
    profiles = [p for p in profiles if p.name in df.columns]
    domain, _ = detect_domain(list(df.columns))

    say("assigning gates")
    assign_gates(profiles, df, scenario)
    for p in profiles:   # a task needs this identifier linkable: keyed pseudonym instead of dropping it
        if p.name in (scenario.pseudonymise or []) and p.gate == Gate.RED:
            p.red_action = RedAction.PSEUDONYMISE
            p.gate_reason = "keyed HMAC pseudonym: a utility task of the scenario needs records of the same person/card linked"
    by_name = {p.name: p for p in profiles}

    known_declared = attacker_known(profiles, scenario)
    known = list(known_declared)
    derived = derive_knowledge(df, profiles, known) if scenario.derive_knowledge else []
    known += [d["column"] for d in derived if d["column"] not in known]
    # attributes joined from linked parent tables are part of the threat model from here on
    ctx_cols = [] if context is None else [c for c in context.columns if c not in df.columns]
    df_risk = df if not ctx_cols else pd.concat([df, context[ctx_cols].reindex(df.index)], axis=1)
    promoted = promote_green(df_risk, profiles, known + ctx_cols, scenario)
    known += [d["column"] for d in promoted if d["column"] not in known]
    known = known + ctx_cols
    # anything the attacker knows is a quasi-identifier by definition, so it joins the search space
    for p in profiles:
        if p.name in known and p.gate == Gate.GREEN and p.ladder is not None and p.role != Role.TARGET:
            p.gate = Gate.YELLOW
            p.gate_reason = "a known attribute in the threat model, so it is treated as a quasi-identifier"
    groups = column_groups(profiles)

    say("fitting ladders")
    ladders = fit_ladders(df, profiles, scenario)

    # ---------------------------------------------------------------- free text
    say("redacting free text")
    text_out: dict[str, pd.Series] = {}
    text_report: dict[str, dict] = {}
    text_examples: list[dict] = []
    text_cols = [p.name for p in profiles if p.role == Role.FREE_TEXT and p.gate != Gate.RED]
    if text_cols:
        # health mentions are the content of health data, not identifiers in it: keep them there
        skip = {"HEALTH"} if domain == "healthcare" else set()
        redactor = Redactor(person_names(df, profiles) + list(extra_names or []), use_presidio=use_presidio,
                            places=place_names(df, profiles) + list(extra_places or []), skip_labels=skip)
        for col in text_cols:
            t1 = time.time()
            red, spans = redactor.redact(df[col])
            text_out[col] = red
            counts: dict[str, int] = {}
            for sp in spans.values():
                for s in sp:
                    counts[s.label] = counts.get(s.label, 0) + 1
            entry = {"engine": redactor.engine, "entities_replaced": counts, "records": int(df[col].notna().sum()),
                     "seconds": round(time.time() - t1, 1), "evaluation": None}
            if gold is not None and (gold_column in (None, col)):
                entry["evaluation"] = evaluate_text(gold, spans, df[col], red)
            text_report[col] = entry
            for idx in list(spans)[:6]:
                text_examples.append({"column": col, "row": idx, "original": str(df.at[idx, col]), "redacted": str(red.at[idx]),
                                      "entities": [s.to_dict() for s in spans[idx]]})

    # ---------------------------------------------------------------- search
    say("searching for the minimum sufficient transformation")
    proxy = ProxyUtility(df, profiles, ladders, scenario.key_columns)
    caps, cap_notes = {}, []
    for col, lvl in (scenario.max_levels or {}).items():
        if col in ladders:
            try:
                caps[col] = ladders[col].level_index(lvl)
                cap_notes.append({"column": col, "max_level": ladders[col].level_names[caps[col]]})
            except ValueError as exc:
                cap_notes.append({"column": col, "error": str(exc)})
    searcher = GreedySearch(df_risk, profiles, ladders, scenario, known, groups, proxy, caps)
    before = searcher.evaluate({c: 0 for c in ladders}, False)
    res = searcher.run()
    yellow_levels = {c: lvl for c, lvl in res.levels.items() if by_name[c].gate == Gate.YELLOW}
    plan = Plan(levels=yellow_levels, suppress=res.suppress, suppressed_rids=res.suppressed_rids)
    protected = apply_plan(df, profiles, ladders, plan, secret, text_out)

    # ---------------------------------------------------------------- metrics
    say("measuring utility")
    # attribute disclosure concerns sensitive values the attacker does not already know
    sensitive = [p.name for p in profiles if p.role == Role.SENSITIVE and p.name in protected.columns and p.name not in known]
    orig_view = pd.DataFrame({c: df_risk[c] for c in known}, index=df.index)
    prot_known = pd.DataFrame({c: (ladders[c].apply(df[c], res.levels.get(c, 0)) if c in ladders else df_risk[c])
                               for c in known}, index=df.index).loc[protected.index]
    k_before = k_metrics(orig_view, known, int(scenario.min_k))
    k_after = k_metrics(prot_known, known, int(scenario.min_k))
    l_before = l_diversity(pd.concat([orig_view, df[sensitive]], axis=1), known, sensitive)
    l_after = l_diversity(pd.concat([prot_known, protected[sensitive]], axis=1), known, sensitive)

    task = None
    curve_task = {}
    target = scenario.target if scenario.target in df.columns else None
    if target and scenario.task_type in ("classification", "regression"):
        task = task_utility(df, profiles, ladders, plan, secret, target, scenario.task_type, scenario.seed)
        idxs = sorted(set(np.linspace(0, len(res.path) - 1, min(6, len(res.path))).round().astype(int)))
        for i in idxs:
            pt = res.path[i]
            m = searcher.evaluate(pt["levels"], pt["suppress"])
            pl = Plan(levels={c: v for c, v in pt["levels"].items() if by_name[c].gate == Gate.YELLOW},
                      suppress=pt["suppress"], suppressed_rids=m["suppressed"] if m else [])
            tu = task if i == len(res.path) - 1 else task_utility(df, profiles, ladders, pl, secret, target,
                                                                    scenario.task_type, scenario.seed, baseline=task)
            curve_task[i] = tu["score"]
    for i, pt in enumerate(res.path):
        pt["task_retained"] = curve_task.get(i)
    descriptive = descriptive_utility(df, protected, profiles, ladders, yellow_levels, scenario.key_columns)
    # every further prediction purpose (e.g. categories AND numbers) gets its own model evaluation
    tasks = [task] if task else []
    for purpose, tgt in scenario.prediction_tasks:
        if tgt in df.columns and not any(t["task"] == purpose and t["target"] == tgt for t in tasks):
            tasks.append(task_utility(df, profiles, ladders, plan, secret, tgt, purpose, scenario.seed))
    # utility per selected purpose; the overall score is their mean
    by_purpose = {}
    for purpose in scenario.purposes:
        if purpose in ("classification", "regression"):
            mine = [t for t in tasks if t["task"] == purpose]
            if mine:
                by_purpose[purpose] = {
                    "score": float(np.mean([t["score"] for t in mine])),
                    "measured_by": (f"mean over {len(mine)} models ({', '.join(t['target'] for t in mine)}); " if len(mine) > 1
                                    else f"model predicting {mine[0]['target']}; ") + mine[0]["score_formula"]}
        elif purpose == "descriptive":
            by_purpose[purpose] = {"score": descriptive["query_score"],
                                   "measured_by": "aggregate queries on the original columns (1 − mean relative error / 0.5)"}
        else:
            by_purpose[purpose] = {"score": descriptive["distribution_score"],
                                   "measured_by": "distributions vs. original values (TVD, Wasserstein / sd)"}

    # ---------------------------------------------------------------- decisions
    decisions = []
    contrib_by_col = {}
    for key, c in res.contributions.items():
        for col in c["columns"]:
            contrib_by_col[col] = c
    unit_cols = {c for u in res.units for c in u}
    for p in profiles:
        row = {"column": p.name, "role": p.role.value, "kind": p.kind.value, "gate": p.gate.value if p.gate else None,
               "attacker_knows": p.name in known}
        if p.gate == Gate.RED:
            row.update(pet=RED_NAMES[p.red_action or RedAction.DROP], level="—", reason=p.gate_reason)
        elif p.role == Role.FREE_TEXT:
            tr = text_report.get(p.name, {})
            ev = tr.get("evaluation")
            rec = f"; measured recall {ev['recall_any_label']:.1%} on {ev['gold_spans']} gold spans" if ev and ev.get("recall_any_label") is not None else ""
            n = sum(tr.get("entities_replaced", {}).values())
            row.update(pet="Free-text PII redaction", level="typed placeholders",
                       reason=f"{n} entities replaced by [PERSON]/[PHONE]/[EMAIL]/[LOCATION]/[DATE]/[ID]{rec}")
        elif p.gate == Gate.YELLOW and p.name in ladders:
            lvl = res.levels.get(p.name, 0)
            lad = ladders[p.name]
            row["level"] = lad.describe(lvl)
            row["pet"] = PET_NAMES.get(p.ladder, "Generalisation") if lvl > 0 else "None needed"
            if lad.drops(lvl):
                row["pet"] = "Suppression (column dropped)"
            if p.name in unit_cols:
                c = contrib_by_col.get(p.name)
                if c and c["steps"]:
                    grp = f" (moved together with {', '.join(x for x in c['columns'] if x != p.name)})" if len(c["columns"]) > 1 else ""
                    row["reason"] = (f"reduced the re-identification rate by {c['risk_pp']:.1f} pp at {c['utility_cost']:.3f} utility "
                                     f"cost over {c['steps']} step(s){grp}")
                    if all(m in ("uniqueness", "prosecutor_risk", "entropy", "below_k") for m in c["metrics"]):
                        row["reason"] += f"; chosen to reduce {', '.join(sorted(set(c['metrics'])))} first"
                else:
                    row["reason"] = "searched, but other moves reached the target more cheaply: left exact"
            elif p.min_level > 0:
                row["reason"] = p.gate_reason
            else:
                row["reason"] = "not a known attribute in the threat model: left at its floor level (see residual risks)"
            row["level_index"] = lvl
            if p.name in caps:
                row["reason"] += f"; utility requirement: at most '{lad.level_names[caps[p.name]]}'"
            if lad.output_name(lvl) != p.name:
                row["level"] += f" (released as column '{lad.output_name(lvl)}')"
        else:
            row.update(pet="None (unchanged)", level="—", reason=p.gate_reason)
        decisions.append(row)
    if res.suppress:
        n = len(res.suppressed_rids)
        sc = res.contributions.get("suppression", {})
        decisions.append({"column": "(records)", "role": "—", "kind": "—", "gate": "yellow", "attacker_knows": True,
                          "pet": "Record suppression", "level": f"{n} rows ({n / len(df):.1%})",
                          "reason": (f"records in equivalence classes with k < {scenario.min_k} removed; reduced the "
                                     f"re-identification rate by {sc.get('risk_pp', 0):.1f} pp at {sc.get('utility_cost', 0):.3f} utility cost")})

    after = res.final
    metrics = {
        "before": _risk_summary(before, k_before),
        "after": _risk_summary(after, k_after),
        "rows_original": int(len(df)), "rows_released": int(len(protected)),
        "rows_suppressed_frac": float(1 - len(protected) / len(df)),
        "columns_original": int(df.shape[1]), "columns_released": int(protected.shape[1]),
        "proxy_before": before["proxy"], "proxy_after": after["proxy"],
        "l_diversity_before": l_before, "l_diversity_after": l_after,
    }
    utility_score = float(np.mean([v["score"] for v in by_purpose.values()])) if by_purpose else descriptive["score"]

    report = {
        "tool": {"name": "TrustED PETs Transformer", "engine_version": __version__},
        "scenario": {**scenario.to_dict(), "recipient_label": scenario.recipient_label},
        "domain_hint": domain,
        "threat_model": {
            "attacker_knows_declared": known_declared,
            "derived_knowledge": derived,
            "promoted_green_columns": promoted,
            "attacker_knows": known,
            "linked_context_columns": ctx_cols,
            "attacker_sample_fraction": scenario.attacker_fraction,
            "attacker_targets": int(after["attack"]["targets"]),
            "statement": _threat_statement(scenario, known, derived),
        },
        "profiles": [{k: v for k, v in p.to_dict().items() if k != "examples"} for p in profiles],
        "groups": groups,
        "decisions": decisions,
        "plan": plan.to_dict(),
        "utility_requirements": cap_notes,
        "kit": {"scenario": scenario.kit_scenario, "tasks_kept": scenario.kit_tasks,
                "tasks_sacrificed": scenario.sacrificed_tasks} if scenario.kit_scenario else None,
        "target_met": res.target_met,
        "metrics": metrics,
        "utility": {"score": utility_score, "task": task, "tasks": tasks, "by_purpose": by_purpose,
                    "descriptive": descriptive},
        "search_path": res.path,
        "text": text_report,
        "text_examples_redacted": [{"column": e["column"], "redacted": e["redacted"]} for e in text_examples],
        "config": {"seed": scenario.seed, "pseudonymisation": "HMAC-SHA256, 16 hex chars", "secret_source": secret_source,
                   "suppression_cap": scenario.suppression_cap, "presidio": use_presidio},
        "runtime_seconds": None,
    }
    if scenario.kit_scenario:
        from .kit import match_table
        m = match_table(list(df.columns))
        if m is not None:
            say("running the kit's re-identification tests (attack_specs.json, tolerant matching)")
            report["kit_attacks"] = kit_attacks(df, protected.index, profiles, ladders, res.levels, searcher.att_ids, *m)
    report["residual_risks"] = residual_risks(report, profiles, known, res)
    report["runtime_seconds"] = round(time.time() - t0, 1)
    preview = build_preview(df, protected, profiles)
    renames = {c: ladders[c].output_name(lvl) for c, lvl in yellow_levels.items()
               if c in ladders and ladders[c].output_name(lvl) != c}
    key = record_key(df, profiles)
    rowmap = pd.DataFrame({"protected_row": range(len(protected)),
                           "original_record_id": df.loc[protected.index, key].to_numpy() if key else protected.index})
    report["export"] = {"renamed_columns": renames, "record_key": key or "(row number)",
                        "rowmap": "rowmap_<table>.csv: protected_row -> original_record_id, for scoring only, never publish"}
    return RunResult(protected=protected, report=report, preview=preview, text_examples=text_examples,
                     rowmap=rowmap, renames=renames)


def record_key(df: pd.DataFrame, profiles: list[ColumnProfile]) -> str | None:
    """The table's own row identifier (unique key column), used by row maps."""
    for p in profiles:
        if p.subtype == "record_key" and p.name in df.columns and df[p.name].notna().all() and df[p.name].is_unique:
            return p.name
    return None


def _risk_summary(m: dict, km: dict) -> dict:
    return {"linkage": m["linkage"], "attack": m["attack"], "uniqueness": km["uniqueness"], "min_k": km["min_k"],
            "prosecutor_risk": km["prosecutor_risk"], "below_k": km["below_k"], "k_distribution": km["k_distribution"],
            "classes": km["classes"]}


def _threat_statement(sc: Scenario, known: list[str], derived: list[dict]) -> str:
    s = (f"Recipient: {sc.recipient_label}. Threat model: someone may already know the exact values of "
         f"{', '.join(known) or 'no columns'} for a random {sc.attacker_fraction:.0%} of the people in the dataset "
         f"(e.g. from a public directory or social media), and may know the release transformations, so the same "
         f"generalisation is applied to that knowledge before matching. "
         f"A person is re-identified when exactly one released record matches and it is theirs. "
         f"Target: re-identification rate ≤ {sc.max_linkage:.0%} and every equivalence class ≥ {sc.min_k} records.")
    if derived:
        s += " Derived knowledge: " + "; ".join(f"{d['column']} follows from {d['from']}" for d in derived) + "."
    return s


def residual_risks(report: dict, profiles: list[ColumnProfile], known: list[str], res) -> list[dict]:
    sc = report["scenario"]
    out = []
    if not res.target_met:
        a = report["metrics"]["after"]
        out.append({"severity": "high", "text": (f"Target NOT met: re-identification rate {a['linkage']:.1%} (target ≤ {sc['max_linkage']:.0%}), "
                                                 f"min k {a['min_k']} (target ≥ {sc['min_k']}). No remaining move reached it; "
                                                 "consider aggregation or DP instead of row-level release.")})
    out.append({"severity": "medium", "text": "Risk is measured only against the modelled threat model; more background "
                                              "knowledge (more columns, more people) can raise it."})
    out.append({"severity": "medium", "text": "k-anonymity and re-identification rates are empirical, not a formal guarantee (unlike "
                                              "differential privacy)."})
    out.append({"severity": "low", "text": "Greedy search is not globally optimal; another combination of levels may keep more "
                                           "utility at the same risk."})
    for col, t in report["text"].items():
        ev = t.get("evaluation")
        if ev and ev.get("recall_any_label") is not None:
            worst = [f"{lab} {v['recall']:.0%}" for lab, v in ev["per_label"].items() if v["recall"] is not None]
            out.append({"severity": "medium", "text": f"Free-text redaction in '{col}': measured recall "
                        f"{ev['recall_any_label']:.1%} overall ({', '.join(worst)}); {ev['leak_rate']:.2%} of gold "
                        "identifier values survive verbatim. Recall on text unlike the ground truth (other templates, "
                        "languages, spelled-out numbers) can be lower."})
        else:
            out.append({"severity": "medium", "text": f"Free-text redaction recall in '{col}' is not measured (no ground "
                                                      "truth); it is below 100 % in general."})
    if any(p.red_action == RedAction.PSEUDONYMISE for p in profiles if p.gate == Gate.RED):
        out.append({"severity": "low", "text": "Pseudonyms are linkable across releases that use the same secret; rotate "
                                               "PETS_SECRET per recipient to prevent cross-release linkage."})
    for col, l in report["metrics"]["l_diversity_after"].items():
        if l["classes_l1"]:
            out.append({"severity": "medium", "text": (f"Attribute disclosure: {l['classes_l1']} of {l['classes']} "
                                                       f"equivalence classes ({l['records_in_l1_classes']} records) have a single "
                                                       f"value of sensitive column '{col}' (l = 1).")})
    unknown_qi = [p.name for p in profiles if p.gate == Gate.YELLOW and p.role in (Role.QUASI_ID, Role.DATE, Role.LOCATION)
                  and p.name not in known and report["plan"]["levels"].get(p.name, 0) == 0]
    if unknown_qi:
        out.append({"severity": "medium", "text": ("Quasi-identifiers released exactly because the threat model does not list them "
                                                   f"as known attributes: {', '.join(unknown_qi)}.")})
    green_hi = [p.name for p in profiles if p.gate == Gate.GREEN and p.role in (Role.OTHER,) and p.unique_ratio > 0.5]
    if green_hi:
        out.append({"severity": "medium", "text": f"High-cardinality columns released unchanged; review manually: {', '.join(green_hi)}."})
    for a in report.get("kit_attacks") or []:
        if a["after"] > sc["max_linkage"]:
            out.append({"severity": "high", "text": (f"Kit re-identification test '{a['adversary']}' (attack_specs.json, tolerant matching on {', '.join(a['qis'])}) "
                                                     f"still re-identifies {a['after']:.1%} of targets.")})
    if "local key file" in report["config"]["secret_source"]:
        out.append({"severity": "low", "text": "Pseudonymisation secret comes from a local key file; set PETS_SECRET from a "
                                               "secrets manager in production."})
    return out


def build_preview(df: pd.DataFrame, protected: pd.DataFrame, profiles: list[ColumnProfile], n: int = 20) -> dict:
    """First n original rows next to their protected counterparts (UI only, never exported)."""
    rows = []
    cols = list(df.columns)
    pcols = list(protected.columns)
    for idx in df.index[:n]:
        orig = {c: _cell(df.at[idx, c]) for c in cols}
        if idx in protected.index:
            prot = {c: _cell(protected.at[idx, c]) for c in pcols}
            changed = [c for c in pcols if prot.get(c) != orig.get(c)]
            rows.append({"row": int(idx), "original": orig, "protected": prot, "changed": changed, "suppressed": False})
        else:
            rows.append({"row": int(idx), "original": orig, "protected": {}, "changed": [], "suppressed": True})
    return {"columns": cols, "protected_columns": pcols, "rows": rows}


def _cell(v) -> str:
    return "" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v)


def kit_attacks(df: pd.DataFrame, released_index: pd.Index, profiles: list[ColumnProfile], ladders: dict[str, Ladder],
                levels: dict[str, int], att_ids, dataset: str, table: str) -> list[dict]:
    """The kit's background-knowledge attacks for this table (metadata/attack_specs.json), run with tolerant matching
    (time windows, age ± years) against the release. Attackers get the exact original values of the targets, i.e. at
    least the knowledge the kit's attacker files contain. QIs whose column is not released are skipped, as in the kit."""
    from .kit import attacks_for
    from .risk import tolerant_linkage
    by = {p.name: p for p in profiles}
    out = []
    for spec in attacks_for(dataset, table):
        exact, windows, used = [], {}, []
        for q in spec["qis"]:
            col = next((c for c in q["protected"] if c in df.columns and by.get(c) and by[c].gate != Gate.RED), None)
            if col is None or q["kind"] == "geo":
                continue
            lad = ladders.get(col)
            if q["kind"] in ("time", "age") and lad is not None and lad.tolerance_value(df[col]) is not None:
                windows[col] = (float(q["window"][0]), float(q["window"][1])) if q["kind"] == "time" else \
                    (float(q.get("tol", 0)), float(q.get("tol", 0)))
            else:
                exact.append(col)
            used.append(f"{col} ({q['kind']})")
        if not used:
            continue
        res = {}
        for name, lv, idx in (("before", {c: 0 for c in levels}, df.index), ("after", levels, released_index)):
            def gen(c):
                lad = ladders.get(c)
                return lad.apply(df[c], lv.get(c, 0)) if lad is not None else df[c]
            ex = pd.DataFrame({c: gen(c) for c in exact}, index=df.index)
            lo = pd.DataFrame({c: ladders[c].interval(df[c], lv.get(c, 0))[0] for c in windows}, index=df.index)
            hi = pd.DataFrame({c: ladders[c].interval(df[c], lv.get(c, 0))[1] for c in windows}, index=df.index)
            val = pd.DataFrame({c: ladders[c].tolerance_value(df[c]) for c in windows}, index=df.index)
            r = tolerant_linkage(ex.loc[att_ids], val.loc[att_ids], ex.loc[idx], lo.loc[idx], hi.loc[idx], windows)
            res[name] = r["success_rate"]
        out.append({"adversary": spec["adversary"], "description": spec.get("description", ""), "qis": used,
                    "tolerances": {c: list(w) for c, w in windows.items()}, "before": res["before"], "after": res["after"],
                    "targets": int(len(att_ids))})
    return out
