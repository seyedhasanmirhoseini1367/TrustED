"""Command line interface.

    python -m engine.cli profile data.csv
    python -m engine.cli run data.csv --scenario scenario.json --out out/ [--spans spans.json] [--profile profile.json]
    python -m engine.cli run a.csv b.csv --scenario scenario.json --out out/       # multi-table, consistent pseudonyms
    python -m engine.cli run ed_visits.csv lab_results.csv --kit-scenario HEALTH-S1 --out out/health
    python -m engine.cli kit-scenarios ed_visits.csv
    python -m engine.cli labels data/health/*.csv --dataset health --out labels.csv
    python -m engine.cli dp data.csv --group department --stat count --eps 1
    python -m engine.cli sample --out data/sample
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import kit
from .dp import dp_query, error_curve
from .pipeline import load_table, run
from .profiling import detect_domain, profile_dataframe
from .report import render_html, to_json
from .roles import ColumnProfile, Scenario
from .text_pii import load_gold
from .transforms import get_secret


def _print(s: str) -> None:
    sys.stdout.buffer.write((s + "\n").encode("utf-8", "replace"))


def cmd_profile(a) -> int:
    df = load_table(a.data)
    profiles = profile_dataframe(df)
    domain, _ = detect_domain(list(df.columns))
    _print(f"{len(df)} rows × {df.shape[1]} columns · looks like: {domain or 'unknown'}")
    for p in profiles:
        _print(f"  {p.name:28} {p.role.value:13} {p.kind.value:11} ladder={str(p.ladder):11} conf={p.confidence:.2f}  {p.reason}")
    if a.out:
        Path(a.out).write_text(json.dumps([p.to_dict() for p in profiles], indent=2, ensure_ascii=False), encoding="utf-8")
        _print(f"wrote {a.out}")
    return 0


def cmd_run(a) -> int:
    sc = Scenario.from_dict(json.loads(Path(a.scenario).read_text(encoding="utf-8"))) if a.scenario else Scenario()
    secret, _ = get_secret()  # one secret for every table -> same value, same token across tables
    out_root = Path(a.out)
    gold, gold_col = None, None
    if a.spans:
        gold_col, gold = load_gold(json.loads(Path(a.spans).read_text(encoding="utf-8")))
    profiles = None
    if a.profile:
        profiles = [ColumnProfile.from_dict(d) for d in json.loads(Path(a.profile).read_text(encoding="utf-8"))]
    out_root.mkdir(parents=True, exist_ok=True)
    tasks = a.kit_tasks.split(",") if a.kit_tasks else None
    for path in a.data:
        df = load_table(path)
        name = Path(path).stem
        if a.kit_scenario:
            table_sc = kit.scenario_from_kit(a.kit_scenario, list(df.columns), tasks, seed=sc.seed)
        else:
            table_sc = Scenario.from_dict({**sc.to_dict(), "attacker_qis": [c for c in sc.attacker_qis if c in df.columns],
                                           "target": sc.target if sc.target in df.columns else None,
                                           "key_columns": [c for c in sc.key_columns if c in df.columns]})
        if table_sc.target is None and table_sc.task_type in ("classification", "regression"):
            table_sc.task_type = "descriptive"
        res = run(df, table_sc, profiles=profiles if len(a.data) == 1 else None, gold=gold if len(a.data) == 1 else None,
                  gold_column=gold_col, secret=secret, use_presidio=not a.no_presidio, progress=lambda m: _print(f"  .. {m}"))
        # kit conventions: same table file name, original column names, rowmap_<table>.csv (never publish the row map)
        res.export_table().to_csv(out_root / f"{name}.csv", index=False)
        res.rowmap.to_csv(out_root / f"rowmap_{name}.csv", index=False)
        rep_dir = out_root / "reports"
        rep_dir.mkdir(exist_ok=True)
        (rep_dir / f"{name}.report.json").write_text(to_json(res.report), encoding="utf-8")
        (rep_dir / f"{name}.report.html").write_text(render_html(res.report), encoding="utf-8")
        r = res.report
        b, af = r["metrics"]["before"], r["metrics"]["after"]
        _print(f"{name}: attack success {b['linkage']:.1%} → {af['linkage']:.1%}; min k {b['min_k']} → {af['min_k']}; "
               f"utility {r['utility']['score']:.3f}; target {'met' if r['target_met'] else 'NOT met'}; {r['runtime_seconds']} s")
        _print(f"  wrote {out_root / (name + '.csv')} + rowmap_{name}.csv + reports/{name}.report.*")
    if a.kit_scenario:
        _print(f"score with the kit: pets-eval scorecard --original <kit>/data/<dataset> --protected {out_root} "
               f"--scenario {a.kit_scenario}")
    return 0


def cmd_kit_scenarios(a) -> int:
    df = load_table(a.data)
    found = kit.scenarios_for(list(df.columns))
    if not found:
        _print("This table does not match a starter-kit table.")
        return 0
    for s in found:
        _print(f"{s['id']}  ({s['trust_level']})  {s['title']}")
        for t in s["tasks"]:
            flag = "keep" if t["default"] else f"sacrificed by default: {t['conflict']}"
            _print(f"    {t['id']:6} {flag:45} {t['desc']}")
    return 0


def cmd_labels(a) -> int:
    rows = []
    for path in a.data:
        df = load_table(path)
        m = kit.match_table(list(df.columns))
        dataset = a.dataset or (m[0] if m else Path(path).parent.name)
        rows.append(kit.field_labels(profile_dataframe(df), dataset, Path(path).stem))
    import pandas as pd
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(a.out, index=False)
    _print(f"wrote {len(out)} column labels to {a.out} (score with: pets-eval detect --pred {a.out})")
    return 0


def cmd_dp(a) -> int:
    df = load_table(a.data)
    res = dp_query(df, a.group, a.stat, a.eps, a.value, seed=a.seed)
    _print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1, default=str))
    for r in res["rows"]:
        _print(f"  {r['group']:30} true={r['true']}  noisy={r['noisy']:.2f}")
    if a.curve:
        for p in error_curve(df, a.group, a.stat, a.value, seed=a.seed):
            _print(f"  ε={p['epsilon']:<5} mean relative error {p['mean_rel_error']:.3f}")
    return 0


def cmd_sample(a) -> int:
    from data.generate_sample import write
    c, s = write(a.out, a.rows, a.seed)
    _print(f"wrote {c} and {s}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pets", description="TrustED PETs Transformer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("profile", help="detect column roles")
    p.add_argument("data")
    p.add_argument("--out", help="write profile JSON (editable, then pass to run --profile)")
    p.set_defaults(fn=cmd_profile)
    p = sub.add_parser("run", help="protect one or more tables")
    p.add_argument("data", nargs="+")
    p.add_argument("--scenario", help="scenario JSON (see examples/scenario_partner.json)")
    p.add_argument("--out", default="out")
    p.add_argument("--spans", help="ground-truth free-text spans JSON")
    p.add_argument("--profile", help="profile JSON with role/gate overrides")
    p.add_argument("--no-presidio", action="store_true", help="regex-only text redaction")
    p.add_argument("--kit-scenario", help="official starter-kit scenario id, e.g. HEALTH-S1")
    p.add_argument("--kit-tasks", help="comma-separated kit utility tasks to keep (default: all non-conflicting)")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("kit-scenarios", help="list official kit scenarios that apply to a table")
    p.add_argument("data")
    p.set_defaults(fn=cmd_kit_scenarios)
    p = sub.add_parser("labels", help="export detected column roles in the kit's label format")
    p.add_argument("data", nargs="+")
    p.add_argument("--dataset", help="dataset name (default: recognised kit dataset or parent folder)")
    p.add_argument("--out", default="field_labels.csv")
    p.set_defaults(fn=cmd_labels)
    p = sub.add_parser("dp", help="differentially private aggregate query")
    p.add_argument("data")
    p.add_argument("--group", required=True)
    p.add_argument("--stat", choices=["count", "mean"], default="count")
    p.add_argument("--value")
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--curve", action="store_true")
    p.set_defaults(fn=cmd_dp)
    p = sub.add_parser("sample", help="generate the synthetic sample dataset")
    p.add_argument("--out", default="data/sample")
    p.add_argument("--rows", type=int, default=5000)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_sample)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
