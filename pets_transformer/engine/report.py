"""JSON and printable HTML report (spec §7). Contains metrics, decisions and redacted text only - no original rows."""
from __future__ import annotations

import html
import json
import math

import numpy as np


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def to_json(report: dict) -> str:
    return json.dumps(_clean(report), indent=2, ensure_ascii=False)


def pct(x, digits=1) -> str:
    return "—" if x is None else f"{x * 100:.{digits}f} %"


def num(x, digits=3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def _e(x) -> str:
    return html.escape("" if x is None else str(x))


def kpis(report: dict) -> list[dict]:
    """Headline numbers shared by the HTML report and the web dashboard."""
    m = report["metrics"]
    b, a = m["before"], m["after"]
    out = [
        {"label": "Re-identification rate", "before": pct(b["linkage"]), "after": pct(a["linkage"]),
         "good": a["linkage"] <= report["scenario"]["max_linkage"],
         "note": f"target ≤ {pct(report['scenario']['max_linkage'], 0)}"},
        {"label": "Unique records", "before": pct(b["uniqueness"]), "after": pct(a["uniqueness"]), "good": a["uniqueness"] < b["uniqueness"] or a["uniqueness"] == 0,
         "note": "on the threat model's known attributes"},
        {"label": "Minimum k", "before": str(b["min_k"]), "after": str(a["min_k"]), "good": a["min_k"] >= report["scenario"]["min_k"],
         "note": f"target ≥ {report['scenario']['min_k']}"},
    ]
    t = report["utility"].get("task")
    if t:
        if t["task"] == "classification":
            out.append({"label": "ROC-AUC (original → protected)", "before": num(t["original"]["roc_auc"]),
                        "after": num(t["protected"]["roc_auc"]), "good": t["retained"]["roc_auc"] >= 0.9,
                        "note": f"{pct(t['retained']['roc_auc'])} retained · F1 {num(t['original']['f1'], 2)} → {num(t['protected']['f1'], 2)}"})
        else:
            out.append({"label": "R² (original → protected)", "before": num(t["original"]["r2"]),
                        "after": num(t["protected"]["r2"]), "good": t["retained"]["r2"] >= 0.9,
                        "note": f"MAE {num(t['original']['mae'], 2)} → {num(t['protected']['mae'], 2)}"})
    d = report["utility"]["descriptive"]
    out.append({"label": "Utility score", "before": "1.00", "after": num(report["utility"]["score"], 2),
                "good": report["utility"]["score"] >= 0.8,
                "note": "task metric retained" if t else f"aggregate error {pct(d['query_mean_rel_error'])}"})
    out.append({"label": "Rows suppressed", "before": "0 %", "after": pct(m["rows_suppressed_frac"]),
                "good": m["rows_suppressed_frac"] <= report["scenario"]["suppression_cap"],
                "note": f"{m['rows_released']} of {m['rows_original']} rows released"})
    return out


def render_html(report: dict, plain_text: str | None = None) -> str:
    sc = report["scenario"]
    tm = report["threat_model"]
    rows_k = "".join(
        f"<tr><td>{_e(k['label'])}</td><td>{_e(k['before'])}</td><td><b>{_e(k['after'])}</b></td>"
        f"<td class='{'ok' if k['good'] else 'bad'}'>{'✓' if k['good'] else '✗'}</td><td>{_e(k['note'])}</td></tr>"
        for k in kpis(report))
    rows_d = "".join(
        f"<tr><td><code>{_e(d['column'])}</code></td><td>{_e(d['role'])}</td>"
        f"<td><span class='gate {_e(d['gate'])}'>{_e(d['gate'])}</span></td><td>{_e(d['pet'])}</td>"
        f"<td>{_e(d['level'])}</td><td>{_e(d.get('reason'))}</td></tr>" for d in report["decisions"])
    rows_p = "".join(
        f"<tr><td>{p['step']}</td><td>{_e(p['move'])}</td><td>{pct(p['linkage'])}</td><td>{pct(p['uniqueness'])}</td>"
        f"<td>{p['min_k']}</td><td>{pct(p['suppressed_frac'])}</td><td>{num(p['proxy'])}</td>"
        f"<td>{'' if p.get('task_retained') is None else pct(p['task_retained'])}</td></tr>" for p in report["search_path"])
    risks = "".join(f"<li class='sev-{_e(r['severity'])}'><b>{_e(r['severity'])}</b> — {_e(r['text'])}</li>"
                    for r in report["residual_risks"])
    text_html = ""
    for col, t in report["text"].items():
        ev = t.get("evaluation")
        ev_rows = ""
        if ev:
            ev_rows = "".join(
                f"<tr><td>{_e(lab)}</td><td>{v['gold']}</td><td>{v['predicted']}</td><td>{pct(v['precision'])}</td>"
                f"<td>{pct(v['recall'])}</td><td>{num(v['f2'])}</td></tr>" for lab, v in ev["per_label"].items())
            ev_rows = (f"<table><tr><th>Entity</th><th>Gold</th><th>Predicted</th><th>Precision</th><th>Recall</th><th>F2</th></tr>"
                       f"{ev_rows}</table><p>Overall recall (any label) {pct(ev['recall_any_label'])}; gold values surviving "
                       f"verbatim {pct(ev['leak_rate'], 2)}.</p>")
        ex = "".join(f"<li><code>{_e(e['redacted'])}</code></li>" for e in report["text_examples_redacted"] if e["column"] == col)
        text_html += (f"<h3><code>{_e(col)}</code></h3><p>Engine: {_e(t['engine'])}. Entities replaced: "
                      f"{_e(', '.join(f'{k} {v}' for k, v in t['entities_replaced'].items()))}.</p>{ev_rows}<ul>{ex}</ul>")
    task = report["utility"].get("task")
    task_html = ""
    if task:
        task_html = (f"<p>{_e(task['model'])}; target <code>{_e(task['target'])}</code>; the ORIGINAL data was split "
                     f"80/20 (seed {sc['seed']}) before any transformation; the same plan was then applied to both parts. "
                     f"Train rows {task['train_rows']} ({task['protected_train_rows']} after suppression), test rows {task['test_rows']}.</p>"
                     f"<pre>{_e(json.dumps(_clean({'original': task['original'], 'protected': task['protected'], 'retained': task['retained']}), indent=1))}</pre>")
    d = report["utility"]["descriptive"]
    plain = f"<section><h2>Summary in plain language</h2><p>{_e(plain_text)}</p><p class='muted'>Generated by an LLM from the JSON report only.</p></section>" if plain_text else ""
    status = ("<span class='ok'>target met</span>" if report["target_met"] else "<span class='bad'>target NOT met</span>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>PETs Transformer report</title>
<style>
body{{font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;color:#1d2433;max-width:1100px;margin:24px auto;padding:0 20px}}
h1{{font-size:24px;margin-bottom:4px}} h2{{font-size:18px;border-bottom:1px solid #dde2ea;padding-bottom:4px;margin-top:28px}}
table{{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px}} td,th{{border:1px solid #dde2ea;padding:5px 7px;text-align:left;vertical-align:top}}
th{{background:#f3f5f9}} code{{font-size:12px}} .ok{{color:#13795b;font-weight:600}} .bad{{color:#b42318;font-weight:600}}
.gate{{padding:1px 7px;border-radius:9px;font-size:12px;font-weight:600}} .gate.red{{background:#fde2e1;color:#912018}}
.gate.yellow{{background:#fdf1c7;color:#854a0e}} .gate.green{{background:#d9f3e6;color:#0b6b46}} .muted{{color:#667085}}
li.sev-high{{color:#912018}} pre{{background:#f6f8fa;padding:8px;font-size:12px;overflow-x:auto}}
@media print{{body{{margin:0}} h2{{break-after:avoid}} table{{break-inside:auto}}}}
</style></head><body>
<h1>TrustED PETs Transformer — protection report</h1>
<p class="muted">Engine {_e(report['tool']['engine_version'])} · seed {sc['seed']} · runtime {report['runtime_seconds']} s · {status}</p>
<section><h2>1. Sharing scenario</h2>
<p><b>Recipient:</b> {_e(sc['recipient_label'])} · <b>Purpose:</b> {_e(sc['task_type'])}{(' of <code>' + _e(sc['target']) + '</code>') if sc.get('target') else ''}
 · <b>Key columns:</b> {_e(', '.join(sc['key_columns']) or '—')}</p>
<p>{_e(sc.get('description') or '')}</p>
<p><b>Threat model.</b> {_e(tm['statement'])}</p></section>
{plain}
<section><h2>2. Before / after</h2><table><tr><th>Metric</th><th>Before</th><th>After</th><th></th><th>Note</th></tr>{rows_k}</table></section>
<section><h2>3. Data types identified and PETs selected</h2>
<table><tr><th>Column</th><th>Role</th><th>Gate</th><th>PET</th><th>Final level</th><th>Reason</th></tr>{rows_d}</table></section>
<section><h2>4. Search path (privacy / utility trade-off)</h2>
<table><tr><th>Step</th><th>Move</th><th>Re-identification rate</th><th>Unique</th><th>min k</th><th>Suppressed</th><th>Utility proxy</th><th>Task metric retained</th></tr>{rows_p}</table></section>
<section><h2>5. Utility</h2>{task_html}
<p>Descriptive utility {num(d['score'], 3)}: {d['n_queries']} aggregate queries (counts/means by {_e(', '.join(d['group_columns']))}) with mean relative
error {pct(d['query_mean_rel_error'], 2)}; distribution score {num(d['distribution_score'], 3)} (TVD / Wasserstein per column).</p></section>
<section><h2>6. Free text</h2>{text_html or '<p>No free-text columns.</p>'}</section>
<section><h2>7. Residual risks and limitations</h2><ul>{risks}</ul></section>
<p class="muted">Pseudonymisation: {_e(report['config']['pseudonymisation'])}; secret from {_e(report['config']['secret_source'])} (never exported).
Inspired by the European project TrustED (Horizon Europe GA No. 101168467). Synthetic data only.</p>
</body></html>"""
