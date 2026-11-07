"""Five-step wizard: Upload → Profile → Scenario → Results → Export."""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.http import FileResponse, Http404, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from engine import kit, llm
from engine.dp import dp_query, error_curve
from engine.gates import _default_red, assign_gates
from engine.ladders import make_ladder
from engine.pipeline import load_table, run
from engine.profiling import detect_domain, profile_dataframe
from engine.report import kpis, render_html, to_json
from engine.roles import RECIPIENTS, ColumnProfile, Gate, Kind, RedAction, Role, Scenario
from engine.text_pii import load_gold

from .models import Job

QI_ROLES = (Role.QUASI_ID.value, Role.DATE.value, Role.LOCATION.value)
TARGET_HINT = re.compile(r"readmit|target|label|outcome|churn|fraud|default|class|y$", re.I)
STEPS = ["Upload", "Profile", "Scenario", "Results", "Export"]


def _ctx(job: Job | None, step: int, **kw) -> dict:
    return {"job": job, "step": step, "steps": STEPS, "llm_enabled": llm.enabled(), **kw}


def _profiles(job: Job) -> list[ColumnProfile]:
    return [ColumnProfile.from_dict(d) for d in job.profile]


def _save_profiles(job: Job, profiles: list[ColumnProfile]) -> None:
    job.profile = [p.to_dict() for p in profiles]
    job.save(update_fields=["profile"])


def _read_json(job: Job, name: str):
    p = job.folder / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ---------------------------------------------------------------- step 1: upload
def home(request):
    return render(request, "wizard/home.html", _ctx(None, 1, jobs=Job.objects.all()[:8], error=request.GET.get("error")))


@require_POST
def upload(request):
    if request.POST.get("action") == "sample":
        src = Path(settings.SAMPLE_DIR) / "health_admissions.csv"
        spans = Path(settings.SAMPLE_DIR) / "health_admissions.spans.json"
        if not src.exists():
            from data.generate_sample import write
            write(settings.SAMPLE_DIR)
        job = Job.objects.create(name="Sample: hospital admissions (synthetic)")
        shutil.copy(src, job.folder / "original.csv")
        shutil.copy(spans, job.folder / "spans.json")
        job.has_gold = True
    else:
        f = request.FILES.get("data")
        if f is None:
            return redirect("/?error=Choose a CSV or XLSX file, or load the sample dataset.")
        job = Job.objects.create(name=f.name[:200])
        raw = job.folder / ("upload.xlsx" if f.name.lower().endswith((".xlsx", ".xls")) else "upload.csv")
        with open(raw, "wb") as out:
            for chunk in f.chunks():
                out.write(chunk)
        try:
            df = load_table(raw)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            job.delete()
            return redirect(f"/?error=Could not read the file: {exc}")
        raw.unlink()
        if df.empty or df.shape[1] == 0:
            shutil.rmtree(job.folder, ignore_errors=True)
            job.delete()
            return redirect("/?error=The file has no rows or no columns.")
        keep = None
        if len(df) > settings.MAX_ROWS:
            # large tables: a reproducible random sample of MAX_ROWS rows (row order preserved)
            total = len(df)
            keep = df.sample(n=settings.MAX_ROWS, random_state=42).index.sort_values()
            df = df.loc[keep].reset_index(drop=True)
            job.note = (f"The file has {total:,} rows; a reproducible random sample of {settings.MAX_ROWS:,} rows "
                        f"(seed 42) is used. Protection and metrics apply to this sample.")
        df.to_csv(job.folder / "original.csv", index=False)
        g = request.FILES.get("spans")
        if g is not None:
            try:
                gold = json.loads(g.read().decode("utf-8"))
                if keep is not None:  # re-number ground-truth rows to the sampled table
                    pos = {int(old): new for new, old in enumerate(keep)}
                    recs = gold["records"] if isinstance(gold, dict) else gold
                    recs = [dict(r, row=pos[int(r["row"])]) for r in recs if int(r["row"]) in pos]
                    gold = dict(gold, records=recs) if isinstance(gold, dict) else recs
                (job.folder / "spans.json").write_text(json.dumps(gold, ensure_ascii=False), encoding="utf-8")
                job.has_gold = True
            except (ValueError, KeyError, TypeError):
                job.note = (job.note + " " if job.note else "") + "The ground-truth spans file could not be read and was ignored."
    df = load_table(job.folder / "original.csv")
    profiles = assign_gates(profile_dataframe(df), df, Scenario())
    job.n_rows, job.n_cols = len(df), df.shape[1]
    job.domain = detect_domain(list(df.columns))[0] or ""
    job.profile = [p.to_dict() for p in profiles]
    job.status = "profiled"
    job.save()
    return redirect("data", job_id=job.id)


def data(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    head = pd.read_csv(job.folder / "original.csv", dtype=str, nrows=10, keep_default_na=False)
    return render(request, "wizard/data.html", _ctx(job, 1, columns=list(head.columns), rows=head.values.tolist()))


# ---------------------------------------------------------------- step 2: profile
def profile(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    return render(request, "wizard/profile.html", _ctx(job, 2, profiles=job.profile, roles=[r.value for r in Role],
                                                         gates=[g.value for g in Gate],
                                                         actions=[a.value for a in RedAction]))


def _ladder_for(p: ColumnProfile) -> str | None:
    if p.ladder:
        return p.ladder
    if p.role == Role.DATE or p.kind == Kind.DATE:
        return "date"
    if p.role == Role.FREE_TEXT:
        return None
    return "numeric" if p.dtype == "numeric" else "categorical"


@require_POST
def profile_update(request, job_id, col):
    job = get_object_or_404(Job, id=job_id)
    profiles = _profiles(job)
    p = next((x for x in profiles if x.name == col), None)
    if p is None:
        raise Http404
    if "role" in request.POST and request.POST["role"] != p.role.value:
        p.role = Role(request.POST["role"])
        if p.role == Role.FREE_TEXT:
            p.kind, p.ladder = Kind.TEXT, None
        elif p.role in (Role.DIRECT_ID, Role.TECHNICAL_ID):
            p.kind = Kind.ID
        else:
            p.ladder = _ladder_for(p)
        p.overridden, p.gate = False, None
        assign_gates([p], load_table(job.folder / "original.csv"), Scenario())
        p.overridden = True
        p.gate_reason = f"role set by the user; {p.gate_reason}"
    elif "gate" in request.POST and request.POST["gate"] != (p.gate.value if p.gate else None):
        p.gate = Gate(request.POST["gate"])
        p.overridden = True
        if p.gate == Gate.RED:
            p.red_action, reason = _default_red(p)
            p.gate_reason = f"set by the user; {reason}"
        else:
            p.red_action = None
            p.gate_reason = "set by the user"
            if p.gate == Gate.YELLOW:
                p.ladder = _ladder_for(p)
    elif "red_action" in request.POST:
        p.red_action = RedAction(request.POST["red_action"])
        p.overridden = True
        p.gate_reason = f"set by the user: {p.red_action.value}"
    _save_profiles(job, profiles)
    return render(request, "wizard/partials/profile_row.html",
                  {"p": p.to_dict(), "job": job, "roles": [r.value for r in Role], "gates": [g.value for g in Gate],
                   "actions": [a.value for a in RedAction], "llm_enabled": llm.enabled()})


@require_POST
def llm_suggest(request, job_id, col):
    job = get_object_or_404(Job, id=job_id)
    profiles = _profiles(job)
    p = next((x for x in profiles if x.name == col), None)
    if p is None:
        raise Http404
    df = load_table(job.folder / "original.csv")
    samples = df[col].dropna().astype(str).drop_duplicates().head(5).tolist()
    s = llm.suggest_role(col, samples)
    p.llm_suggestion = f"{s['role']}: {s['reason']}" if s else "no suggestion (LLM unavailable or declined)"
    _save_profiles(job, profiles)
    return HttpResponse(f"<span class='llm'>LLM suggests {p.llm_suggestion}</span>")


# ---------------------------------------------------------------- step 3: scenario
def _default_scenario(job: Job) -> dict:
    profs = job.profile
    qis = [p["name"] for p in profs if p["role"] in QI_ROLES and p["gate"] == "yellow"]
    target = next((p["name"] for p in profs if TARGET_HINT.search(p["name"]) and p["role"] not in ("DIRECT_ID",)), None)
    desc = ""
    if job.has_gold and job.name.startswith("Sample"):
        qis = [c for c in ("date_of_birth", "gender", "postcode", "admission_date") if c in qis] or qis
        desc = ("A hospital shares admissions data with an external research partner to build a 30-day readmission "
                "model. The attacker is assumed to know age/DOB, gender, postcode and admission date.")
    return {"recipient": "partner", "attacker_qis": qis, "task_type": "classification" if target else "descriptive",
            "target": target, "key_columns": [], "max_linkage": None, "min_k": None, "description": desc, "seed": 42}


def scenario(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    sc = job.scenario or _default_scenario(job)
    error = None
    if request.method == "POST":
        try:
            max_link = request.POST.get("max_linkage", "").strip()
            min_k = request.POST.get("min_k", "").strip()
            target = request.POST.get("target") or None
            task = request.POST.get("task_type", "descriptive")
            if task in ("classification", "regression") and not target:
                raise ValueError("Choose a target column for a classification or regression task.")
            sc = {
                "recipient": request.POST.get("recipient", "partner"),
                "attacker_qis": request.POST.getlist("attacker_qis"),
                "task_type": task, "target": target,
                "key_columns": request.POST.getlist("key_columns"),
                "max_linkage": float(max_link) / 100 if max_link else None,
                "min_k": int(min_k) if min_k else None,
                "description": request.POST.get("description", "")[:2000],
                "seed": int(request.POST.get("seed") or 42),
            }
            caps = {k[len("cap__"):]: v for k, v in request.POST.items() if k.startswith("cap__") and v != ""}
            sc["max_levels"] = caps
            kit_id = request.POST.get("kit_scenario") or ""
            if kit_id:
                cols = [p["name"] for p in job.profile]
                scen = kit.scenario_from_kit(kit_id, cols, request.POST.getlist(f"kit_tasks__{kit_id}"), seed=sc["seed"])
                scen.max_levels = {**scen.max_levels, **caps}
            else:
                scen = Scenario.from_dict(sc)
            job.scenario = scen.to_dict()
            _run_job(job, scen)
            if request.headers.get("HX-Request"):
                resp = HttpResponse()
                resp["HX-Redirect"] = f"/job/{job.id}/results/"
                return resp
            return redirect("results", job_id=job.id)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            job.status, job.error = "error", str(exc)
            job.save(update_fields=["status", "error"])
            error = str(exc)
    profs = job.profile
    cap_cols = []
    for p in profs:
        if p["gate"] == "yellow" and p["role"] != "FREE_TEXT" and p.get("ladder"):
            lad = make_ladder(p["name"], p["ladder"])
            if lad is not None:
                cur = (sc.get("max_levels") or {}).get(p["name"], "")
                cap_cols.append({"name": p["name"], "levels": list(enumerate(lad.level_names))[1:], "current": str(cur)})
    kit_options = kit.scenarios_for([p["name"] for p in profs])
    ctx = _ctx(job, 3, sc=sc, error=error, recipients=RECIPIENTS, cap_cols=cap_cols, kit_options=kit_options,
               qi_cols=[p for p in profs if p["gate"] != "red" and p["role"] != "FREE_TEXT"],
               all_cols=[p for p in profs if p["gate"] != "red" and p["role"] not in ("FREE_TEXT",)],
               recipients_json=json.dumps(RECIPIENTS))
    if error and request.headers.get("HX-Request"):
        return render(request, "wizard/partials/error.html", {"error": error})
    return render(request, "wizard/scenario.html", ctx)


def _run_job(job: Job, scen: Scenario) -> None:
    df = load_table(job.folder / "original.csv")
    gold, gold_col = None, None
    if job.has_gold and (job.folder / "spans.json").exists():
        gold_col, gold = load_gold(json.loads((job.folder / "spans.json").read_text(encoding="utf-8")))
    t = time.time()
    res = run(df, scen, profiles=_profiles(job), gold=gold, gold_column=gold_col)
    res.export_table().to_csv(job.folder / "protected.csv", index=False)
    res.rowmap.to_csv(job.folder / "rowmap.csv", index=False)
    (job.folder / "report.json").write_text(to_json(res.report), encoding="utf-8")
    (job.folder / "preview.json").write_text(json.dumps(res.preview, ensure_ascii=False), encoding="utf-8")
    (job.folder / "text_examples.json").write_text(json.dumps(res.text_examples, ensure_ascii=False), encoding="utf-8")
    (job.folder / "plain.txt").unlink(missing_ok=True)
    job.status, job.error, job.runtime = "done", "", round(time.time() - t, 1)
    job.save()


# ---------------------------------------------------------------- step 4: results
def results(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    report = _read_json(job, "report.json")
    if report is None:
        return redirect("scenario", job_id=job.id)
    preview = _read_json(job, "preview.json")
    examples = [dict(e, segments=_segments(e["original"], e["entities"]))
                for e in (_read_json(job, "text_examples.json") or [])]
    path = report["search_path"]
    final = path[-1]["step"]
    chart = {
        "proxy": [{"x": 1 - p["linkage"], "y": p["proxy"], "step": p["step"], "move": p["move"]} for p in path],
        "task": [{"x": 1 - p["linkage"], "y": p["task_retained"], "step": p["step"], "move": p["move"]}
                 for p in path if p.get("task_retained") is not None],
        "final": final,
        "target_x": 1 - report["scenario"]["max_linkage"],
    }
    pcols = preview["protected_columns"]
    prev_rows = []
    for r in preview["rows"]:
        prev_rows.append({
            "row": r["row"], "suppressed": r["suppressed"],
            "orig": [(c, r["original"].get(c, "")) for c in preview["columns"]],
            "prot": [(c, r["protected"].get(c, ""), c in r["changed"]) for c in pcols],
        })
    # group-by candidates for the DP panel: low-cardinality categorical columns first
    cand = [p for p in report["profiles"] if p["gate"] != "red" and p["role"] not in ("FREE_TEXT", "DIRECT_ID", "TECHNICAL_ID")]
    sens_cols = [p["name"] for p in sorted(cand, key=lambda p: (p["unique_ratio"] > 0.01, p["unique_ratio"]))]
    num_cols = [p["name"] for p in report["profiles"] if p["dtype"] == "numeric" and p["gate"] != "red"]
    plain = (job.folder / "plain.txt").read_text(encoding="utf-8") if (job.folder / "plain.txt").exists() else None
    ctx = _ctx(job, 4, r=report, kpis=kpis(report), chart=chart, preview_cols=preview["columns"], pcols=pcols,
               preview_rows=prev_rows, examples=examples[:6], group_cols=sens_cols, num_cols=num_cols, plain=plain,
               k_dist_before=report["metrics"]["before"]["k_distribution"],
               k_dist_after=report["metrics"]["after"]["k_distribution"])
    return render(request, "wizard/results.html", ctx)


def _segments(text: str, entities: list[dict]) -> list[tuple[str, str | None]]:
    """Split a text into (piece, label) segments so detected entities can be highlighted safely in the template."""
    out, pos = [], 0
    for e in sorted(entities, key=lambda e: e["start"]):
        if e["start"] < pos:
            continue
        out.append((text[pos:e["start"]], None))
        out.append((text[e["start"]:e["end"]], e["label"]))
        pos = e["end"]
    out.append((text[pos:], None))
    return out


@require_POST
def dp_panel(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    df = load_table(job.folder / "original.csv")
    group = request.POST.get("group")
    stat = request.POST.get("stat", "count")
    value = request.POST.get("value") or None
    eps = float(request.POST.get("eps", 1.0))
    seed = int((job.scenario or {}).get("seed", 42))
    try:
        res = dp_query(df, group, stat, eps, value, seed=seed)
        curve = error_curve(df, group, stat, value, seed=seed, repeats=10)
    except Exception as exc:  # noqa: BLE001
        return render(request, "wizard/partials/error.html", {"error": str(exc)})
    rows = sorted(res["rows"], key=lambda r: -(r["true"] or 0))[:25]
    return render(request, "wizard/partials/dp.html", {"res": res, "rows": rows, "curve": curve,
                                                        "chart": json.dumps({"rows": rows, "curve": curve})})


@require_POST
def plain_summary(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    report = _read_json(job, "report.json")
    text = llm.plain_language(report) if report else None
    if text:
        (job.folder / "plain.txt").write_text(text, encoding="utf-8")
    return render(request, "wizard/partials/plain.html", {"plain": text})


# ---------------------------------------------------------------- step 5: export
def export(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    report = _read_json(job, "report.json")
    if report is None:
        return redirect("scenario", job_id=job.id)
    return render(request, "wizard/export.html", _ctx(job, 5, r=report))


DOWNLOADS = {"protected.csv": "text/csv", "rowmap.csv": "text/csv", "report.json": "application/json",
             "report.html": "text/html"}


def download(request, job_id, name):
    """Only protected outputs can be downloaded; the original data and the pseudonymisation secret never are."""
    job = get_object_or_404(Job, id=job_id)
    if name not in DOWNLOADS:
        raise Http404
    stem = Path(job.name).stem if not job.name.startswith("Sample") else "health_admissions"
    if name == "report.html":
        report = _read_json(job, "report.json")
        if report is None:
            raise Http404
        plain = (job.folder / "plain.txt").read_text(encoding="utf-8") if (job.folder / "plain.txt").exists() else None
        resp = HttpResponse(render_html(report, plain), content_type="text/html; charset=utf-8")
        resp["Content-Disposition"] = f'attachment; filename="{stem}_report.html"'
        return resp
    path = job.folder / name
    if not path.exists():
        raise Http404
    # kit conventions: the table keeps its original file name; the row map is rowmap_<table>.csv
    fname = {"protected.csv": f"{stem}.csv", "rowmap.csv": f"rowmap_{stem}.csv"}.get(name, f"{stem}_{name}")
    return FileResponse(open(path, "rb"), as_attachment=True, filename=fname, content_type=DOWNLOADS[name])


def report_view(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    report = _read_json(job, "report.json")
    if report is None:
        raise Http404
    plain = (job.folder / "plain.txt").read_text(encoding="utf-8") if (job.folder / "plain.txt").exists() else None
    return HttpResponse(render_html(report, plain))


@require_POST
def delete_job(request, job_id):
    job = get_object_or_404(Job, id=job_id)
    shutil.rmtree(job.folder, ignore_errors=True)
    job.delete()
    return redirect("home")
