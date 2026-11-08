"""Several related tables uploaded together: detect links, protect them as one project, export everything."""
from __future__ import annotations

import io
import shutil
import time
import zipfile
from pathlib import Path

from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from engine import kit
from engine.pipeline import load_table
from engine.project import run_project
from engine.relations import detect_links
from engine.report import render_html, to_json
from engine.roles import RECIPIENTS, ColumnProfile, Scenario

from .models import Job
from .views import PURPOSES, _column_groups, _ctx, known_preset, save_result, table_job_from_upload

PROJECT_PURPOSES = [p for p in PURPOSES if p[0] in ("descriptive", "visualization")]


def _members(project: Job) -> list[Job]:
    return list(project.members.order_by("created"))


def create_project(request, files) -> HttpResponse:
    project = Job.objects.create(name="Project: " + ", ".join(f.name for f in files)[:180], kind="project")
    errors = []
    for f in files:
        _, err = table_job_from_upload(f, project=project)
        if err:
            errors.append(err)
    members = _members(project)
    if len(members) < 2:
        for m in members:
            shutil.rmtree(m.folder, ignore_errors=True)
        shutil.rmtree(project.folder, ignore_errors=True)
        project.delete()
        return redirect("/?error=" + (" ".join(errors) or "A project needs at least two readable tables."))
    tables = {Path(m.name).stem: load_table(m.folder / "original.csv") for m in members}
    profiles = {Path(m.name).stem: [ColumnProfile.from_dict(d) for d in m.profile] for m in members}
    project.links = [dict(l, keep=True) for l in detect_links(tables, profiles)]
    project.n_rows = sum(m.n_rows for m in members)
    project.n_cols = sum(m.n_cols for m in members)
    project.note = " ".join(errors)
    project.status = "profiled"
    project.save()
    return redirect("project", project_id=project.id)


def project(request, project_id):
    proj = get_object_or_404(Job, id=project_id, kind="project")
    if request.method == "POST":   # keep or drop detected links
        keep = set(request.POST.getlist("keep"))
        proj.links = [dict(l, keep=str(i) in keep) for i, l in enumerate(proj.links)]
        proj.save(update_fields=["links"])
        return redirect("project_scenario", project_id=proj.id)
    return render(request, "wizard/project.html", _ctx(proj, 1, members=_members(proj),
                                                         links=list(enumerate(proj.links))))


def _kit_options(members: list[Job]) -> list[dict]:
    for m in members:
        opts = kit.scenarios_for([p["name"] for p in m.profile])
        if opts:
            return opts
    return []


def _member_scenario(post, m: Job) -> Scenario:
    cols = [p["name"] for p in m.profile]
    kit_id = post.get("kit_scenario") or ""
    seed = int(post.get("seed") or 42)
    if kit_id:
        try:
            return kit.scenario_from_kit(kit_id, cols, seed=seed)
        except ValueError:
            pass   # this table is not part of the kit scenario's dataset: use the manual settings
    groups = _column_groups(m.profile, m.n_rows)
    recipient = post.get("recipient", "partner")
    purposes = [p for p in post.getlist("purposes") if p in ("descriptive", "visualization")] or ["descriptive"]
    targets = {}
    for kind in ("classification", "regression"):
        t = post.get(f"target_{kind}__{m.id}")
        if t:
            targets[kind] = [t]
            purposes = [kind] + purposes
    known = [c for c in known_preset(recipient, groups) if c not in {c for v in targets.values() for c in v}]
    max_link = (post.get("max_linkage") or "").strip()
    min_k = (post.get("min_k") or "").strip()
    return Scenario(recipient=recipient, attacker_qis=known, purposes=purposes, targets=targets,
                    task_type=purposes[0], max_linkage=float(max_link) / 100 if max_link else None,
                    min_k=int(min_k) if min_k else None, seed=seed, description=post.get("description", "")[:2000])


def project_scenario(request, project_id):
    proj = get_object_or_404(Job, id=project_id, kind="project")
    members = _members(proj)
    error = None
    if request.method == "POST":
        try:
            names = {m.id: Path(m.name).stem for m in members}
            tables = {names[m.id]: load_table(m.folder / "original.csv") for m in members}
            profiles = {names[m.id]: [ColumnProfile.from_dict(d) for d in m.profile] for m in members}
            scenarios = {names[m.id]: _member_scenario(request.POST, m) for m in members}
            links = [{k: v for k, v in l.items() if k != "keep"} for l in proj.links if l.get("keep", True)]
            t = time.time()
            res = run_project(tables, scenarios, links=links, profiles=profiles)
            for m in members:
                n = names[m.id]
                m.scenario = scenarios[n].to_dict()
                save_result(m, res.results[n], round(res.results[n].report["runtime_seconds"] or 0, 1))
            proj.result = res.to_dict()
            proj.scenario = {"kit_scenario": request.POST.get("kit_scenario") or None,
                             "recipient": request.POST.get("recipient", "partner")}
            proj.status, proj.error, proj.runtime = "done", "", round(time.time() - t, 1)
            proj.save()
            if request.headers.get("HX-Request"):
                resp = HttpResponse()
                resp["HX-Redirect"] = f"/project/{proj.id}/results/"
                return resp
            return redirect("project_results", project_id=proj.id)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            proj.status, proj.error = "error", str(exc)
            proj.save(update_fields=["status", "error"])
            error = str(exc)
            if request.headers.get("HX-Request"):
                return render(request, "wizard/partials/error.html", {"error": error})
    rows = []
    for m in members:
        g = _column_groups(m.profile, m.n_rows)["target_by_kind"]
        rows.append({"job": m, "cat": g["classification"]["plain"], "num": g["regression"]["plain"],
                     "qis": [p["name"] for p in m.profile if p["role"] in ("QUASI_ID", "DATE", "LOCATION") and p["gate"] != "red"]})
    return render(request, "wizard/project_scenario.html", _ctx(
        proj, 3, rows=rows, recipients=RECIPIENTS, purposes=PROJECT_PURPOSES, kit_options=_kit_options(members),
        error=error, links=[l for l in proj.links if l.get("keep", True)]))


def project_results(request, project_id):
    proj = get_object_or_404(Job, id=project_id, kind="project")
    if not proj.result:
        return redirect("project_scenario", project_id=proj.id)
    by_name = {Path(m.name).stem: m for m in _members(proj)}
    tables = [dict(t, job=by_name.get(t["table"])) for t in proj.result.get("tables", [])]
    return render(request, "wizard/project_results.html", _ctx(proj, 4, tables=tables, links=proj.result.get("links", [])))


def project_download(request, project_id):
    """One zip: every protected table (original file name), its row map (scoring only) and its reports."""
    proj = get_object_or_404(Job, id=project_id, kind="project")
    if not proj.result:
        return redirect("project_scenario", project_id=proj.id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for m in _members(proj):
            n = Path(m.name).stem
            if (m.folder / "protected.csv").exists():
                z.write(m.folder / "protected.csv", f"{n}.csv")
                z.write(m.folder / "rowmap.csv", f"rowmap_{n}.csv")
                report = (m.folder / "report.json").read_text(encoding="utf-8")
                z.writestr(f"reports/{n}.report.json", report)
                import json
                z.writestr(f"reports/{n}.report.html", render_html(json.loads(report)))
        z.writestr("reports/project.json", to_json(proj.result))
        z.writestr("README.txt", "Protected tables (original file names) and their row maps.\n"
                                 "rowmap_<table>.csv links released rows to original record ids: for scoring only, "
                                 "never publish it.\n")
    resp = HttpResponse(buf.getvalue(), content_type="application/zip")
    resp["Content-Disposition"] = 'attachment; filename="protected_project.zip"'
    return resp


@require_POST
def project_delete(request, project_id):
    proj = get_object_or_404(Job, id=project_id, kind="project")
    for m in _members(proj):
        shutil.rmtree(m.folder, ignore_errors=True)
    shutil.rmtree(proj.folder, ignore_errors=True)
    proj.delete()
    return redirect("home")
