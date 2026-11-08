"""End-to-end wizard flow through the Django test client (Upload → Profile → Scenario → Results → Export)."""
import json
import os

import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webapp.settings")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import django
    from django.conf import settings
    django.setup()
    settings.MEDIA_ROOT = tmp_path_factory.mktemp("media")
    settings.ALLOWED_HOSTS = ["testserver"]
    from django.db import connection
    from django.test.utils import setup_test_environment, teardown_test_environment
    setup_test_environment()
    old = connection.creation.create_test_db(verbosity=0)
    from django.test import Client
    yield Client()
    connection.creation.destroy_test_db(old, verbosity=0)
    teardown_test_environment()


def test_full_wizard_flow(client):
    from webapp.wizard.models import Job
    r = client.get("/")
    assert r.status_code == 200 and b"Load sample dataset" in r.content

    r = client.post("/upload/", {"action": "sample"})
    assert r.status_code == 302
    job = Job.objects.latest("created")
    assert job.n_rows == 5000 and job.has_gold

    r = client.get(f"/job/{job.id}/")
    assert b"First 10 rows" in r.content

    r = client.get(f"/job/{job.id}/profile/")
    assert r.status_code == 200 and b"national_id" in r.content and b"gate red" in r.content

    # HTMX inline override: make department a quasi-identifier, then back to categorical
    r = client.post(f"/job/{job.id}/profile/department/", {"role": "QUASI_ID"}, HTTP_HX_REQUEST="true")
    assert r.status_code == 200 and b"gate yellow" in r.content
    r = client.post(f"/job/{job.id}/profile/department/", {"role": "CATEGORICAL"}, HTTP_HX_REQUEST="true")
    assert b"gate green" in r.content

    r = client.get(f"/job/{job.id}/scenario/")
    assert r.status_code == 200 and b"Threat model: known attributes" in r.content

    r = client.post(f"/job/{job.id}/scenario/", {
        "recipient": "partner", "attacker_qis": ["date_of_birth", "gender", "postcode", "admission_date"],
        "task_type": "classification", "target": "readmitted_30d", "key_columns": ["department"],
        "max_linkage": "", "min_k": "", "description": "demo", "seed": "42"}, HTTP_HX_REQUEST="true")
    assert r.status_code == 200, r.content[:500]
    assert r["HX-Redirect"].endswith("/results/")

    r = client.get(f"/job/{job.id}/results/")
    assert r.status_code == 200
    assert b"target met" in r.content and b"Privacy / utility trade-off" in r.content
    assert b"Example redactions" in r.content

    r = client.post(f"/job/{job.id}/dp/", {"group": "department", "stat": "mean", "value": "length_of_stay", "eps": "1"})
    assert r.status_code == 200 and b"Laplace" in r.content

    r = client.get(f"/job/{job.id}/export/")
    assert b"Download protected CSV" in r.content
    csv = b"".join(client.get(f"/job/{job.id}/download/protected.csv").streaming_content).decode("utf-8")
    rep = json.loads(b"".join(client.get(f"/job/{job.id}/download/report.json").streaming_content))
    html = client.get(f"/job/{job.id}/download/report.html").content.decode("utf-8")
    assert rep["target_met"] and "patient_name" not in csv.splitlines()[0]
    original = (job.folder / "original.csv").read_text(encoding="utf-8").splitlines()[1].split(",")
    name, email = original[0], original[1]
    for blob in (csv, json.dumps(rep, ensure_ascii=False), html):
        assert name not in blob and email not in blob
    # the original file can never be downloaded
    assert client.get(f"/job/{job.id}/download/original.csv").status_code == 404


def test_each_upload_uses_its_own_file_and_large_files_are_sampled(client):
    from django.core.files.uploadedfile import SimpleUploadedFile
    from webapp.wizard.models import Job

    def up(name, text):
        r = client.post("/upload/", {"data": SimpleUploadedFile(name, text.encode("utf-8"), content_type="text/csv")})
        assert r.status_code == 302 and "/job/" in r["Location"], r["Location"]
        return Job.objects.get(id=r["Location"].strip("/").split("/")[1])

    a = up("first.csv", "email,age\n" + "".join(f"a{i}@example.fi,{20 + i % 50}\n" for i in range(300)))
    b = up("second.csv", "postcode,score,city\n" + "".join(f"20{i % 9}00,{i},Turku\n" for i in range(200)))
    assert a.id != b.id
    assert (a.n_rows, a.n_cols) == (300, 2) and (b.n_rows, b.n_cols) == (200, 3)
    assert [p["name"] for p in b.profile] == ["postcode", "score", "city"]
    assert b"second.csv" in client.get(f"/job/{b.id}/").content

    big = up("big.csv", "id,value\n" + "".join(f"{i},{i % 7}\n" for i in range(60_000)))
    assert big.n_rows == 50_000 and "60,000 rows" in big.note
    r = client.get(f"/job/{big.id}/")
    assert b"random sample of 50,000" in r.content and b"Continue" in r.content


def test_results_with_utility_caps_and_rowmap_download(client):
    from webapp.wizard.models import Job
    client.post("/upload/", {"action": "sample"})
    job = Job.objects.latest("created")
    r = client.post(f"/job/{job.id}/scenario/", {
        "recipient": "partner", "attacker_qis": ["date_of_birth", "gender", "postcode", "admission_date"],
        "task_type": "descriptive", "target": "", "seed": "42", "cap__admission_date": "1", "cap__postcode": "nonsense-level"})
    assert r.status_code in (200, 302)
    r = client.get(f"/job/{job.id}/results/")
    assert r.status_code == 200 and b"Utility requirements" in r.content
    rm = client.get(f"/job/{job.id}/download/rowmap.csv")
    assert rm.status_code == 200 and "rowmap_health_admissions.csv" in rm["Content-Disposition"]
    head = b"".join(rm.streaming_content).decode().splitlines()[0]
    assert head == "protected_row,original_record_id"


def test_scenario_summary_and_target_rules(client):
    from webapp.wizard.models import Job
    from webapp.wizard.views import _column_groups, known_preset
    client.post("/upload/", {"action": "sample"})
    job = Job.objects.latest("created")
    page = client.get(f"/job/{job.id}/scenario/").content.decode()
    for text in ("Who receives the data", "Purpose: what will the data be used for?", "Train a prediction model: categories",
                 "Threat model: known attributes", "Advanced settings", "Target columns for"):
        assert text in page
    assert page.index("Who receives the data") < page.index("Purpose:") < page.index("Threat model: known attributes") \
        < page.index("Advanced settings")
    assert '<option value="patient_name"' not in page            # direct identifiers are not offered as target
    base = {"recipient": "partner", "task_type": "classification", "target": "readmitted_30d", "seed": "42",
            "attacker_qis": ["date_of_birth", "gender", "postcode"]}
    r = client.post(f"/job/{job.id}/scenario/summary/", base).content.decode()
    assert "With <b>3</b> known attributes" in r and "re-identifiable (unique)" in r and "ROC-AUC and F1" in r
    r = client.post(f"/job/{job.id}/scenario/summary/", {**base, "attacker_qis": ["gender", "readmitted_30d"]}).content.decode()
    assert "also ticked as a known attribute" in r
    r = client.post(f"/job/{job.id}/scenario/", {**base, "attacker_qis": ["gender", "readmitted_30d"]}, HTTP_HX_REQUEST="true")
    assert b"also ticked as a known attribute" in r.content
    r = client.post(f"/job/{job.id}/scenario/", {**base, "target": ""}, HTTP_HX_REQUEST="true")
    assert b"Choose at least one target column" in r.content
    groups = _column_groups(job.profile, job.n_rows)
    assert set(known_preset("public", groups)) == set(groups["qis"])
    assert set(known_preset("internal", groups)) > set(groups["qis"]) and "department" in known_preset("internal", groups)


def test_results_show_constraints_and_unmet_target(client):
    from webapp.wizard.models import Job
    client.post("/upload/", {"action": "sample"})
    job = Job.objects.latest("created")
    # an impossible target (0 % with every attribute kept exact) cannot be met
    caps = {f"cap__{c}": "0" for c in ("date_of_birth", "postcode", "admission_date", "discharge_date", "gender", "city")}
    r = client.post(f"/job/{job.id}/scenario/", {
        "recipient": "public", "attacker_qis": ["date_of_birth", "gender", "postcode", "admission_date"],
        "task_type": "descriptive", "seed": "42", "max_linkage": "0", **caps}, HTTP_HX_REQUEST="true")
    assert r.status_code == 200
    page = client.get(f"/job/{job.id}/results/").content.decode()
    assert "target NOT met" in page and "Final point (target not met)" in page and "No step met the target" in page
    assert "What constrained each column" in page and "utility limit" in page and "reached this limit" in page


def test_several_purposes_at_once(client):
    from webapp.wizard.models import Job
    client.post("/upload/", {"action": "sample"})
    job = Job.objects.latest("created")
    page = client.get(f"/job/{job.id}/scenario/").content.decode()
    assert 'type="checkbox" name="purposes" value="visualization"' in page
    base = {"recipient": "partner", "seed": "42", "attacker_qis": ["date_of_birth", "gender", "postcode", "admission_date"],
            "purposes": ["classification", "regression", "visualization"],
            "target_classification": "readmitted_30d", "target_regression": "length_of_stay"}
    r = client.post(f"/job/{job.id}/scenario/summary/", base).content.decode()
    assert "predicting readmitted_30d" in r and "predicting length_of_stay" in r and "mean over these purposes" in r
    r = client.post(f"/job/{job.id}/scenario/summary/", {**base, "target_regression": ""}).content.decode()
    assert "Choose at least one target column for" in r
    r = client.post(f"/job/{job.id}/scenario/", base, HTTP_HX_REQUEST="true")
    assert r.status_code == 200 and r["HX-Redirect"].endswith("/results/")
    job.refresh_from_db()
    import json
    rep = json.loads((job.folder / "report.json").read_text(encoding="utf-8"))
    u = rep["utility"]
    assert set(u["by_purpose"]) == {"classification", "regression", "visualization"}
    assert {t["task"] for t in u["tasks"]} == {"classification", "regression"}
    assert abs(u["score"] - sum(v["score"] for v in u["by_purpose"].values()) / 3) < 1e-9
    page = client.get(f"/job/{job.id}/results/").content.decode()
    assert "Utility by purpose" in page and page.count("Task utility:") == 2


def test_several_targets_per_prediction_purpose(client):
    import json
    from webapp.wizard.models import Job
    client.post("/upload/", {"action": "sample"})
    job = Job.objects.latest("created")
    base = {"recipient": "partner", "seed": "42", "attacker_qis": ["date_of_birth", "gender", "postcode", "admission_date"],
            "purposes": ["regression"], "target_regression": ["length_of_stay", "lab_value_1"]}
    r = client.post(f"/job/{job.id}/scenario/summary/", base).content.decode()
    assert "predicting length_of_stay, lab_value_1" in r
    too_many = {**base, "target_regression": ["length_of_stay", "lab_value_1", "lab_value_2", "num_prior_visits",
                                              "readmitted_30d", "department"]}
    assert "At most 5 target columns" in client.post(f"/job/{job.id}/scenario/summary/", too_many).content.decode()
    r = client.post(f"/job/{job.id}/scenario/", base, HTTP_HX_REQUEST="true")
    assert r["HX-Redirect"].endswith("/results/")
    job.refresh_from_db()
    u = json.loads((job.folder / "report.json").read_text(encoding="utf-8"))["utility"]
    assert sorted(t["target"] for t in u["tasks"]) == ["lab_value_1", "length_of_stay"]
    assert abs(u["by_purpose"]["regression"]["score"] - sum(t["score"] for t in u["tasks"]) / 2) < 1e-9
    assert "mean over 2 models" in u["by_purpose"]["regression"]["measured_by"]


def test_kit_scenario_is_an_editable_preset(client):
    from django.core.files.uploadedfile import SimpleUploadedFile
    from engine import kit
    from webapp.wizard.models import Job
    cols = list(kit.catalog().query("dataset == 'health' and table == 'ed_visits'")["column"])
    rows = []
    for i in range(60):
        row = {c: f"v{i % 7}" for c in cols}
        row.update({"visit_id": f"ED25-{i:06d}", "date_of_birth": f"19{50 + i % 40}-0{1 + i % 9}-1{i % 9}",
                    "sex": "FM"[i % 2], "municipality": ["Turku", "Salo", "Raisio"][i % 3],
                    "arrival_ts": f"2025-01-{1 + i % 28:02d} 1{i % 10}:00:00", "readmit_30d": str(i % 2)})
        rows.append(row)
    import csv, io
    buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=cols); w.writeheader(); w.writerows(rows)
    r = client.post("/upload/", {"data": SimpleUploadedFile("ed_visits.csv", buf.getvalue().encode(), content_type="text/csv")})
    job = Job.objects.get(id=r["Location"].strip("/").split("/")[1])
    page = client.get(f"/job/{job.id}/scenario/").content.decode()
    assert 'value="HEALTH-S1" selected' in page
    assert "disabled" not in page.split('id="manual"')[1][:40]           # the sections stay editable
    assert 'name="attacker_qis" value="sex"' in page and "Filled in from" in page
    # the form overrides the kit preset: recipient changed to internal -> its thresholds apply
    post = {"kit_scenario": "HEALTH-S1", "kit_tasks__HEALTH-S1": ["H-U2", "H-U6"], "recipient": "internal", "seed": "42",
            "purposes": ["visualization"], "attacker_qis": ["sex", "municipality"]}
    r = client.post(f"/job/{job.id}/scenario/summary/", post).content.decode()
    assert "Target: ≤ <b>10 %</b>" in r and "With <b>2</b> known attributes" in r and "(HEALTH-S1)" in r


def test_target_options_follow_column_type(client):
    from webapp.wizard.models import Job
    from webapp.wizard.views import _column_groups
    client.post("/upload/", {"action": "sample"})
    job = Job.objects.latest("created")
    g = _column_groups(job.profile, job.n_rows)["target_by_kind"]
    cat = g["classification"]["plain"] + g["classification"]["qi"]
    num = g["regression"]["plain"] + g["regression"]["qi"]
    assert "readmitted_30d" in cat and "department" in cat           # 0/1 flag and a text category
    assert "length_of_stay" in num and "lab_value_1" in num and "readmitted_30d" not in num
    assert "lab_value_1" not in cat and "clinician_notes" not in cat + num and "patient_name" not in cat + num
    assert "postcode" not in num and "date_of_birth" not in cat + num
    r = client.post(f"/job/{job.id}/scenario/summary/", {"recipient": "partner", "seed": "42", "purposes": ["regression"],
                                                         "target_regression": ["department"], "attacker_qis": ["gender"]})
    assert "cannot be a target for" in r.content.decode()


def test_project_upload_links_run_and_zip(client):
    import io, zipfile
    from django.core.files.uploadedfile import SimpleUploadedFile
    from webapp.wizard.models import Job
    visits = "visit_id,date_of_birth,sex,postcode,arrival_ts,diagnosis\n" + "".join(
        f"V{i:04d},19{50 + i % 45}-0{1 + i % 9}-1{i % 9},{'FM'[i % 2]},20{100 + (i % 9) * 100},"
        f"2025-0{1 + i % 9}-1{i % 9} 1{i % 10}:00:00,{['flu', 'fracture', 'asthma'][i % 3]}\n" for i in range(300))
    labs = "lab_id,visit_id,test_code,value\n" + "".join(
        f"L{i:05d},V{(i * 7) % 300:04d},{['CRP', 'HB', 'K'][i % 3]},{(i * 13) % 97}.5\n" for i in range(700))
    r = client.post("/upload/", {"data": [SimpleUploadedFile("visits.csv", visits.encode(), content_type="text/csv"),
                                          SimpleUploadedFile("labs.csv", labs.encode(), content_type="text/csv")]})
    assert r.status_code == 302 and "/project/" in r["Location"], r["Location"]
    proj = Job.objects.get(id=r["Location"].strip("/").split("/")[1])
    assert proj.kind == "project" and proj.members.count() == 2
    page = client.get(f"/project/{proj.id}/").content.decode()
    assert "Links between tables" in page and "labs.<b>visit_id</b>" in page and "visits.<b>visit_id</b>" in page
    r = client.post(f"/project/{proj.id}/", {"keep": ["0"]})
    assert r.status_code == 302
    assert "Prediction models per table" in client.get(f"/project/{proj.id}/scenario/").content.decode()
    r = client.post(f"/project/{proj.id}/scenario/", {"recipient": "partner", "purposes": ["descriptive"], "seed": "42"},
                    HTTP_HX_REQUEST="true")
    assert r.status_code == 200 and r["HX-Redirect"].endswith("/results/"), r.content[:300]
    page = client.get(f"/project/{proj.id}/results/").content.decode()
    assert "Links after protection" in page and "attributes of visits" in page
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/project/{proj.id}/download/").content))
    names = set(z.namelist())
    assert {"visits.csv", "labs.csv", "rowmap_visits.csv", "rowmap_labs.csv", "reports/project.json"} <= names
    labs_out = z.read("labs.csv").decode()
    assert "V0001" not in labs_out                    # the link column is pseudonymised
    home = client.get("/").content.decode()
    assert "Project: visits.csv, labs.csv" in home and 'href="/job/' not in home.split("Project: visits.csv")[1][:200]
