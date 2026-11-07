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
