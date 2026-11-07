import uuid
from pathlib import Path

from django.conf import settings
from django.db import models


class Job(models.Model):
    """One protection job. Files live in MEDIA_ROOT/jobs/<id>/; the original never leaves that folder."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=200)
    n_rows = models.IntegerField(default=0)
    n_cols = models.IntegerField(default=0)
    domain = models.CharField(max_length=60, blank=True, default="")
    has_gold = models.BooleanField(default=False)
    profile = models.JSONField(default=list)
    scenario = models.JSONField(default=dict, blank=True)
    status = models.CharField(max_length=20, default="uploaded")  # uploaded | profiled | done | error
    error = models.TextField(blank=True, default="")
    note = models.TextField(blank=True, default="")
    runtime = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ["-created"]

    @property
    def folder(self) -> Path:
        p = Path(settings.MEDIA_ROOT) / "jobs" / str(self.id)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def step(self) -> int:
        return {"uploaded": 1, "profiled": 2, "done": 4, "error": 3}.get(self.status, 1)
