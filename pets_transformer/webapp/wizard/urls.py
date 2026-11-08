from django.urls import path

from . import project_views, views

urlpatterns = [
    path("", views.home, name="home"),
    path("upload/", views.upload, name="upload"),
    path("job/<uuid:job_id>/", views.data, name="data"),
    path("job/<uuid:job_id>/profile/", views.profile, name="profile"),
    path("job/<uuid:job_id>/profile/<str:col>/", views.profile_update, name="profile_update"),
    path("job/<uuid:job_id>/suggest/<str:col>/", views.llm_suggest, name="llm_suggest"),
    path("job/<uuid:job_id>/scenario/", views.scenario, name="scenario"),
    path("job/<uuid:job_id>/scenario/summary/", views.scenario_summary, name="scenario_summary"),
    path("job/<uuid:job_id>/results/", views.results, name="results"),
    path("job/<uuid:job_id>/dp/", views.dp_panel, name="dp"),
    path("job/<uuid:job_id>/plain/", views.plain_summary, name="plain"),
    path("job/<uuid:job_id>/export/", views.export, name="export"),
    path("job/<uuid:job_id>/report/", views.report_view, name="report"),
    path("job/<uuid:job_id>/download/<str:name>", views.download, name="download"),
    path("job/<uuid:job_id>/delete/", views.delete_job, name="delete"),
    path("project/<uuid:project_id>/", project_views.project, name="project"),
    path("project/<uuid:project_id>/scenario/", project_views.project_scenario, name="project_scenario"),
    path("project/<uuid:project_id>/results/", project_views.project_results, name="project_results"),
    path("project/<uuid:project_id>/download/", project_views.project_download, name="project_download"),
    path("project/<uuid:project_id>/delete/", project_views.project_delete, name="project_delete"),
]
