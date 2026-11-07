from django.contrib.staticfiles.views import serve
from django.urls import include, path, re_path

urlpatterns = [
    path("", include("webapp.wizard.urls")),
    # local tool: serve the app's CSS also when DEBUG is off (behind a real web server, use collectstatic instead)
    re_path(r"^static/(?P<path>.*)$", serve, {"insecure": True}),
]
