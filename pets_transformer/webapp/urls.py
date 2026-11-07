from django.urls import include, path

urlpatterns = [path("", include("webapp.wizard.urls"))]
