from django.urls import path
from . import views

app_name = "ex_form8"
urlpatterns = [
    path("ex08/", views.ex08, name="ex08")
]

