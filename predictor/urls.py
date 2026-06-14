from django.urls import path

from . import views

app_name = "predictor"

urlpatterns = [
    path("", views.index, name="index"),
    path("random-lineup/", views.random_lineup, name="random_lineup"),
]
