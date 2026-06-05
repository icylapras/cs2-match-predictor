from django.contrib import admin

from .models import PredictionLog


@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = ("created_at", "favored", "confidence", "team_a", "team_b")
    list_filter = ("favored",)
    readonly_fields = ("created_at",)
