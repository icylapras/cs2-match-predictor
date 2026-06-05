from django.db import models


class PredictionLog(models.Model):
    """A record of one matchup prediction made through the app.

    Stored so the homepage can show recent predictions and to demonstrate a
    real persistence layer (and as raw material for future analysis).
    """

    team_a = models.CharField(max_length=255)  # comma-joined nicknames
    team_b = models.CharField(max_length=255)
    prob_a = models.FloatField(help_text="Model probability that team A wins")
    favored = models.CharField(max_length=8)  # "Team A" or "Team B"
    confidence = models.FloatField(help_text="Confidence in the favored team (0-1)")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.favored} {self.confidence:.0%} ({self.created_at:%Y-%m-%d %H:%M})"
