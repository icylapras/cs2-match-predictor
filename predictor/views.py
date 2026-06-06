from django.shortcuts import render

from src.faceit_api import FaceitError
from src.features import FEATURE_KEYS

from .forms import DEMO_NICKNAMES, MatchForm, random_lineup_initial
from .models import PredictionLog
from .services import run_prediction

COIN_FLIP_BAND = 0.05  # |prob - 0.5| below this is "too close to call"


def _result_context(team_a, team_b, prediction):
    """Shape a prediction dict into template-friendly values and log it."""
    prob_a = prediction["prob_a"]
    if prob_a >= 0.5:
        favored, confidence = "Team A", prob_a
    else:
        favored, confidence = "Team B", 1 - prob_a

    PredictionLog.objects.create(
        team_a=", ".join(team_a),
        team_b=", ".join(team_b),
        prob_a=prob_a,
        favored=favored,
        confidence=confidence,
    )

    rows = [
        {
            "metric": key.replace("_", " "),
            "a": prediction["avg_a"][key],
            "b": prediction["avg_b"][key],
        }
        for key in FEATURE_KEYS
    ]
    return {
        "prob_a": prob_a,
        "prob_a_pct": round(prob_a * 100),
        "favored": favored,
        "confidence_pct": round(confidence * 100),
        "coin_flip": abs(prob_a - 0.5) < COIN_FLIP_BAND,
        "rows": rows,
    }


def index(request):
    context = {
        "recent": PredictionLog.objects.all()[:5],
        "demo_nicknames": DEMO_NICKNAMES,  # pool for the client-side shuffle button
    }

    if request.method == "POST":
        form = MatchForm(request.POST)
        if form.is_valid():
            team_a, team_b = form.team_a(), form.team_b()
            try:
                prediction = run_prediction(team_a, team_b)
            except FaceitError as exc:
                context["error"] = str(exc)
            else:
                context["result"] = _result_context(team_a, team_b, prediction)
                context["recent"] = PredictionLog.objects.all()[:5]
    else:
        form = MatchForm(initial=random_lineup_initial())

    context["form"] = form
    return render(request, "predictor/index.html", context)
