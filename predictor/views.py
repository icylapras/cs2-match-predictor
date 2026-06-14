from django.http import JsonResponse
from django.shortcuts import render

from src.faceit_api import FaceitError, random_ranked_players
from src.features import FEATURE_KEYS

from .forms import DEMO_NICKNAMES, TEAM_SIZE, MatchForm, field_names, random_lineup_initial
from .models import PredictionLog
from .services import get_metrics, run_prediction

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

    # FACEIT Elo baseline bar (model-free), shown beside the model's bar.
    prob_a_elo = prediction.get("prob_a_elo")
    elo = None
    if prob_a_elo is not None:
        elo_favored = "Team A" if prob_a_elo >= 0.5 else "Team B"
        elo = {
            "prob_a_pct": round(prob_a_elo * 100),
            "favored": elo_favored,
            "confidence_pct": round(max(prob_a_elo, 1 - prob_a_elo) * 100),
            "elo_a": round(prediction.get("elo_a", 0)),
            "elo_b": round(prediction.get("elo_b", 0)),
        }

    return {
        "prob_a": prob_a,
        "prob_a_pct": round(prob_a * 100),
        "favored": favored,
        "confidence_pct": round(confidence * 100),
        "coin_flip": abs(prob_a - 0.5) < COIN_FLIP_BAND,
        "rows": rows,
        "elo": elo,
        "players_a": prediction.get("players_a"),
        "players_b": prediction.get("players_b"),
    }


def index(request):
    context = {
        "recent": PredictionLog.objects.all()[:5],
        "demo_nicknames": DEMO_NICKNAMES,  # pool for the client-side shuffle button
        "metrics": get_metrics(),  # track-record table (model vs FACEIT Elo)
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


def random_lineup(request):
    """AJAX endpoint: 10 nicknames of real, currently-ranked CS2 players.

    Used by the "Random real players" button — unlike the demo-pool shuffle,
    this hits the FACEIT leaderboards so it can return any of thousands of
    real, currently-active players.
    """
    try:
        picks = random_ranked_players(2 * TEAM_SIZE)
    except FaceitError as exc:
        return JsonResponse({"error": str(exc)}, status=502)
    return JsonResponse(dict(zip(field_names(), picks)))
