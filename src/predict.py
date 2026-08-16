"""Predict the winner of a CS2 matchup from two teams of 5 FACEIT nicknames.

Loads the trained model and each player's *recent form*, builds the same
team-difference features used in training, and prints which team is favored
and by how much.

Usage (comma-separated so nicknames starting with '-' are safe):
    python -m src.predict \
        --team-a 'name1,name2,name3,name4,name5' \
        --team-b 'name6,name7,name8,name9,name10'

Nicknames must be exact, currently-existing FACEIT accounts — the API 404s on
anything else, so avoid hardcoding examples that can go inactive.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.faceit_api import FaceitError, get_player_stats
from src.features import (
    FEATURE_KEYS,
    MODEL_FEATURE_COLS,
    elo_win_probability,
    match_feature_row,
    team_average,
    team_elo,
)

DEFAULT_MODEL = Path("data/processed/best_model.joblib")


def _fetch_team(names: list[str]) -> tuple[list[dict], list[tuple[str, str]]]:
    """Fetch recent stats for each name; return (stats, failures)."""
    stats, failures = [], []
    for name in names:
        try:
            stats.append(get_player_stats(name))
        except FaceitError as exc:
            failures.append((name, str(exc)))
    return stats, failures


def load_model(model_path: Path = DEFAULT_MODEL):
    """Load the trained model from disk. Cheap to call, but worth caching in a
    long-running server (see predictor/services.py) instead of per request."""
    return joblib.load(model_path)


def predict_from_stats(a_stats: list[dict], b_stats: list[dict], model) -> dict:
    """Run the model on already-fetched team stats.

    Split out from predict_match so callers that cache player stats and the
    model (e.g. the Django app) can reuse the exact same scoring path.
    """
    avg_a, avg_b = team_average(a_stats), team_average(b_stats)
    elo_a, elo_b = team_elo(a_stats), team_elo(b_stats)

    # The model reasons from BOTH the FACEIT-Elo gap and the recent-form stat
    # diffs (see src.features.match_feature_row / MODEL_FEATURE_COLS).
    vector = match_feature_row(a_stats, b_stats)
    x = pd.DataFrame([vector])[MODEL_FEATURE_COLS]
    prob_a = float(model.predict_proba(x)[0, 1])

    # FACEIT Elo baseline: a model-free prediction from the raw ratings, for
    # side-by-side comparison. None if Elo is missing for either team.
    prob_a_elo = elo_win_probability(elo_a, elo_b) if (elo_a and elo_b) else None

    return {
        "prob_a": prob_a,
        "vector": vector,
        "avg_a": avg_a,
        "avg_b": avg_b,
        "prob_a_elo": prob_a_elo,
        "elo_a": elo_a,
        "elo_b": elo_b,
    }


def predict_match(
    team_a: list[str], team_b: list[str], *, model_path: Path = DEFAULT_MODEL
) -> dict:
    """Return prediction details for a matchup, fetching each player's recent form."""
    a_stats, a_fail = _fetch_team(team_a)
    b_stats, b_fail = _fetch_team(team_b)
    failures = a_fail + b_fail
    if failures:
        lines = "\n".join(f"  - {n}: {e}" for n, e in failures)
        raise FaceitError(f"Could not fetch these players:\n{lines}")

    return predict_from_stats(a_stats, b_stats, load_model(model_path))


def _split(arg: str) -> list[str]:
    return [n.strip() for n in arg.split(",") if n.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-a", required=True, help="5 nicknames, comma-separated")
    parser.add_argument("--team-b", required=True, help="5 nicknames, comma-separated")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    team_a, team_b = _split(args.team_a), _split(args.team_b)
    if len(team_a) != 5 or len(team_b) != 5:
        parser.error("Each team needs exactly 5 comma-separated nicknames.")

    result = predict_match(team_a, team_b, model_path=args.model)
    prob_a = result["prob_a"]
    favored, conf = ("Team A", prob_a) if prob_a >= 0.5 else ("Team B", 1 - prob_a)

    print("Team A:", ", ".join(team_a))
    print("Team B:", ", ".join(team_b))
    print("\nTeam averages (recent form):")
    print(f"  {'metric':12s} {'Team A':>8s} {'Team B':>8s}")
    for key in FEATURE_KEYS:
        print(f"  {key:12s} {result['avg_a'][key]:8.2f} {result['avg_b'][key]:8.2f}")
    print(f"\n>>> {favored} favored — {conf * 100:.0f}% confidence")
    if abs(prob_a - 0.5) < 0.05:
        print("    (essentially a coin flip — these teams look evenly matched)")


if __name__ == "__main__":
    main()
