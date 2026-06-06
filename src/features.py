"""Build model-ready features from two teams of FACEIT usernames.

A match is represented as the difference between team A's and team B's
average player stats. That keeps the feature vector small and symmetric:
positive values favour team A, negative favour team B.
"""

from __future__ import annotations

from typing import Any

from src.faceit_api import get_player_stats

# Per-player metrics we average and diff. Keys must match get_player_stats output.
FEATURE_KEYS = ("kd", "adr", "hs_percent", "win_rate")


def team_average(player_stats: list[dict]) -> dict[str, float]:
    """Average the feature metrics across a team's per-player stat dicts."""
    n = len(player_stats)
    return {key: sum(p[key] for p in player_stats) / n for key in FEATURE_KEYS}


def team_elo(player_stats: list[dict]) -> float:
    """Average current FACEIT Elo across a team (0 if unavailable)."""
    elos = [p.get("elo") or 0.0 for p in player_stats]
    return sum(elos) / len(elos) if elos else 0.0


def elo_win_probability(elo_a: float, elo_b: float) -> float:
    """Standard Elo expected score: P(team A wins) on the 400-point logistic scale.

    This is FACEIT's own rating used directly as a baseline — no model, just
    the raw skill numbers — to compare against the trained model's output.
    """
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def team_diff(avg_a: dict[str, float], avg_b: dict[str, float]) -> dict[str, float]:
    """Team-A-minus-team-B difference per metric, as {kd_diff, adr_diff, ...}.

    Shared by prediction (build_feature_vector) and training (dataset builder)
    so a row means the same thing in both places.
    """
    return {f"{key}_diff": round(avg_a[key] - avg_b[key], 3) for key in FEATURE_KEYS}


def build_feature_vector(team_a: list[str], team_b: list[str]) -> dict[str, float]:
    """Return team-A-minus-team-B differences for each metric.

    Args:
        team_a, team_b: lists of exactly 5 FACEIT usernames.

    Returns a dict like {"kd_diff": 0.12, "adr_diff": 4.3, ...}, ready to
    feed into a model (e.g. pd.DataFrame([vector])).
    """
    if len(team_a) != 5 or len(team_b) != 5:
        raise ValueError("Each team must have exactly 5 usernames.")

    avg_a = team_average([get_player_stats(name) for name in team_a])
    avg_b = team_average([get_player_stats(name) for name in team_b])
    return team_diff(avg_a, avg_b)


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 11:
        sys.exit("Usage: python -m src.features <5 team-A names> <5 team-B names>")

    a, b = sys.argv[1:6], sys.argv[6:11]
    print(json.dumps(build_feature_vector(a, b), indent=2))
