"""Service layer between the Django views and the ML package in ``src/``.

This is where the production-relevant engineering lives:

* The trained model is loaded **once** and reused (not re-read per request).
* Each player's FACEIT stats are **cached**, so repeat predictions don't
  re-hit the rate-limited FACEIT API. Caching the slow external call is the
  single most important scaling decision for this app.

The actual scoring is delegated to ``src.predict.predict_from_stats`` so the
web path and the CLI path compute predictions identically.
"""

from __future__ import annotations

from django.conf import settings
from django.core.cache import cache

from src.faceit_api import FaceitError, get_player_stats
from src.predict import load_model, predict_from_stats

MODEL_PATH = settings.BASE_DIR / "data" / "processed" / "best_model.joblib"

_model = None  # module-level singleton, populated on first use


def get_model():
    """Return the trained model, loading it from disk only once."""
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model


def _cache_key(name: str) -> str:
    return f"faceit:stats:{name.strip().lower()}"


def cached_player_stats(name: str) -> dict:
    """Return a player's recent stats, using the cache to avoid repeat API calls.

    Raises FaceitError (propagated from get_player_stats) on a cache miss when
    the player can't be fetched.
    """
    key = _cache_key(name)
    stats = cache.get(key)
    if stats is None:
        stats = get_player_stats(name)
        cache.set(key, stats, settings.PLAYER_STATS_TTL)
    return stats


def run_prediction(team_a: list[str], team_b: list[str]) -> dict:
    """Fetch (cached) stats for all 10 players and score the matchup.

    Returns the dict from ``predict_from_stats`` (prob_a, vector, avg_a, avg_b).
    Raises FaceitError listing every nickname that could not be resolved.
    """
    a_stats: list[dict] = []
    b_stats: list[dict] = []
    failures: list[str] = []
    for names, bucket in ((team_a, a_stats), (team_b, b_stats)):
        for name in names:
            try:
                bucket.append(cached_player_stats(name))
            except FaceitError as exc:
                failures.append(f"{name}: {exc}")

    if failures:
        raise FaceitError("Could not fetch these players:\n" + "\n".join(failures))

    return predict_from_stats(a_stats, b_stats, get_model())
