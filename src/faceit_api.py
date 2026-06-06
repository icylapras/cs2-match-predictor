"""FACEIT Data API client for CS2 player statistics.

Fetches recent match history for a player and aggregates the core
performance metrics used as model features: K/D, ADR, headshot %, win rate.

Requires a FACEIT API key (server-side key) exposed via the
``FACEIT_API_KEY`` environment variable. Get one at
https://developers.faceit.com/ -> Apps -> create app -> API Keys.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests

try:  # optional: auto-load FACEIT_API_KEY from a local .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

API_BASE = "https://open.faceit.com/data/v4"
GAME_ID = "cs2"
DEFAULT_LIMIT = 20
TIMEOUT = 15  # seconds


class FaceitError(RuntimeError):
    """Raised when the FACEIT API returns an error or unexpected data."""


def _session(api_key: str | None = None) -> requests.Session:
    """Build a requests session with the bearer auth header attached."""
    key = api_key or os.environ.get("FACEIT_API_KEY")
    if not key:
        raise FaceitError(
            "No FACEIT API key found. Set the FACEIT_API_KEY environment "
            "variable or pass api_key=... explicitly."
        )
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {key}", "Accept": "application/json"})
    return session


def _get(session: requests.Session, path: str, params: dict[str, Any] | None = None) -> dict:
    """GET a FACEIT endpoint and return the decoded JSON body.

    Retries with backoff on 429 (rate limit), which matters when the dataset
    builder fans out across hundreds of players.
    """
    for attempt in range(4):
        resp = session.get(f"{API_BASE}{path}", params=params, timeout=TIMEOUT)
        if resp.status_code == 429:
            time.sleep(2 ** attempt)  # 1s, 2s, 4s, 8s
            continue
        if resp.status_code == 404:
            raise FaceitError(f"Not found: {path} (params={params})")
        if resp.status_code == 401:
            raise FaceitError("Unauthorized — check that FACEIT_API_KEY is valid.")
        resp.raise_for_status()
        return resp.json()
    raise FaceitError(f"Rate-limited repeatedly on {path}; giving up.")


def get_player_profile(username: str, session: requests.Session) -> dict[str, Any]:
    """Resolve a nickname to its player_id and current FACEIT Elo in one call.

    The /players response carries the live Elo under games.cs2.faceit_elo, so
    we get it for free alongside the id (no extra request).
    """
    data = _get(session, "/players", params={"nickname": username, "game": GAME_ID})
    player_id = data.get("player_id")
    if not player_id:
        raise FaceitError(f"Could not resolve player_id for '{username}'.")
    elo = data.get("games", {}).get(GAME_ID, {}).get("faceit_elo")
    return {"player_id": player_id, "elo": _to_float(elo)}


def get_player_id(username: str, session: requests.Session) -> str:
    """Resolve a FACEIT nickname to its internal player_id."""
    return get_player_profile(username, session)["player_id"]


def _to_float(value: Any) -> float:
    """FACEIT returns stat values as strings; coerce safely to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def match_finished_seconds(item: dict) -> float:
    """Unix seconds a match finished, from a per-match stats item.

    The stats endpoint reports 'Match Finished At' in milliseconds; normalise
    to seconds so it can be compared against /matches finished_at (seconds).
    """
    val = _to_float(item.get("stats", {}).get("Match Finished At"))
    return val / 1000 if val > 1e11 else val


def aggregate_stats(items: list[dict]) -> dict[str, Any]:
    """Aggregate a list of per-match stats items into the feature metrics.

    Shared by the recent (prediction-time) and leakage-safe (training-time)
    paths so both compute features identically.
    """
    total_kills = total_deaths = total_headshots = 0.0
    wins = 0
    adr_sum = 0.0
    for item in items:
        stats = item.get("stats", {})
        total_kills += _to_float(stats.get("Kills"))
        total_deaths += _to_float(stats.get("Deaths"))
        total_headshots += _to_float(stats.get("Headshots"))
        wins += int(_to_float(stats.get("Result")))
        # CS2 exposes "ADR"; older CS:GO payloads only had "K/R Ratio".
        adr_sum += _to_float(stats.get("ADR") or stats.get("Average Damage per Round"))

    n = len(items)
    return {
        "matches": n,
        "kd": round(total_kills / total_deaths, 2) if total_deaths else float(total_kills),
        "adr": round(adr_sum / n, 1),
        "hs_percent": round(total_headshots / total_kills * 100, 1) if total_kills else 0.0,
        "win_rate": round(wins / n * 100, 1),
    }


def get_match_stats_items(
    player_id: str, session: requests.Session, *, limit: int = 100
) -> list[dict]:
    """Return the player's recent per-match stats items (newest first)."""
    payload = _get(
        session,
        f"/players/{player_id}/games/{GAME_ID}/stats",
        params={"offset": 0, "limit": limit},
    )
    return payload.get("items", [])


def features_before(
    items: list[dict], before_ts: float, *, window: int = DEFAULT_LIMIT
) -> dict[str, Any] | None:
    """Aggregate up to ``window`` matches that finished strictly before ``before_ts``.

    This is the leakage guard: a match's own result can never feed its features.
    Returns None if there are no prior matches to aggregate.
    """
    prior = [it for it in items if match_finished_seconds(it) < before_ts]
    if not prior:
        return None
    prior.sort(key=match_finished_seconds, reverse=True)  # newest first
    return aggregate_stats(prior[:window])


def get_player_stats(
    username: str,
    *,
    limit: int = DEFAULT_LIMIT,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Fetch the last ``limit`` CS2 matches for ``username`` and aggregate stats.

    Returns a dict with keys: username, player_id, elo, matches, kd, adr,
    hs_percent, win_rate. Used at prediction time (most recent form).

    Raises FaceitError if the player can't be found or has no recent matches.
    """
    session = _session(api_key)
    profile = get_player_profile(username, session)
    player_id = profile["player_id"]
    items = get_match_stats_items(player_id, session, limit=limit)
    if not items:
        raise FaceitError(f"No recent {GAME_ID} matches found for '{username}'.")

    return {
        "username": username,
        "player_id": player_id,
        "elo": profile["elo"],
        **aggregate_stats(items),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Fetch recent CS2 stats for a FACEIT player.")
    parser.add_argument("username", help="FACEIT nickname")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="matches to fetch")
    args = parser.parse_args()

    print(json.dumps(get_player_stats(args.username, limit=args.limit), indent=2))
