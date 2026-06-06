"""Build a leakage-safe training dataset from real FACEIT matches.

For each historical match we record:
    - team A (faction1) vs team B (faction2) stat differences, where every
      player's stats come ONLY from matches they finished *before* this match
      (no leakage), and
    - the label: 1 if faction1 won, else 0.

Each player's match history is fetched once and cached, so the same pros being
in many matches doesn't multiply API calls.

Usage:
    python -m src.dataset --seed s1mple-_--- --max-matches 200 --out data/processed/matches.csv
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import pandas as pd

from src.faceit_api import (
    GAME_ID,
    FaceitError,
    _get,
    _session,
    aggregate_stats,  # noqa: F401  (kept importable for tests)
    features_before,
    get_match_stats_items,
    get_player_id,
)
from src.features import FEATURE_KEYS, team_average, team_diff

DEFAULT_OUT = Path("data/processed/matches.csv")
FEATURE_COLS = [f"{key}_diff" for key in FEATURE_KEYS]


def collect_match_ids(seeds: list[str], session, per_seed: int) -> list[str]:
    """Gather candidate match_ids from each seed player's recent history."""
    match_ids: list[str] = []
    seen: set[str] = set()
    for nickname in seeds:
        try:
            pid = get_player_id(nickname, session)
        except FaceitError as exc:
            print(f"  ! seed '{nickname}' skipped: {exc}")
            continue
        payload = _get(
            session,
            f"/players/{pid}/history",
            params={"game": GAME_ID, "offset": 0, "limit": per_seed},
        )
        for item in payload.get("items", []):
            mid = item.get("match_id")
            if mid and mid not in seen:
                seen.add(mid)
                match_ids.append(mid)
        time.sleep(0.05)
    return match_ids


class HistoryCache:
    """Fetch-once cache of each player's per-match stats items."""

    def __init__(self, session, lookback: int = 100):
        self._session = session
        self._lookback = lookback
        self._cache: dict[str, list[dict]] = {}

    def get(self, player_id: str) -> list[dict]:
        if player_id not in self._cache:
            self._cache[player_id] = get_match_stats_items(
                player_id, self._session, limit=self._lookback
            )
            time.sleep(0.05)
        return self._cache[player_id]


def _roster_ids(team: dict) -> list[str]:
    return [p["player_id"] for p in team.get("roster", [])]


def build_row(
    match: dict, cache: HistoryCache, *, window: int, min_history: int
) -> dict | None:
    """Build one labeled training row for a match, or None if not usable."""
    if match.get("status") != "FINISHED":
        return None
    results = match.get("results", {})
    winner = results.get("winner")
    teams = match.get("teams", {})
    finished_at = match.get("finished_at")
    if winner not in ("faction1", "faction2") or finished_at is None:
        return None

    a_ids = _roster_ids(teams.get("faction1", {}))
    b_ids = _roster_ids(teams.get("faction2", {}))
    if len(a_ids) != 5 or len(b_ids) != 5:
        return None  # not a standard 5v5

    a_stats, b_stats = [], []
    for ids, bucket in ((a_ids, a_stats), (b_ids, b_stats)):
        for pid in ids:
            feats = features_before(cache.get(pid), finished_at, window=window)
            if feats is None or feats["matches"] < min_history:
                return None  # a player lacks enough pre-match history
            bucket.append(feats)

    row = team_diff(team_average(a_stats), team_average(b_stats))
    row["label"] = 1 if winner == "faction1" else 0
    row["date"] = pd.to_datetime(finished_at, unit="s")
    row["match_id"] = match.get("match_id")
    return row


def build_dataset(
    seeds: list[str],
    *,
    max_matches: int,
    per_seed: int,
    window: int,
    min_history: int,
) -> pd.DataFrame:
    session = _session()
    print(f"Collecting candidate matches from {len(seeds)} seed(s)...")
    match_ids = collect_match_ids(seeds, session, per_seed)
    print(f"  found {len(match_ids)} candidate matches")

    cache = HistoryCache(session)
    rows: list[dict] = []
    skipped = 0
    for i, mid in enumerate(match_ids[:max_matches], 1):
        try:
            match = _get(session, f"/matches/{mid}")
        except FaceitError as exc:
            print(f"  ! {mid} skipped: {exc}")
            skipped += 1
            continue
        row = build_row(match, cache, window=window, min_history=min_history)
        if row is None:
            skipped += 1
        else:
            rows.append(row)
        if i % 25 == 0:
            print(f"  processed {i}/{min(len(match_ids), max_matches)} | kept {len(rows)}")
        time.sleep(0.05)

    print(f"\nDone: {len(rows)} usable rows, {skipped} skipped.")
    return pd.DataFrame(rows, columns=[*FEATURE_COLS, "label", "date", "match_id"])


def _player_match_ids(pid: str, session, limit: int) -> list[str]:
    """Recent match_ids from a player's match history."""
    payload = _get(
        session,
        f"/players/{pid}/history",
        params={"game": GAME_ID, "offset": 0, "limit": limit},
    )
    return [it["match_id"] for it in payload.get("items", []) if it.get("match_id")]


def snowball_crawl(
    seeds: list[str],
    *,
    target: int,
    per_player: int,
    window: int,
    min_history: int,
    out: Path,
    checkpoint_every: int = 50,
    resume: bool = True,
) -> pd.DataFrame:
    """Grow the dataset by snowball sampling the FACEIT player graph.

    Starting from seed players, we process their matches, then add every player
    seen in those matches to the frontier and repeat (breadth-first). This
    reaches thousands of matches from a handful of seeds.

    The crawl is checkpointed: rows are written to ``out`` and the frontier /
    seen-matches state to ``<out>.state.json`` every ``checkpoint_every``
    matches, so a long run can be resumed (or survive a rate-limit crash).
    """
    session = _session()
    out = Path(out)
    state_path = out.with_suffix(out.suffix + ".state.json")

    rows: list[dict] = []
    seen_matches: set[str] = set()
    queued_players: set[str] = set()
    frontier: deque[str] = deque()

    if resume and out.exists():
        existing = pd.read_csv(out)
        rows = existing.to_dict("records")
        seen_matches.update(existing["match_id"].astype(str))
        print(f"Resuming from {len(rows)} existing rows in {out}")
    if resume and state_path.exists():
        state = json.loads(state_path.read_text())
        seen_matches.update(state.get("seen_matches", []))
        queued_players.update(state.get("queued_players", []))
        frontier.extend(state.get("frontier", []))
        print(f"  restored crawl state: frontier={len(frontier)}, seen={len(seen_matches)}")

    if not frontier:  # fresh crawl: seed the frontier
        for nick in seeds:
            try:
                pid = get_player_id(nick, session)
            except FaceitError as exc:
                print(f"  ! seed '{nick}' skipped: {exc}")
                continue
            if pid not in queued_players:
                queued_players.add(pid)
                frontier.append(pid)

    cache = HistoryCache(session)

    def checkpoint() -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=[*FEATURE_COLS, "label", "date", "match_id"]).to_csv(
            out, index=False
        )
        state_path.write_text(
            json.dumps(
                {
                    "seen_matches": list(seen_matches),
                    "queued_players": list(queued_players),
                    "frontier": list(frontier),
                }
            )
        )

    since_ckpt = 0
    try:
        while frontier and len(rows) < target:
            pid = frontier.popleft()
            try:
                mids = _player_match_ids(pid, session, per_player)
            except FaceitError:
                continue
            for mid in mids:
                if len(rows) >= target:
                    break
                if mid in seen_matches:
                    continue
                seen_matches.add(mid)
                try:
                    match = _get(session, f"/matches/{mid}")
                except FaceitError:
                    continue
                row = build_row(match, cache, window=window, min_history=min_history)
                if row is not None:
                    rows.append(row)
                # expand the frontier with everyone in this match
                teams = match.get("teams", {})
                for faction in ("faction1", "faction2"):
                    for p in teams.get(faction, {}).get("roster", []):
                        npid = p.get("player_id")
                        if npid and npid not in queued_players:
                            queued_players.add(npid)
                            frontier.append(npid)
                since_ckpt += 1
                if since_ckpt >= checkpoint_every:
                    checkpoint()
                    since_ckpt = 0
                    print(
                        f"  kept {len(rows)}/{target} | frontier {len(frontier)} | "
                        f"seen {len(seen_matches)}"
                    )
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nInterrupted — writing final checkpoint...")
    finally:
        checkpoint()

    df = pd.DataFrame(rows, columns=[*FEATURE_COLS, "label", "date", "match_id"])
    print(f"\nDone: {len(df)} rows -> {out}")
    if not df.empty:
        print(f"Label balance (faction1 win rate): {df['label'].mean():.3f}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="append", default=[], help="seed nickname (repeatable)")
    parser.add_argument("--max-matches", type=int, default=200)
    parser.add_argument("--per-seed", type=int, default=100, help="matches pulled per seed")
    parser.add_argument("--window", type=int, default=20, help="prior matches per player")
    parser.add_argument("--min-history", type=int, default=5, help="min prior matches required")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    # Snowball mode: set --target to crawl the player graph up to N rows.
    parser.add_argument("--target", type=int, default=0, help="snowball crawl until N rows")
    parser.add_argument("--per-player", type=int, default=30, help="matches pulled per player (snowball)")
    parser.add_argument("--no-resume", action="store_true", help="ignore existing checkpoint")
    args = parser.parse_args()

    seeds = args.seed or ["s1mple-_---"]

    if args.target > 0:
        snowball_crawl(
            seeds,
            target=args.target,
            per_player=args.per_player,
            window=args.window,
            min_history=args.min_history,
            out=args.out,
            resume=not args.no_resume,
        )
        return

    df = build_dataset(
        seeds,
        max_matches=args.max_matches,
        per_seed=args.per_seed,
        window=args.window,
        min_history=args.min_history,
    )
    if df.empty:
        print("No rows built — try different/more active seeds or lower --min-history.")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows -> {args.out}")
    print(f"Label balance (faction1 win rate): {df['label'].mean():.3f}")


if __name__ == "__main__":
    main()
