"""From-scratch, leakage-safe Elo ratings as model features.

The raw stat features (kd/adr/hs%/win_rate) suffer from *strength-of-schedule*
bias: they ignore *who* a player faced. FACEIT matchmaking is balanced, so it
compresses everyone's K/D toward ~1.0 and win-rate toward ~50% regardless of
absolute skill. That is pathological for this tool, whose use case is
unbalanced cross-bracket matchups.

This module fixes that by replaying every match in the dataset in chronological
order and maintaining a per-player Elo rating. For each match we record the
team-average Elo *before* the match is played, so a match's own result can never
leak into its own features (the same leakage guard used everywhere else).

Two feature families come out of the replay, both as team-A-minus-team-B diffs:
    - ``elo_diff``    — the replayed skill rating (opponent-adjusted by design).
    - ``oppelo_diff`` — average strength of the opponents each player has faced
      so far (an explicit strength-of-schedule signal).

Pipeline:
    1. ``enrich_rosters`` — re-fetch /matches/{id} for every match_id in the
       processed dataset to recover both 5-man rosters. Cached to disk and
       resumable (skips match_ids already saved).
    2. ``replay`` — chronological Elo + opponent-strength replay over those
       rosters, joined to the dataset for dates/labels.

Usage:
    python -m src.elo --data data/processed/matches.csv
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.faceit_api import FaceitError, _get, _session

DEFAULT_DATA = Path("data/processed/matches.csv")
DEFAULT_ROSTERS = Path("data/processed/rosters.csv")
DEFAULT_FEATURES = Path("data/processed/elo_features.csv")
DEFAULT_RATINGS = Path("data/processed/player_elo.csv")
DEFAULT_FACEIT_ELO = Path("data/processed/player_faceit_elo.csv")

BASE_ELO = 1500.0  # everyone starts here; ratings diverge as matches are played
K_FACTOR = 32.0  # Elo update step (standard chess value)


def _roster_ids(team: dict) -> list[str]:
    return [p["player_id"] for p in team.get("roster", [])]


def enrich_rosters(
    match_ids: list[str], *, out: Path = DEFAULT_ROSTERS, resume: bool = True
) -> pd.DataFrame:
    """Fetch both rosters for each match_id, caching to ``out`` (resumable).

    Returns a frame with columns: match_id, a_ids, b_ids (player_ids
    space-joined). Only standard 5v5 matches with a clear winner are kept,
    matching the dataset builder's own filter so the join lines up.
    """
    out = Path(out)
    have: dict[str, dict] = {}
    if resume and out.exists():
        existing = pd.read_csv(out, dtype=str)
        have = {rec["match_id"]: rec for rec in existing.to_dict("records")}
        print(f"Resuming roster cache: {len(have)} already fetched")

    session = _session()
    todo = [m for m in match_ids if m not in have]
    print(f"Fetching rosters for {len(todo)} new matches...")

    rows = list(have.values())
    for i, mid in enumerate(todo, 1):
        try:
            match = _get(session, f"/matches/{mid}")
        except FaceitError:
            continue
        teams = match.get("teams", {})
        a_ids = _roster_ids(teams.get("faction1", {}))
        b_ids = _roster_ids(teams.get("faction2", {}))
        if len(a_ids) != 5 or len(b_ids) != 5:
            continue
        rows.append({"match_id": mid, "a_ids": " ".join(a_ids), "b_ids": " ".join(b_ids)})
        if i % 50 == 0:
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"  fetched {i}/{len(todo)} | cached {len(rows)}")
        time.sleep(0.05)

    df = pd.DataFrame(rows, columns=["match_id", "a_ids", "b_ids"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Rosters cached: {len(df)} -> {out}")
    return df


def unique_players(rosters: pd.DataFrame) -> list[str]:
    """Every distinct player_id appearing in either roster across all matches."""
    ids: set[str] = set()
    for col in ("a_ids", "b_ids"):
        for cell in rosters[col]:
            ids.update(str(cell).split())
    return sorted(ids)


def snapshot_faceit_elo(
    player_ids: list[str], *, out: Path = DEFAULT_FACEIT_ELO, resume: bool = True
) -> dict[str, float]:
    """Fetch each player's *current* FACEIT CS2 Elo via /players/{id}.

    FACEIT does not expose point-in-time Elo, so this is the live rating. Because
    the dataset only spans ~48 days, current Elo is a low-drift proxy for each
    player's Elo at match time (the leakage is small, and smallest on the newest
    matches that form the test set). Cached to ``out`` and resumable; players we
    can't resolve (deleted / no CS2 data) are recorded with a blank elo so we
    don't re-request them.
    """
    out = Path(out)
    elo: dict[str, float] = {}
    done: set[str] = set()
    if resume and out.exists():
        cached = pd.read_csv(out, dtype={"player_id": str})
        for _, r in cached.iterrows():
            done.add(r["player_id"])
            if pd.notna(r["elo"]):
                elo[r["player_id"]] = float(r["elo"])
        print(f"Resuming FACEIT-Elo snapshot: {len(done)} already fetched")

    session = _session()
    todo = [p for p in player_ids if p not in done]
    print(f"Fetching current FACEIT Elo for {len(todo)} new players...")

    records = [{"player_id": p, "elo": elo.get(p, "")} for p in done]
    for i, pid in enumerate(todo, 1):
        value: float | str = ""
        try:
            data = _get(session, f"/players/{pid}")
            raw = data.get("games", {}).get("cs2", {}).get("faceit_elo")
            if raw is not None:
                value = float(raw)
                elo[pid] = value
        except FaceitError:
            pass  # deleted account / no cs2 — leave blank, still cache as attempted
        records.append({"player_id": pid, "elo": value})
        if i % 100 == 0:
            pd.DataFrame(records).to_csv(out, index=False)
            print(f"  fetched {i}/{len(todo)} | with elo {len(elo)}")
        time.sleep(0.05)

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"FACEIT Elo snapshot: {len(elo)}/{len(records)} players have an Elo -> {out}")
    return elo


def faceit_features(
    dataset: pd.DataFrame, rosters: pd.DataFrame, elo: dict[str, float]
) -> pd.DataFrame:
    """Per-match current-FACEIT-Elo team features (the user-facing skill signal).

    Missing player Elos are filled with the median of all known Elos so a single
    unresolved account doesn't drop a whole match. Returns match_id, faceit_elo_a,
    faceit_elo_b, faceit_elo_diff.
    """
    merged = dataset[["match_id"]].merge(rosters, on="match_id", how="inner")
    fill = float(pd.Series(list(elo.values())).median()) if elo else 1000.0

    def team_elo(ids: list[str]) -> float:
        vals = [elo.get(p, fill) for p in ids]
        return sum(vals) / len(vals)

    rows = []
    for _, row in merged.iterrows():
        ea = team_elo(row["a_ids"].split())
        eb = team_elo(row["b_ids"].split())
        rows.append(
            {
                "match_id": row["match_id"],
                "faceit_elo_a": ea,
                "faceit_elo_b": eb,
                "faceit_elo_diff": ea - eb,
            }
        )
    return pd.DataFrame(rows)


def stacking_features(
    dataset: pd.DataFrame, rosters: pd.DataFrame, *, date_col: str = "date"
) -> pd.DataFrame:
    """Leakage-safe teammate 'stacking' signal — orthogonal to FACEIT Elo.

    A single Elo number can't tell a coordinated 5-stack from five strangers at
    the same rating. For each team we measure how much its players have played
    together *before* this match: the average, over the team's 10 player-pairs,
    of the number of prior matches that pair were teammates. Counts use only
    matches earlier than the current one (chronological), so no leakage.

    Returns match_id, stack_a, stack_b, stack_diff (A - B).
    """
    from collections import defaultdict
    from itertools import combinations

    merged = dataset[["match_id", date_col]].merge(rosters, on="match_id", how="inner")
    merged[date_col] = pd.to_datetime(merged[date_col])
    merged = merged.sort_values([date_col, "match_id"]).reset_index(drop=True)

    pair_count: dict[tuple[str, str], int] = defaultdict(int)

    def team_stack(ids: list[str]) -> float:
        pairs = list(combinations(sorted(ids), 2))
        return sum(pair_count[p] for p in pairs) / len(pairs) if pairs else 0.0

    rows = []
    for _, r in merged.iterrows():
        a, b = r["a_ids"].split(), r["b_ids"].split()
        sa, sb = team_stack(a), team_stack(b)
        rows.append({"match_id": r["match_id"], "stack_a": sa, "stack_b": sb, "stack_diff": sa - sb})
        for team in (a, b):  # update AFTER recording the feature (leakage-safe)
            for pair in combinations(sorted(team), 2):
                pair_count[pair] += 1
    return pd.DataFrame(rows)


def _expected_a(elo_a: float, elo_b: float) -> float:
    """Standard Elo expected score for team A on the 400-point logistic scale."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def replay(
    dataset: pd.DataFrame,
    rosters: pd.DataFrame,
    *,
    base: float = BASE_ELO,
    k: float = K_FACTOR,
    date_col: str = "date",
    label_col: str = "label",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Chronologically replay Elo + opponent strength over the matches.

    For every match, in time order, we:
      1. read each team's *pre-match* average Elo and average opponent-Elo-faced
         (these become the leakage-safe features for that match), then
      2. update both teams' player ratings from the result, and record that
         each team's players just faced the other team's pre-match strength.

    Returns ``(features, final_ratings)`` where features is indexed/keyed by
    match_id with columns elo_a, elo_b, elo_diff, oppelo_a, oppelo_b,
    oppelo_diff, and final_ratings maps player_id -> latest Elo (for inference).
    """
    merged = dataset.merge(rosters, on="match_id", how="inner")
    merged[date_col] = pd.to_datetime(merged[date_col])
    # Sort by (date, match_id) so the replay order is deterministic on ties.
    merged = merged.sort_values([date_col, "match_id"]).reset_index(drop=True)

    ratings: dict[str, float] = defaultdict(lambda: base)
    sched_sum: dict[str, float] = defaultdict(float)  # sum of opponent Elos faced
    sched_n: dict[str, int] = defaultdict(int)  # number of opponents faced

    def team_opp_elo(ids: list[str]) -> float:
        vals = [sched_sum[p] / sched_n[p] for p in ids if sched_n[p] > 0]
        return sum(vals) / len(vals) if vals else base

    out_rows: list[dict] = []
    for _, row in merged.iterrows():
        a_ids = row["a_ids"].split()
        b_ids = row["b_ids"].split()

        elo_a = sum(ratings[p] for p in a_ids) / len(a_ids)
        elo_b = sum(ratings[p] for p in b_ids) / len(b_ids)
        oppelo_a = team_opp_elo(a_ids)
        oppelo_b = team_opp_elo(b_ids)

        out_rows.append(
            {
                "match_id": row["match_id"],
                "elo_a": elo_a,
                "elo_b": elo_b,
                "elo_diff": elo_a - elo_b,
                "oppelo_a": oppelo_a,
                "oppelo_b": oppelo_b,
                "oppelo_diff": oppelo_a - oppelo_b,
            }
        )

        # --- update after recording features (so no self-leakage) ---
        exp_a = _expected_a(elo_a, elo_b)
        actual_a = float(row[label_col])  # 1 if faction1 (team A) won
        delta = k * (actual_a - exp_a)
        for p in a_ids:
            ratings[p] += delta
        for p in b_ids:
            ratings[p] -= delta  # team B's expected = 1 - exp_a, actual = 1 - actual_a
        # each player just faced the opposing team's pre-match average Elo
        for p in a_ids:
            sched_sum[p] += elo_b
            sched_n[p] += 1
        for p in b_ids:
            sched_sum[p] += elo_a
            sched_n[p] += 1

    features = pd.DataFrame(out_rows)
    return features, dict(ratings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--rosters", type=Path, default=DEFAULT_ROSTERS)
    parser.add_argument("--features-out", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--ratings-out", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument("--faceit-elo-out", type=Path, default=DEFAULT_FACEIT_ELO)
    parser.add_argument("--k", type=float, default=K_FACTOR)
    parser.add_argument("--base", type=float, default=BASE_ELO)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    dataset = pd.read_csv(args.data)
    rosters = enrich_rosters(
        dataset["match_id"].astype(str).tolist(),
        out=args.rosters,
        resume=not args.no_resume,
    )

    # (1) current FACEIT Elo per match (the user-facing skill feature)
    elo_map = snapshot_faceit_elo(
        unique_players(rosters), out=args.faceit_elo_out, resume=not args.no_resume
    )
    faceit = faceit_features(dataset, rosters, elo_map)

    # (2) leakage-free replayed Elo + opponent strength (rigor cross-check)
    replayed, ratings = replay(dataset, rosters, base=args.base, k=args.k)

    features = faceit.merge(replayed, on="match_id", how="inner")
    args.features_out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.features_out, index=False)
    print(f"\nElo features: {len(features)} rows -> {args.features_out}")

    ratings_df = pd.DataFrame(
        sorted(ratings.items(), key=lambda kv: kv[1], reverse=True),
        columns=["player_id", "elo"],
    )
    ratings_df.to_csv(args.ratings_out, index=False)
    print(f"Final ratings: {len(ratings_df)} players -> {args.ratings_out}")
    print(f"Elo spread: min={ratings_df['elo'].min():.0f} "
          f"max={ratings_df['elo'].max():.0f} std={ratings_df['elo'].std():.0f}")


if __name__ == "__main__":
    main()
