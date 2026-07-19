"""Roster features: lineup stability, five-stack synergy, player quality.

Targets the biggest blind spot of team-level features — substitutions and
stand-ins. Same leak-free chronological discipline as the other feature
modules: read state, emit features, then update with the match result.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

PLAYER_PRIOR = 10.0  # pseudo-games pulling a player's winrate toward 0.5

ROSTER_COLS = ["roster_stability_diff", "lineup_games_diff", "player_wr_diff"]


def build_roster_features(matches: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Return matches with ROSTER_COLS added (NaN when a roster is unknown).

    Features (radiant minus dire):
      roster_stability_diff — share of today's five that also played the
        team's previous match (1.0 = same lineup, lower = stand-ins)
      lineup_games_diff     — log1p(games this exact five-stack played together)
      player_wr_diff        — mean smoothed individual winrate of the five
    """
    rosters: dict[int, tuple[frozenset, frozenset]] = {}
    for match_id, grp in players.groupby("match_id"):
        rad = frozenset(grp.loc[grp["is_radiant"], "account_id"])
        dire = frozenset(grp.loc[~grp["is_radiant"], "account_id"])
        if len(rad) == 5 and len(dire) == 5:
            rosters[match_id] = (rad, dire)

    last_lineup: dict[int, frozenset] = {}
    lineup_games: dict[frozenset, int] = defaultdict(int)
    player_wins: dict[int, float] = defaultdict(float)
    player_games: dict[int, float] = defaultdict(float)

    def stability(team: int, lineup: frozenset) -> float:
        prev = last_lineup.get(team)
        return np.nan if prev is None else len(lineup & prev) / 5.0

    def player_wr(lineup: frozenset) -> float:
        return float(
            np.mean(
                [
                    (player_wins[p] + PLAYER_PRIOR / 2) / (player_games[p] + PLAYER_PRIOR)
                    for p in lineup
                ]
            )
        )

    rows: list[dict] = []
    ordered = matches.sort_values("start_time").reset_index(drop=True)
    for row in ordered.itertuples():
        roster = rosters.get(row.match_id)
        feat = {
            "match_id": row.match_id,
            "roster_stability_diff": np.nan,
            "lineup_games_diff": np.nan,
            "player_wr_diff": np.nan,
        }
        if roster is not None:
            rad, dire = roster
            stab_rad = stability(row.radiant_team_id, rad)
            stab_dire = stability(row.dire_team_id, dire)
            if not (np.isnan(stab_rad) or np.isnan(stab_dire)):
                feat["roster_stability_diff"] = stab_rad - stab_dire
            feat["lineup_games_diff"] = float(
                np.log1p(lineup_games[rad]) - np.log1p(lineup_games[dire])
            )
            feat["player_wr_diff"] = player_wr(rad) - player_wr(dire)
        rows.append(feat)

        if roster is not None:
            rad, dire = roster
            outcome = 1.0 if row.radiant_win else 0.0
            lineup_games[rad] += 1
            lineup_games[dire] += 1
            last_lineup[row.radiant_team_id] = rad
            last_lineup[row.dire_team_id] = dire
            for p in rad:
                player_wins[p] += outcome
                player_games[p] += 1
            for p in dire:
                player_wins[p] += 1.0 - outcome
                player_games[p] += 1

    return matches.merge(pd.DataFrame(rows), on="match_id")
