"""Patch-meta and tournament-context features, computed leak-free.

Same chronological discipline as elo.py/draft.py: for each match, features
are read from state accumulated strictly BEFORE the match, then the state
is updated with its result.

Two feature families:

* Patch meta — hero winrates and pick popularity accumulated WITHIN the
  current patch only (stats reset at every patch boundary). Early in a
  patch the features are honestly uninformative and converge as the patch
  meta settles.
* Tournament context — the team's record at this league/tournament before
  the match (first game of an event gets the neutral prior), and how well
  the drafted heroes have performed at this specific tournament so far.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

HERO_PRIOR = 20.0   # pseudo-games pulling hero winrates toward 0.5
EVENT_PRIOR = 4.0   # pseudo-games for a team's record at one tournament
PICK_PRIOR = 10.0   # pseudo-matches for pick-share smoothing

META_COLS = ["patch_wr_diff", "patch_pick_diff", "event_form_diff", "event_hero_wr_diff"]


def build_meta_features(
    matches: pd.DataFrame, drafts: pd.DataFrame, patches: pd.DataFrame
) -> pd.DataFrame:
    """Return matches with META_COLS added (NaN for draft-dependent columns
    when the draft is unknown)."""
    picks: dict[int, tuple[list[int], list[int]]] = {}
    for match_id, grp in drafts.groupby("match_id"):
        rad = grp.loc[grp["is_radiant"], "hero_id"].tolist()
        dire = grp.loc[~grp["is_radiant"], "hero_id"].tolist()
        if len(rad) == 5 and len(dire) == 5:
            picks[match_id] = (rad, dire)

    patch_starts = patches["start_time"].to_numpy()

    # Per-patch hero state, reset at every patch boundary.
    current_patch = -1
    hero_wins: dict[int, float] = defaultdict(float)
    hero_games: dict[int, float] = defaultdict(float)
    patch_matches = 0

    # Per-tournament state, keyed by (leagueid, ...).
    event_team: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0])  # wins, games
    event_hero: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0])

    def hero_wr(h: int) -> float:
        return (hero_wins[h] + HERO_PRIOR / 2) / (hero_games[h] + HERO_PRIOR)

    def pick_share(h: int) -> float:
        return (hero_games[h] + 1.0) / (patch_matches + PICK_PRIOR)

    def event_form(league: int, team: int) -> float:
        wins, games = event_team[(league, team)]
        return (wins + EVENT_PRIOR / 2) / (games + EVENT_PRIOR)

    def event_hero_wr(league: int, h: int) -> float:
        wins, games = event_hero[(league, h)]
        return (wins + PICK_PRIOR / 2) / (games + PICK_PRIOR)

    rows: list[dict] = []
    ordered = matches.sort_values("start_time").reset_index(drop=True)
    for row in ordered.itertuples():
        patch_idx = int(np.searchsorted(patch_starts, row.start_time, side="right")) - 1
        if patch_idx != current_patch:
            current_patch = patch_idx
            hero_wins.clear()
            hero_games.clear()
            patch_matches = 0

        league = row.leagueid
        rad_team, dire_team = row.radiant_team_id, row.dire_team_id
        feat = {
            "match_id": row.match_id,
            "event_form_diff": event_form(league, rad_team) - event_form(league, dire_team),
            "patch_wr_diff": np.nan,
            "patch_pick_diff": np.nan,
            "event_hero_wr_diff": np.nan,
        }
        draft = picks.get(row.match_id)
        if draft is not None:
            rad, dire = draft
            feat["patch_wr_diff"] = np.mean([hero_wr(h) for h in rad]) - np.mean(
                [hero_wr(h) for h in dire]
            )
            feat["patch_pick_diff"] = np.mean([pick_share(h) for h in rad]) - np.mean(
                [pick_share(h) for h in dire]
            )
            feat["event_hero_wr_diff"] = np.mean(
                [event_hero_wr(league, h) for h in rad]
            ) - np.mean([event_hero_wr(league, h) for h in dire])
        rows.append(feat)

        # Update state only after features are recorded.
        outcome = 1.0 if row.radiant_win else 0.0
        event_team[(league, rad_team)][0] += outcome
        event_team[(league, rad_team)][1] += 1
        event_team[(league, dire_team)][0] += 1.0 - outcome
        event_team[(league, dire_team)][1] += 1
        if draft is not None:
            patch_matches += 1
            for h, win in [(h, outcome) for h in rad] + [(h, 1.0 - outcome) for h in dire]:
                hero_wins[h] += win
                hero_games[h] += 1
                event_hero[(league, h)][0] += win
                event_hero[(league, h)][1] += 1

    return matches.merge(pd.DataFrame(rows), on="match_id")
