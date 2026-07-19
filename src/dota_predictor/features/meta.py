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
USAGE_PRIOR = 5.0   # pseudo-matches for a team's hero-usage share

META_COLS = ["patch_wr_diff", "patch_pick_diff", "event_form_diff", "event_hero_wr_diff"]
BAN_COLS = ["ban_wr_diff", "ban_pick_diff", "targeted_ban_diff"]


def build_meta_features(
    matches: pd.DataFrame,
    drafts: pd.DataFrame,
    patches: pd.DataFrame,
    bans: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return matches with META_COLS added (NaN for draft-dependent columns
    when the draft is unknown)."""
    picks: dict[int, tuple[list[int], list[int]]] = {}
    for match_id, grp in drafts.groupby("match_id"):
        rad = grp.loc[grp["is_radiant"], "hero_id"].tolist()
        dire = grp.loc[~grp["is_radiant"], "hero_id"].tolist()
        if len(rad) == 5 and len(dire) == 5:
            picks[match_id] = (rad, dire)

    ban_map: dict[int, tuple[list[int], list[int]]] = {}
    if bans is not None and not bans.empty:
        for match_id, grp in bans.groupby("match_id"):
            ban_map[match_id] = (
                grp.loc[grp["banned_by_radiant"], "hero_id"].tolist(),
                grp.loc[~grp["banned_by_radiant"], "hero_id"].tolist(),
            )

    patch_starts = patches["start_time"].to_numpy()

    # Per-patch hero state, reset at every patch boundary.
    current_patch = -1
    hero_wins: dict[int, float] = defaultdict(float)
    hero_games: dict[int, float] = defaultdict(float)
    patch_matches = 0

    # Per-tournament state, keyed by (leagueid, ...).
    event_team: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0])  # wins, games
    event_hero: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0])

    # Per-patch team hero usage (for targeted-ban detection), reset with the patch.
    team_hero: dict[tuple, float] = defaultdict(float)
    team_games: dict[int, float] = defaultdict(float)

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

    def usage_share(team: int, h: int) -> float:
        """How often the team has played this hero in the current patch."""
        return (team_hero[(team, h)] + 0.5) / (team_games[team] + USAGE_PRIOR)

    rows: list[dict] = []
    ordered = matches.sort_values("start_time").reset_index(drop=True)
    for row in ordered.itertuples():
        patch_idx = int(np.searchsorted(patch_starts, row.start_time, side="right")) - 1
        if patch_idx != current_patch:
            current_patch = patch_idx
            hero_wins.clear()
            hero_games.clear()
            team_hero.clear()
            team_games.clear()
            patch_matches = 0

        league = row.leagueid
        rad_team, dire_team = row.radiant_team_id, row.dire_team_id
        feat = {
            "match_id": row.match_id,
            "event_form_diff": event_form(league, rad_team) - event_form(league, dire_team),
            "patch_wr_diff": np.nan,
            "patch_pick_diff": np.nan,
            "event_hero_wr_diff": np.nan,
            "ban_wr_diff": np.nan,
            "ban_pick_diff": np.nan,
            "targeted_ban_diff": np.nan,
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
        match_bans = ban_map.get(row.match_id)
        if match_bans is not None and match_bans[0] and match_bans[1]:
            rad_bans, dire_bans = match_bans
            feat["ban_wr_diff"] = np.mean([hero_wr(h) for h in rad_bans]) - np.mean(
                [hero_wr(h) for h in dire_bans]
            )
            feat["ban_pick_diff"] = np.mean([pick_share(h) for h in rad_bans]) - np.mean(
                [pick_share(h) for h in dire_bans]
            )
            # How much the opponent's bans target this team's comfort heroes:
            # positive diff = radiant is the more feared side.
            feared_rad = np.mean([usage_share(rad_team, h) for h in dire_bans])
            feared_dire = np.mean([usage_share(dire_team, h) for h in rad_bans])
            feat["targeted_ban_diff"] = feared_rad - feared_dire
        rows.append(feat)

        # Update state only after features are recorded.
        outcome = 1.0 if row.radiant_win else 0.0
        event_team[(league, rad_team)][0] += outcome
        event_team[(league, rad_team)][1] += 1
        event_team[(league, dire_team)][0] += 1.0 - outcome
        event_team[(league, dire_team)][1] += 1
        if draft is not None:
            patch_matches += 1
            team_games[rad_team] += 1
            team_games[dire_team] += 1
            for h in rad:
                team_hero[(rad_team, h)] += 1
            for h in dire:
                team_hero[(dire_team, h)] += 1
            for h, win in [(h, outcome) for h in rad] + [(h, 1.0 - outcome) for h in dire]:
                hero_wins[h] += win
                hero_games[h] += 1
                event_hero[(league, h)][0] += win
                event_hero[(league, h)][1] += 1

    return matches.merge(pd.DataFrame(rows), on="match_id")
