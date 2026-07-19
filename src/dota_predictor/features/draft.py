"""Draft features: hero winrates and bag-of-heroes, computed leak-free.

Hero winrates are maintained in a chronological pass, exactly like Elo in
features/elo.py: for each match we first read the current winrates (that's
the feature), then update them with the match result.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

# Beta(prior_games/2, prior_games/2) smoothing: a hero with no history has
# winrate 0.5, and early results move it slowly.
PRIOR_GAMES = 20.0

# Hero stats decay with a 90-day half-life so the winrate tracks the current
# patch meta instead of averaging over years.
HALF_LIFE_DAYS = 90.0


def build_draft_features(
    matches: pd.DataFrame, drafts: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (matches with draft feature columns, bag-of-heroes frame).

    Draft features per match:
      hero_wr_diff   — mean smoothed hero winrate, radiant minus dire
      hero_xp_diff   — mean log(1 + games in sample) per hero, rad minus dire
    Bag-of-heroes: one column per hero, +1 if picked by radiant, -1 by dire.
    Matches without a known draft get NaN features and a zero bag row.
    """
    picks: dict[int, tuple[list[int], list[int]]] = {}
    for match_id, grp in drafts.groupby("match_id"):
        rad = grp.loc[grp["is_radiant"], "hero_id"].tolist()
        dire = grp.loc[~grp["is_radiant"], "hero_id"].tolist()
        if len(rad) == 5 and len(dire) == 5:
            picks[match_id] = (rad, dire)

    hero_ids = sorted(drafts["hero_id"].unique())
    hero_col = {h: i for i, h in enumerate(hero_ids)}
    bag = np.zeros((len(matches), len(hero_ids)), dtype=np.int8)

    wins: dict[int, float] = defaultdict(float)
    games: dict[int, float] = defaultdict(float)
    last_ts: dict[int, float] = {}

    def decay_to(hero: int, now: float) -> None:
        prev = last_ts.get(hero)
        if prev is not None and now > prev:
            factor = 0.5 ** ((now - prev) / (HALF_LIFE_DAYS * 86400))
            wins[hero] *= factor
            games[hero] *= factor
        last_ts[hero] = now

    def smoothed_wr(hero: int) -> float:
        return (wins[hero] + PRIOR_GAMES / 2) / (games[hero] + PRIOR_GAMES)

    rows: list[dict] = []
    ordered = matches.sort_values("start_time").reset_index(drop=True)
    for idx, row in enumerate(ordered.itertuples()):
        draft = picks.get(row.match_id)
        if draft is None:
            rows.append({"match_id": row.match_id, "hero_wr_diff": np.nan, "hero_xp_diff": np.nan})
            continue
        rad, dire = draft
        for h in rad + dire:
            decay_to(h, row.start_time)

        rad_wr = np.mean([smoothed_wr(h) for h in rad])
        dire_wr = np.mean([smoothed_wr(h) for h in dire])
        rad_xp = np.mean([np.log1p(games[h]) for h in rad])
        dire_xp = np.mean([np.log1p(games[h]) for h in dire])
        rows.append(
            {
                "match_id": row.match_id,
                "hero_wr_diff": rad_wr - dire_wr,
                "hero_xp_diff": rad_xp - dire_xp,
            }
        )
        for h in rad:
            bag[idx, hero_col[h]] = 1
        for h in dire:
            bag[idx, hero_col[h]] = -1

        # Update winrates only after features are recorded.
        outcome = 1.0 if row.radiant_win else 0.0
        for h in rad:
            wins[h] += outcome
            games[h] += 1
        for h in dire:
            wins[h] += 1.0 - outcome
            games[h] += 1

    features = ordered.merge(pd.DataFrame(rows), on="match_id")
    bag_df = pd.DataFrame(bag, columns=[f"hero_{h}" for h in hero_ids])
    bag_df.insert(0, "match_id", ordered["match_id"].to_numpy())
    return features, bag_df
