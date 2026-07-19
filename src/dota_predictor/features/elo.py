"""Pre-match features computed chronologically: Elo ratings and recent form.

All features use only information available BEFORE the match starts —
ratings are read first, then updated with the match result. This is what
makes the dataset leak-free.
"""

from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd

INITIAL_ELO = 1500.0
K_FACTOR = 32.0
FORM_WINDOW = 10  # matches


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Walk matches in chronological order, emitting pre-match features.

    Expects columns: match_id, start_time, radiant_team_id, dire_team_id,
    radiant_win. Returns the input plus feature columns.
    """
    elo: dict[int, float] = defaultdict(lambda: INITIAL_ELO)
    recent: dict[int, deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    games_played: dict[int, int] = defaultdict(int)

    feats: list[dict] = []
    for row in matches.sort_values("start_time").itertuples():
        rad, dire = row.radiant_team_id, row.dire_team_id
        rad_elo, dire_elo = elo[rad], elo[dire]

        # Form = win rate over the last FORM_WINDOW matches, shrunk toward
        # 0.5 when a team has little history so new teams aren't extreme.
        def form(team: int) -> float:
            hist = recent[team]
            return (sum(hist) + 0.5 * (FORM_WINDOW - len(hist))) / FORM_WINDOW

        feats.append(
            {
                "match_id": row.match_id,
                "elo_diff": rad_elo - dire_elo,
                "form_diff": form(rad) - form(dire),
                "rad_games": games_played[rad],
                "dire_games": games_played[dire],
            }
        )

        # Update state with the match outcome (after features are recorded).
        exp_rad = expected_score(rad_elo, dire_elo)
        outcome = 1.0 if row.radiant_win else 0.0
        elo[rad] = rad_elo + K_FACTOR * (outcome - exp_rad)
        elo[dire] = dire_elo + K_FACTOR * ((1.0 - outcome) - (1.0 - exp_rad))
        recent[rad].append(outcome)
        recent[dire].append(1.0 - outcome)
        games_played[rad] += 1
        games_played[dire] += 1

    features = pd.DataFrame(feats)
    return matches.merge(features, on="match_id").sort_values("start_time").reset_index(drop=True)


def current_ratings(matches: pd.DataFrame) -> pd.DataFrame:
    """Return the latest Elo rating per team, for inspection/prediction."""
    elo: dict[int, float] = defaultdict(lambda: INITIAL_ELO)
    names: dict[int, str] = {}
    for row in matches.sort_values("start_time").itertuples():
        rad, dire = row.radiant_team_id, row.dire_team_id
        exp_rad = expected_score(elo[rad], elo[dire])
        outcome = 1.0 if row.radiant_win else 0.0
        elo[rad] += K_FACTOR * (outcome - exp_rad)
        elo[dire] += K_FACTOR * ((1.0 - outcome) - (1.0 - exp_rad))
        names[rad] = getattr(row, "radiant_name", None) or str(rad)
        names[dire] = getattr(row, "dire_name", None) or str(dire)
    return (
        pd.DataFrame(
            {"team_id": list(elo), "team": [names[t] for t in elo], "elo": list(elo.values())}
        )
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )
