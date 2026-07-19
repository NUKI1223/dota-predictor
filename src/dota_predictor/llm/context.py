"""Data layer for the preview agent: team stats from the cached match data.

Everything is computed from the local parquet caches (populated by
`python -m dota_predictor.pipeline`) — no network calls here.
"""

from __future__ import annotations

import difflib
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from dota_predictor.features.elo import INITIAL_ELO, K_FACTOR, expected_score
from dota_predictor.features.elo import build_features
from dota_predictor.models.baseline import FEATURE_COLS

FORM_WINDOW = 10


class ContextStore:
    """Team stats, head-to-head history and a win-probability model."""

    def __init__(self, data_dir: Path, tiers: tuple[str, ...] = ("premium", "professional")):
        self.matches = pd.read_parquet(data_dir / "raw" / "pro_matches.parquet")
        leagues = pd.read_parquet(data_dir / "raw" / "leagues.parquet")
        self.matches = self.matches.merge(
            leagues[["leagueid", "tier"]], on="leagueid", how="left"
        ).sort_values("start_time")

        self._build_team_state()
        self._fit_model(tiers)

    def _build_team_state(self) -> None:
        elo: dict[int, float] = defaultdict(lambda: INITIAL_ELO)
        recent: dict[int, deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
        games: dict[int, int] = defaultdict(int)
        names: dict[int, str] = {}
        for row in self.matches.itertuples():
            rad, dire = row.radiant_team_id, row.dire_team_id
            exp = expected_score(elo[rad], elo[dire])
            outcome = 1.0 if row.radiant_win else 0.0
            elo[rad] += K_FACTOR * (outcome - exp)
            elo[dire] += K_FACTOR * ((1.0 - outcome) - (1.0 - exp))
            recent[rad].append(outcome)
            recent[dire].append(1.0 - outcome)
            games[rad] += 1
            games[dire] += 1
            if getattr(row, "radiant_name", None):
                names[rad] = row.radiant_name
            if getattr(row, "dire_name", None):
                names[dire] = row.dire_name

        self.elo, self.recent, self.games, self.names = elo, recent, games, names
        # Team names are not unique (fake/duplicate registrations exist);
        # on collision, prefer the team with the most matches played.
        best: dict[str, int] = {}
        for tid, name in names.items():
            key = str(name).strip().lower()
            if key and (key not in best or games[tid] > games[best[key]]):
                best[key] = tid
        self._name_to_id = best

    def _fit_model(self, tiers: tuple[str, ...]) -> None:
        features = build_features(self.matches)
        train = features[
            features["tier"].isin(tiers)
            & (features["rad_games"] >= 5)
            & (features["dire_games"] >= 5)
        ]
        self.model = LogisticRegression()
        self.model.fit(train[FEATURE_COLS], train["radiant_win"])

    # ---- lookups -------------------------------------------------------

    def find_team(self, query: str) -> tuple[int | None, str | list[str]]:
        """Resolve a team name; returns (team_id, name) or (None, suggestions)."""
        key = query.strip().lower()
        if key in self._name_to_id:
            tid = self._name_to_id[key]
            return tid, self.names[tid]
        # Auto-resolve only on a single near-exact match; a lone weak match
        # (e.g. sharing just the word "team") must not silently resolve.
        strong = difflib.get_close_matches(key, self._name_to_id.keys(), n=2, cutoff=0.87)
        if len(strong) == 1:
            tid = self._name_to_id[strong[0]]
            return tid, self.names[tid]
        close = difflib.get_close_matches(key, self._name_to_id.keys(), n=5, cutoff=0.6)
        return None, [self.names[self._name_to_id[c]] for c in close]

    def _form(self, tid: int) -> float:
        hist = self.recent[tid]
        return (sum(hist) + 0.5 * (FORM_WINDOW - len(hist))) / FORM_WINDOW

    def team_overview(self, tid: int) -> dict:
        rank = sorted(self.elo.values(), reverse=True).index(self.elo[tid]) + 1
        return {
            "team": self.names.get(tid, str(tid)),
            "elo": round(self.elo[tid], 1),
            "elo_rank": f"{rank} of {len(self.elo)}",
            "total_matches_in_sample": self.games[tid],
            "recent_form_last10": f"{sum(self.recent[tid]):.0f} wins of {len(self.recent[tid])}",
        }

    def recent_matches(self, tid: int, n: int = 10) -> list[dict]:
        mask = (self.matches["radiant_team_id"] == tid) | (self.matches["dire_team_id"] == tid)
        rows = self.matches[mask].tail(n)
        out = []
        for row in rows.itertuples():
            is_radiant = row.radiant_team_id == tid
            opp = row.dire_team_id if is_radiant else row.radiant_team_id
            won = row.radiant_win == is_radiant
            out.append(
                {
                    "date": pd.Timestamp(row.start_time, unit="s").date().isoformat(),
                    "opponent": self.names.get(opp, str(opp)),
                    "result": "win" if won else "loss",
                    "tier": row.tier if isinstance(row.tier, str) else "unknown",
                }
            )
        return out

    def head_to_head(self, tid_a: int, tid_b: int) -> dict:
        mask = (
            (self.matches["radiant_team_id"] == tid_a) & (self.matches["dire_team_id"] == tid_b)
        ) | ((self.matches["radiant_team_id"] == tid_b) & (self.matches["dire_team_id"] == tid_a))
        rows = self.matches[mask]
        wins_a = int(
            (
                ((rows["radiant_team_id"] == tid_a) & rows["radiant_win"])
                | ((rows["dire_team_id"] == tid_a) & ~rows["radiant_win"])
            ).sum()
        )
        return {
            "matches": len(rows),
            f"{self.names.get(tid_a, tid_a)}_wins": wins_a,
            f"{self.names.get(tid_b, tid_b)}_wins": len(rows) - wins_a,
        }

    def predict_radiant(self, tid_rad: int, tid_dire: int) -> float:
        """Radiant win probability when sides are known (e.g. a live game)."""
        x = pd.DataFrame(
            [[self.elo[tid_rad] - self.elo[tid_dire], self._form(tid_rad) - self._form(tid_dire)]],
            columns=FEATURE_COLS,
        )
        return float(self.model.predict_proba(x)[0][1])

    def predict(self, tid_a: int, tid_b: int) -> dict:
        """Win probability for team A; side unknown, so both orientations are averaged."""
        elo_diff = self.elo[tid_a] - self.elo[tid_b]
        form_diff = self._form(tid_a) - self._form(tid_b)
        x = pd.DataFrame([[elo_diff, form_diff], [-elo_diff, -form_diff]], columns=FEATURE_COLS)
        proba = self.model.predict_proba(x)[:, 1]
        p = (proba[0] + (1.0 - proba[1])) / 2
        return {
            "team_a": self.names.get(tid_a, str(tid_a)),
            "team_b": self.names.get(tid_b, str(tid_b)),
            "team_a_win_probability": round(float(p), 3),
            "elo_diff": round(elo_diff, 1),
            "model": "logistic regression on Elo + recent form, trained on "
                     "premium/professional matches; draft not taken into account",
        }
