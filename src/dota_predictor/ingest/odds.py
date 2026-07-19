"""Bookmaker odds: loading and model-vs-market comparison (v2).

Expected CSV format — one row per match, decimal (European) closing odds:

    match_id,odds_radiant,odds_dire
    8902610619,1.45,2.75

Any source works (OddsPapi export, PandaScore, manual collection); the
match_id ties a row to OpenDota data. Implied probabilities are de-vigged
by normalization: p = (1/o_r) / (1/o_r + 1/o_d).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dota_predictor.models.baseline import evaluate

MIN_EDGE = 0.03  # bet only when model EV per unit stake exceeds this


def load_odds_csv(path: Path) -> pd.DataFrame:
    odds = pd.read_csv(path)
    required = {"match_id", "odds_radiant", "odds_dire"}
    missing = required - set(odds.columns)
    if missing:
        raise ValueError(f"odds CSV is missing columns: {sorted(missing)}")
    odds = odds.dropna(subset=list(required)).astype(
        {"match_id": "int64", "odds_radiant": "float64", "odds_dire": "float64"}
    )
    if (odds[["odds_radiant", "odds_dire"]] <= 1.0).any().any():
        raise ValueError("decimal odds must be > 1.0")
    return odds.drop_duplicates(subset="match_id")


def implied_prob(odds_radiant: pd.Series, odds_dire: pd.Series) -> pd.Series:
    """De-vigged probability of a radiant win."""
    inv_r, inv_d = 1.0 / odds_radiant, 1.0 / odds_dire
    return inv_r / (inv_r + inv_d)


def compare_with_odds(test: pd.DataFrame, model_proba: np.ndarray, odds: pd.DataFrame) -> None:
    """Compare model vs bookmaker on matches present in both; simulate flat betting."""
    df = test[["match_id", "radiant_win"]].copy()
    df["model_p"] = model_proba
    df = df.merge(odds, on="match_id")
    if df.empty:
        print("No overlap between test matches and the odds file — nothing to compare.")
        return
    y = df["radiant_win"].to_numpy()
    book_p = implied_prob(df["odds_radiant"], df["odds_dire"]).to_numpy()

    print(f"\nModel vs bookmaker on {len(df)} overlapping test matches:")
    print(evaluate("bookmaker (de-vigged)", y, book_p))
    print(evaluate("model", y, df["model_p"].to_numpy()))

    # Flat 1-unit stake on any side with positive expected value above MIN_EDGE.
    edge_rad = df["model_p"] * df["odds_radiant"] - 1.0
    edge_dire = (1.0 - df["model_p"]) * df["odds_dire"] - 1.0
    bet_rad = (edge_rad > MIN_EDGE) & (edge_rad >= edge_dire)
    bet_dire = (edge_dire > MIN_EDGE) & ~bet_rad
    profit = np.where(
        bet_rad,
        np.where(y == 1, df["odds_radiant"] - 1.0, -1.0),
        np.where(bet_dire, np.where(y == 0, df["odds_dire"] - 1.0, -1.0), 0.0),
    )
    n_bets = int(bet_rad.sum() + bet_dire.sum())
    if n_bets:
        wins = int(((bet_rad & (y == 1)) | (bet_dire & (y == 0))).sum())
        print(
            f"Flat betting sim (edge > {MIN_EDGE:.0%}): {n_bets} bets, "
            f"hit rate {wins / n_bets:.3f}, ROI {profit.sum() / n_bets:+.3f} per unit"
        )
    else:
        print("Flat betting sim: no bets cleared the edge threshold.")
