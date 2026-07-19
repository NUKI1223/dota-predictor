"""Baseline model: logistic regression on Elo/form differences.

Validation is a time split (train on the past, test on the future) — the
only honest way to evaluate match prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

FEATURE_COLS = ["elo_diff", "form_diff"]


@dataclass
class EvalResult:
    name: str
    accuracy: float
    logloss: float
    brier: float

    def __str__(self) -> str:
        return (
            f"{self.name:<24} accuracy={self.accuracy:.3f}  "
            f"log_loss={self.logloss:.4f}  brier={self.brier:.4f}"
        )


def time_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("start_time")
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut], df.iloc[cut:]


def evaluate(name: str, y_true: np.ndarray, proba: np.ndarray) -> EvalResult:
    return EvalResult(
        name=name,
        accuracy=accuracy_score(y_true, proba >= 0.5),
        logloss=log_loss(y_true, proba),
        brier=brier_score_loss(y_true, proba),
    )


def train_and_evaluate(features: pd.DataFrame, min_games: int = 5) -> LogisticRegression:
    """Train the baseline and print metrics against naive references.

    Matches where either team has fewer than `min_games` of history are
    dropped: their Elo is still near the initial value and mostly noise.
    """
    df = features[
        (features["rad_games"] >= min_games) & (features["dire_games"] >= min_games)
    ].copy()
    train, test = time_split(df)
    y_train = train["radiant_win"].to_numpy()
    y_test = test["radiant_win"].to_numpy()

    model = LogisticRegression()
    model.fit(train[FEATURE_COLS], y_train)
    proba = model.predict_proba(test[FEATURE_COLS])[:, 1]

    print(f"\nDataset: {len(df)} matches ({len(train)} train / {len(test)} test)")
    print(f"Radiant win rate in test: {y_test.mean():.3f}\n")

    results = [
        evaluate("always 0.5", y_test, np.full(len(y_test), 0.5)),
        evaluate("elo favorite (hard)", y_test, np.where(test["elo_diff"] > 0, 0.75, 0.25)),
        evaluate("logreg (elo + form)", y_test, proba),
    ]
    for r in results:
        print(r)

    coef = dict(zip(FEATURE_COLS, model.coef_[0]))
    print(f"\nCoefficients: {coef}, intercept={model.intercept_[0]:.3f}")
    return model
