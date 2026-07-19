"""Probability calibration and reliability diagnostics (v2).

The GBDT is trained on `fit` and early-stopped on `val`; the same `val`
predictions provide (proba, outcome) pairs for fitting calibrators, so the
test set stays untouched until the final evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from dota_predictor.models.baseline import evaluate

EPS = 0.01  # keep probabilities away from 0/1 so log loss stays finite


def ece(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: bin-weighted |observed - predicted|."""
    idx = np.clip((proba * bins).astype(int), 0, bins - 1)
    err = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            err += mask.mean() * abs(y_true[mask].mean() - proba[mask].mean())
    return err


def reliability_table(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> pd.DataFrame:
    idx = np.clip((proba * bins).astype(int), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            rows.append(
                {
                    "bin": f"{b / bins:.1f}-{(b + 1) / bins:.1f}",
                    "n": int(mask.sum()),
                    "predicted": proba[mask].mean(),
                    "observed": y_true[mask].mean(),
                }
            )
    return pd.DataFrame(rows)


def calibrate_and_report(
    y_val: np.ndarray, val_proba: np.ndarray, y_test: np.ndarray, test_proba: np.ndarray
) -> np.ndarray:
    """Fit Platt and isotonic on val, evaluate on test, return best test probs."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS)
    iso.fit(val_proba, y_val)

    platt = LogisticRegression()
    platt.fit(_logit(val_proba), y_val)

    candidates = {
        "uncalibrated": np.clip(test_proba, EPS, 1 - EPS),
        "platt": platt.predict_proba(_logit(test_proba))[:, 1],
        "isotonic": iso.predict(test_proba),
    }
    print("\nCalibration (fit on early-stop val, evaluated on test):")
    results = {}
    for name, p in candidates.items():
        r = evaluate(f"catboost {name}", y_test, p)
        results[name] = r.logloss
        print(f"{r}  ece={ece(y_test, p):.4f}")

    print("\nReliability of uncalibrated probabilities:")
    print(
        reliability_table(y_test, test_proba).to_string(
            index=False, float_format=lambda v: f"{v:.3f}"
        )
    )
    best = min(results, key=results.get)
    print(f"\nBest by log loss: {best}")
    return candidates[best]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)
