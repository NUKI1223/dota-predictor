"""Gradient boosting on Elo + form + draft features.

Falls back to sklearn's HistGradientBoosting if CatBoost is unavailable.
"""

from __future__ import annotations

import pandas as pd

from dota_predictor.models.baseline import FEATURE_COLS, evaluate, time_split

DRAFT_COLS = ["hero_wr_diff", "hero_xp_diff"]


def _make_model():
    try:
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=2000,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=10,
            loss_function="Logloss",
            early_stopping_rounds=100,
            random_seed=42,
            verbose=0,
        ), "catboost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            random_state=42, early_stopping=True, validation_fraction=0.15
        ), "hist_gbdt"


def train_and_evaluate_gbdt(
    features: pd.DataFrame, bag: pd.DataFrame, min_games: int = 5
) -> dict:
    """Train GBDT variants, print test metrics.

    Returns val/test predictions of the draft-aggregates variant (the
    production candidate) for downstream calibration and odds comparison.
    """
    df = features.merge(bag, on="match_id")
    df = df[
        (df["rad_games"] >= min_games)
        & (df["dire_games"] >= min_games)
        & df["hero_wr_diff"].notna()
    ]
    hero_cols = [c for c in bag.columns if c.startswith("hero_")]

    train, test = time_split(df)
    # Last 15% of train (chronologically) is the early-stopping validation set.
    fit, val = time_split(train, test_frac=0.15)
    y_fit = fit["radiant_win"].to_numpy()
    y_val = val["radiant_win"].to_numpy()
    y_test = test["radiant_win"].to_numpy()
    print(f"\nGBDT dataset: {len(df)} matches "
          f"({len(fit)} fit / {len(val)} early-stop val / {len(test)} test)")

    out: dict = {}
    for name, cols in [
        ("gbdt elo+form", FEATURE_COLS),
        ("gbdt +draft aggregates", FEATURE_COLS + DRAFT_COLS),
        ("gbdt +bag-of-heroes", FEATURE_COLS + DRAFT_COLS + hero_cols),
    ]:
        model, impl = _make_model()
        if impl == "catboost":
            model.fit(fit[cols], y_fit, eval_set=(val[cols], y_val))
        else:
            model.fit(train[cols], train["radiant_win"].to_numpy())
        proba = model.predict_proba(test[cols])[:, 1]
        print(evaluate(f"{name} [{impl}]", y_test, proba))
        if cols == FEATURE_COLS + DRAFT_COLS:
            out = {
                "y_val": y_val,
                "val_proba": model.predict_proba(val[cols])[:, 1],
                "y_test": y_test,
                "test_proba": proba,
                "test": test,
            }
    return out
