"""In-game win-probability model trained on historical gold graphs.

Training samples are (minute, gold lead, pre-match prior) snapshots taken
every 5 minutes of every professional match; the label is the final map
winner. Split is by match (time-based), so test matches are entirely
unseen. The trained model replaces the hand-tuned gold heuristic in the
live tracker.

Train:
    python -m dota_predictor.models.ingame
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from dota_predictor.models.baseline import FEATURE_COLS, time_split

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
GOLD_PATH = DATA_DIR / "raw" / "gold.parquet"
MODEL_PATH = DATA_DIR / "models" / "ingame.cbm"

INGAME_FEATURES = ["minute", "gold_1k", "prior_logit"]
SAMPLE_MINUTES = range(5, 61, 5)


def gold_logit_shift(gold_lead: float, game_time: float) -> float:
    """Legacy heuristic: 1k gold is worth ~0.04 logits early, ~0.10 late."""
    per_1k = 0.04 + 0.06 * min(max(game_time, 0.0) / 2400.0, 1.0)
    return per_1k * gold_lead / 1000.0


def heuristic_probability(pre_map: float, gold_lead: float, game_time: float) -> float:
    logit = np.log(pre_map / (1.0 - pre_map)) + gold_logit_shift(gold_lead, game_time)
    return float(1.0 / (1.0 + np.exp(-logit)))


def load_ingame_model(path: Path = MODEL_PATH):
    """Return the trained model or None when it hasn't been trained yet."""
    if not path.exists():
        return None
    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def model_probability(model, pre_map: float, gold_lead: float, game_time: float) -> float:
    prior_logit = float(np.log(pre_map / (1.0 - pre_map)))
    x = pd.DataFrame(
        [[game_time / 60.0, gold_lead / 1000.0, prior_logit]], columns=INGAME_FEATURES
    )
    return float(model.predict_proba(x)[0][1])


def build_samples(df: pd.DataFrame, gold: pd.DataFrame, prior: np.ndarray) -> pd.DataFrame:
    graphs = dict(zip(gold["match_id"], gold["radiant_gold_adv"]))
    prior_logit = np.log(np.clip(prior, 0.01, 0.99) / (1 - np.clip(prior, 0.01, 0.99)))
    rows = []
    for (row, pl) in zip(df.itertuples(), prior_logit):
        adv = graphs.get(row.match_id)
        if adv is None or len(adv) < 6:
            continue
        for minute in SAMPLE_MINUTES:
            if minute >= len(adv):
                break
            rows.append(
                {
                    "match_id": row.match_id,
                    "start_time": row.start_time,
                    "minute": float(minute),
                    "gold_1k": adv[minute] / 1000.0,
                    "prior_logit": pl,
                    "prior": float(1 / (1 + np.exp(-pl))),
                    "y": bool(row.radiant_win),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from catboost import CatBoostClassifier

    from dota_predictor.ingest.opendota import load_or_fetch_gold

    features = pd.read_parquet(DATA_DIR / "processed" / "features.parquet")
    leagues = pd.read_parquet(DATA_DIR / "raw" / "leagues.parquet")
    features = features.merge(leagues[["leagueid", "tier"]], on="leagueid", how="left")
    df = features[
        features["tier"].isin(["premium", "professional"])
        & (features["rad_games"] >= 5)
        & (features["dire_games"] >= 5)
    ].sort_values("start_time")

    train_m, test_m = time_split(df)
    prior_model = LogisticRegression()
    prior_model.fit(train_m[FEATURE_COLS], train_m["radiant_win"])

    gold = load_or_fetch_gold(GOLD_PATH, df["match_id"].tolist())
    print(f"Gold graphs available for {gold['match_id'].nunique()}/{len(df)} matches")

    samples = pd.concat(
        [
            build_samples(part, gold, prior_model.predict_proba(part[FEATURE_COLS])[:, 1])
            for part in (train_m, test_m)
        ],
        ignore_index=True,
    )
    is_test = samples["match_id"].isin(set(test_m["match_id"]))
    train_s, test_s = samples[~is_test], samples[is_test]
    fit_s, val_s = time_split(train_s, test_frac=0.15)
    print(f"Samples: {len(fit_s)} fit / {len(val_s)} val / {len(test_s)} test "
          f"({samples['match_id'].nunique()} matches)")

    model = CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=4, l2_leaf_reg=10,
        loss_function="Logloss", early_stopping_rounds=100, random_seed=42, verbose=0,
    )
    model.fit(fit_s[INGAME_FEATURES], fit_s["y"], eval_set=(val_s[INGAME_FEATURES], val_s["y"]))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"Model saved to {MODEL_PATH}")

    y = test_s["y"].to_numpy()
    p_model = model.predict_proba(test_s[INGAME_FEATURES])[:, 1]
    p_heur = np.array([
        heuristic_probability(r.prior, r.gold_1k * 1000, r.minute * 60)
        for r in test_s.itertuples()
    ])
    p_prior = test_s["prior"].to_numpy()

    print("\nTest (по срезам времени):")
    print(f"{'минута':>7} {'n':>6} | {'prior':>7} {'эвристика':>9} {'модель':>7}  (log loss)")
    for minute in (10, 20, 30, 40, 50):
        m = test_s["minute"] == minute
        if m.sum() < 100:
            continue
        print(f"{minute:>7} {int(m.sum()):>6} | {log_loss(y[m], p_prior[m]):>7.4f} "
              f"{log_loss(y[m], p_heur[m]):>9.4f} {log_loss(y[m], p_model[m]):>7.4f}")
    print(f"{'все':>7} {len(y):>6} | {log_loss(y, p_prior):>7.4f} "
          f"{log_loss(y, p_heur):>9.4f} {log_loss(y, p_model):>7.4f}")
    print(f"Accuracy модели на тесте: {accuracy_score(y, p_model >= 0.5):.3f}")

    print("\nЦена золота (P победы при равном предматчевом прогнозе):")
    print(f"{'лид':>8} | " + " ".join(f"{m:>5} мин" for m in (10, 20, 30, 40)))
    for g in (5, 10, 20):
        row = [
            model_probability(model, 0.5, g * 1000, m * 60) for m in (10, 20, 30, 40)
        ]
        print(f"{g:>6}k  | " + " ".join(f"{p:>8.1%}" for p in row))


if __name__ == "__main__":
    main()
