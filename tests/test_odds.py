import numpy as np
import pandas as pd
import pytest

from dota_predictor.ingest.odds import compare_with_odds, implied_prob, load_odds_csv


def test_implied_prob_removes_vig():
    # 1.50 / 2.80 with vig: raw implied 0.6667 + 0.3571 = 1.0238
    p = implied_prob(pd.Series([1.50]), pd.Series([2.80]))
    assert p.iloc[0] == pytest.approx(0.6667 / 1.0238, abs=1e-3)


def test_load_odds_csv_validates(tmp_path):
    path = tmp_path / "odds.csv"
    path.write_text("match_id,odds_radiant\n1,1.5\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_odds_csv(path)

    path.write_text("match_id,odds_radiant,odds_dire\n1,1.5,0.9\n")
    with pytest.raises(ValueError, match="must be > 1.0"):
        load_odds_csv(path)

    path.write_text("match_id,odds_radiant,odds_dire\n1,1.5,2.8\n1,1.5,2.8\n2,2.0,1.9\n")
    odds = load_odds_csv(path)
    assert len(odds) == 2  # duplicate match dropped


def test_compare_with_odds_betting_math(capsys):
    # Model strongly favors radiant at generous odds -> bets radiant in both.
    # Match 1: radiant wins, +1.0 profit; match 2: radiant loses, -1.0.
    test = pd.DataFrame({"match_id": [1, 2, 3], "radiant_win": [True, False, True]})
    proba = np.array([0.70, 0.70, 0.50])
    odds = pd.DataFrame(
        {"match_id": [1, 2], "odds_radiant": [2.0, 2.0], "odds_dire": [2.0, 2.0]}
    )
    compare_with_odds(test, proba, odds)
    out = capsys.readouterr().out
    assert "2 overlapping test matches" in out  # match 3 has no odds
    assert "2 bets" in out
    assert "hit rate 0.500" in out
    assert "ROI +0.000" in out
