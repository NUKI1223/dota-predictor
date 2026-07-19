"""End-to-end pipeline: fetch pro matches -> Elo features -> baseline model.

Usage:
    python -m dota_predictor.pipeline [--pages 20] [--refresh]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dota_predictor.features.draft import build_draft_features
from dota_predictor.features.elo import build_features, current_ratings
from dota_predictor.ingest.opendota import (
    extend_history,
    load_or_fetch,
    load_or_fetch_drafts,
    load_or_fetch_leagues,
)
from dota_predictor.ingest.odds import compare_with_odds, load_odds_csv
from dota_predictor.models.baseline import train_and_evaluate
from dota_predictor.models.calibration import calibrate_and_report
from dota_predictor.models.gbdt import train_and_evaluate_gbdt

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_PATH = DATA_DIR / "raw" / "pro_matches.parquet"
DRAFTS_PATH = DATA_DIR / "raw" / "drafts.parquet"
LEAGUES_PATH = DATA_DIR / "raw" / "leagues.parquet"
FEATURES_PATH = DATA_DIR / "processed" / "features.parquet"


def main() -> None:
    # Team names contain unicode the default Windows console codepage can't print.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=20, help="pages of 100 matches to fetch")
    parser.add_argument("--refresh", action="store_true", help="re-fetch even if cache exists")
    parser.add_argument(
        "--history-days",
        type=int,
        default=730,
        help="how far back the match history should reach (fetched via /explorer)",
    )
    parser.add_argument(
        "--tiers",
        default="premium,professional",
        help="comma-separated league tiers to train/evaluate on; empty = all matches",
    )
    parser.add_argument(
        "--odds-csv",
        type=Path,
        default=None,
        help="CSV with match_id,odds_radiant,odds_dire for model-vs-market comparison",
    )
    args = parser.parse_args()

    matches = load_or_fetch(RAW_PATH, pages=args.pages, refresh=args.refresh)
    if args.history_days:
        matches = extend_history(RAW_PATH, matches, args.history_days)
    print(f"\n{len(matches)} matches, "
          f"{matches['radiant_team_id'].nunique()} unique radiant teams, "
          f"span {matches['start_time'].min()} .. {matches['start_time'].max()} (unix)")

    drafts = load_or_fetch_drafts(DRAFTS_PATH, matches["match_id"].tolist())
    print(f"Drafts available for {drafts['match_id'].nunique()}/{len(matches)} matches")

    # Features (Elo, form, hero winrates) are computed over ALL matches —
    # more history means better ratings. The tier filter below only decides
    # which matches the models are trained and evaluated on.
    features = build_features(matches)
    features, bag = build_draft_features(features, drafts)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_PATH, index=False)

    leagues = load_or_fetch_leagues(LEAGUES_PATH, refresh=args.refresh)
    features = features.merge(leagues[["leagueid", "tier"]], on="leagueid", how="left")
    print("\nMatches by league tier:")
    print(features["tier"].value_counts(dropna=False).to_string())

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    model_df = features[features["tier"].isin(tiers)] if tiers else features
    print(f"\nTraining on tiers {tiers or 'ALL'}: {len(model_df)}/{len(features)} matches")

    train_and_evaluate(model_df)
    gbdt_out = train_and_evaluate_gbdt(model_df, bag)

    calibrated_proba = calibrate_and_report(
        gbdt_out["y_val"], gbdt_out["val_proba"], gbdt_out["y_test"], gbdt_out["test_proba"]
    )
    if args.odds_csv:
        compare_with_odds(gbdt_out["test"], calibrated_proba, load_odds_csv(args.odds_csv))
    else:
        print("\nNo --odds-csv given; skipping model-vs-bookmaker comparison "
              "(format: match_id,odds_radiant,odds_dire).")

    print("\nTop-10 teams by current Elo:")
    print(current_ratings(matches).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
