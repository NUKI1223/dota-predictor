"""Live win-probability tracker for ongoing pro matches.

Usage:
    python -m dota_predictor.live                 # one snapshot
    python -m dota_predictor.live --watch 60      # refresh every 60 s
    python -m dota_predictor.live --odds 1.45,2.75  # compare vs a bookmaker line

Data source: the free OpenDota /live endpoint (teams, score, game time,
gold lead of the radiant side). The pre-map probability comes from the
serving model (Elo + form, radiant-oriented). The in-game adjustment
blends the current gold lead into the logit with a time-dependent weight.

NOTE: the gold-lead weighting is a transparent heuristic, not (yet) a
model trained on historical gold graphs — treat in-game numbers as
directional. Bookmaker odds can be passed manually via --odds; an API
odds source (e.g. OddsPapi) can plug in through ingest.odds.implied_prob
the same way.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx
import numpy as np

from dota_predictor.ingest.odds import implied_prob
from dota_predictor.llm.context import ContextStore

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def fetch_live() -> list[dict]:
    resp = httpx.get("https://api.opendota.com/api/live", timeout=30)
    resp.raise_for_status()
    games = resp.json()
    return [
        g for g in games
        if g.get("league_id") and g.get("team_id_radiant") and g.get("team_id_dire")
    ]


def gold_logit_shift(gold_lead: float, game_time: float) -> float:
    """Heuristic: a 1k-gold lead is worth ~0.04 logits early, ~0.10 late."""
    per_1k = 0.04 + 0.06 * min(max(game_time, 0.0) / 2400.0, 1.0)
    return per_1k * gold_lead / 1000.0


def live_probability(pre_map: float, gold_lead: float, game_time: float) -> float:
    logit = np.log(pre_map / (1.0 - pre_map)) + gold_logit_shift(gold_lead, game_time)
    return float(1.0 / (1.0 + np.exp(-logit)))


def snapshot(store: ContextStore, odds: tuple[float, float] | None = None) -> None:
    games = fetch_live()
    if not games:
        print("Живых лиговых матчей сейчас нет.")
        return
    for g in games:
        rad, dire = int(g["team_id_radiant"]), int(g["team_id_dire"])
        name_rad = g.get("team_name_radiant") or store.names.get(rad, str(rad))
        name_dire = g.get("team_name_dire") or store.names.get(dire, str(dire))
        game_time = g.get("game_time") or 0
        lead = g.get("radiant_lead") or 0

        pre = store.predict_radiant(rad, dire)
        live = live_probability(pre, lead, game_time)
        print(
            f"[{game_time // 60:>3} мин] {name_rad} vs {name_dire}  "
            f"счёт {g.get('radiant_score', '?')}:{g.get('dire_score', '?')}  "
            f"золото {lead:+,}"
        )
        print(
            f"          P({name_rad}): до карты {pre:.1%} -> сейчас {live:.1%}"
        )
        if odds is not None:
            import pandas as pd

            book = float(implied_prob(pd.Series([odds[0]]), pd.Series([odds[1]])).iloc[0])
            print(
                f"          букмекер (де-виг): {book:.1%}  |  наш эдж: {live - book:+.1%}"
            )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", type=int, default=0, help="refresh interval, seconds")
    parser.add_argument("--odds", default=None,
                        help="decimal odds 'radiant,dire' to compare against")
    args = parser.parse_args()
    odds = tuple(float(x) for x in args.odds.split(",")) if args.odds else None

    print("Загружаю данные и модель...", file=sys.stderr)
    store = ContextStore(DATA_DIR)
    while True:
        print(f"--- {time.strftime('%H:%M:%S')} ---")
        snapshot(store, odds)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
