"""Client for the OpenDota public API (free tier: ~2000 calls/day, 60/min)."""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import pandas as pd

BASE_URL = "https://api.opendota.com/api"
PAGE_SIZE = 100  # /proMatches returns 100 matches per page

COLUMNS = [
    "match_id",
    "start_time",
    "duration",
    "leagueid",
    "league_name",
    "radiant_team_id",
    "radiant_name",
    "dire_team_id",
    "dire_name",
    "radiant_score",
    "dire_score",
    "radiant_win",
]


def fetch_pro_matches(pages: int = 20, pause: float = 1.1) -> pd.DataFrame:
    """Fetch recent pro matches, paginating backwards from the newest one.

    Returns a DataFrame sorted by start_time ascending, with matches that
    lack either team id dropped (they are unusable for team-level features).
    """
    rows: list[dict] = []
    less_than: int | None = None
    with httpx.Client(base_url=BASE_URL, timeout=30) as client:
        for page in range(pages):
            params = {"less_than_match_id": less_than} if less_than else {}
            resp = client.get("/proMatches", params=params)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            rows.extend(batch)
            less_than = batch[-1]["match_id"]
            print(f"  page {page + 1}/{pages}: {len(rows)} matches total")
            time.sleep(pause)  # stay under the free-tier rate limit

    df = pd.DataFrame(rows)
    df = df[[c for c in COLUMNS if c in df.columns]]
    df = df.dropna(subset=["radiant_team_id", "dire_team_id"])
    df = df.drop_duplicates(subset="match_id")
    df["radiant_team_id"] = df["radiant_team_id"].astype("int64")
    df["dire_team_id"] = df["dire_team_id"].astype("int64")
    df["radiant_win"] = df["radiant_win"].astype(bool)
    return df.sort_values("start_time").reset_index(drop=True)


def fetch_matches_history(
    cutoff_ts: int, before_match_id: int, batch: int = 5000, pause: float = 2.0
) -> pd.DataFrame:
    """Fetch pro matches older than `before_match_id` back to `cutoff_ts`.

    Uses /explorer with keyset pagination on match_id (descending): the
    matches table PK makes these queries cheap, and one call returns
    thousands of matches. match_id only roughly follows start_time, so the
    stop condition uses the batch median, and the result is trimmed to the
    cutoff at the end.
    """
    frames: list[pd.DataFrame] = []
    last = before_match_id
    with httpx.Client(base_url=BASE_URL, timeout=120) as client:
        while True:
            sql = (
                "SELECT match_id, start_time, duration, leagueid, radiant_team_id, "
                "dire_team_id, radiant_score, dire_score, radiant_win FROM matches "
                f"WHERE match_id < {last} AND radiant_team_id IS NOT NULL "
                "AND dire_team_id IS NOT NULL "
                f"ORDER BY match_id DESC LIMIT {batch}"
            )
            resp = client.get("/explorer", params={"sql": sql})
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("err"):
                raise RuntimeError(f"explorer error: {payload['err']}")
            rows = pd.DataFrame(payload["rows"])
            if rows.empty:
                break
            frames.append(rows)
            last = int(rows["match_id"].min())
            median_day = pd.Timestamp(rows["start_time"].median(), unit="s").date()
            total = sum(len(f) for f in frames)
            print(f"  history: {total} matches, reached ~{median_day}")
            if rows["start_time"].median() < cutoff_ts:
                break
            time.sleep(pause)

    df = pd.concat(frames, ignore_index=True)
    return df[df["start_time"] >= cutoff_ts].drop_duplicates(subset="match_id")


def fetch_team_names() -> pd.DataFrame:
    """All known team names in one /explorer call (~22k rows)."""
    with httpx.Client(base_url=BASE_URL, timeout=120) as client:
        resp = client.get("/explorer", params={"sql": "SELECT team_id, name FROM teams"})
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("err"):
            raise RuntimeError(f"explorer error: {payload['err']}")
    return pd.DataFrame(payload["rows"])


def extend_history(path: Path, matches: pd.DataFrame, days_back: int) -> pd.DataFrame:
    """Ensure the match cache reaches `days_back` into the past.

    Fetches older matches via /explorer if needed, merges them into the
    cache at `path` and returns the extended DataFrame.
    """
    cutoff = int(time.time()) - days_back * 86400
    if matches["start_time"].min() <= cutoff:
        return matches

    print(f"Extending history to {days_back} days via /explorer...")
    older = fetch_matches_history(cutoff, before_match_id=int(matches["match_id"].min()))
    names = fetch_team_names().set_index("team_id")["name"]
    older["radiant_name"] = older["radiant_team_id"].map(names)
    older["dire_name"] = older["dire_team_id"].map(names)
    older = older.reindex(columns=matches.columns)

    merged = pd.concat([matches, older], ignore_index=True)
    merged = merged.drop_duplicates(subset="match_id").sort_values("start_time")
    merged = merged.astype(
        {"radiant_team_id": "int64", "dire_team_id": "int64", "radiant_win": "bool"}
    ).reset_index(drop=True)
    merged.to_parquet(path, index=False)
    print(f"Cache extended to {len(merged)} matches")
    return merged


def load_or_fetch_leagues(path: Path, refresh: bool = False) -> pd.DataFrame:
    """League directory with tier: premium / professional / amateur / excluded."""
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    print("Fetching league directory from OpenDota...")
    with httpx.Client(base_url=BASE_URL, timeout=60) as client:
        resp = client.get("/leagues")
        resp.raise_for_status()
        leagues = pd.DataFrame(resp.json())[["leagueid", "name", "tier"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    leagues.to_parquet(path, index=False)
    print(f"Saved {len(leagues)} leagues to {path}")
    return leagues


def fetch_drafts(match_ids: list[int], chunk_size: int = 800, pause: float = 2.0) -> pd.DataFrame:
    """Fetch hero picks for the given matches via the /explorer SQL endpoint.

    One SQL call covers a contiguous match_id range (~chunk_size of our
    matches), so thousands of drafts cost tens of API calls instead of one
    call per match. Rows outside our id set (other parsed matches in the
    range) are filtered out client-side.

    Returns long format: match_id, hero_id, is_radiant.
    """
    wanted = sorted(set(match_ids))
    frames: list[pd.DataFrame] = []
    with httpx.Client(base_url=BASE_URL, timeout=120) as client:
        for i in range(0, len(wanted), chunk_size):
            chunk = wanted[i : i + chunk_size]
            sql = (
                "SELECT match_id, hero_id, player_slot FROM player_matches "
                f"WHERE match_id BETWEEN {chunk[0]} AND {chunk[-1]}"
            )
            resp = client.get("/explorer", params={"sql": sql})
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("err"):
                raise RuntimeError(f"explorer error: {payload['err']}")
            rows = pd.DataFrame(payload["rows"])
            if not rows.empty:
                rows = rows[rows["match_id"].isin(chunk)]
                frames.append(rows)
            print(f"  drafts: {i + len(chunk)}/{len(wanted)} matches requested")
            time.sleep(pause)

    drafts = pd.concat(frames, ignore_index=True)
    drafts["is_radiant"] = drafts["player_slot"] < 128
    return drafts[["match_id", "hero_id", "is_radiant"]]


def load_or_fetch_drafts(path: Path, match_ids: list[int]) -> pd.DataFrame:
    """Load cached drafts, fetching only matches missing from the cache."""
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame(
        columns=["match_id", "hero_id", "is_radiant"]
    )
    missing = sorted(set(match_ids) - set(cached["match_id"]))
    if missing:
        print(f"Fetching drafts for {len(missing)} matches via /explorer...")
        fresh = fetch_drafts(missing)
        cached = pd.concat([cached, fresh], ignore_index=True)
        # Concat with the empty seed frame degrades dtypes to object.
        cached = cached.astype({"match_id": "int64", "hero_id": "int64", "is_radiant": "bool"})
        path.parent.mkdir(parents=True, exist_ok=True)
        cached.to_parquet(path, index=False)
    else:
        print(f"All {len(match_ids)} drafts already cached in {path}")
    cached = cached.astype({"match_id": "int64", "hero_id": "int64", "is_radiant": "bool"})
    return cached[cached["match_id"].isin(set(match_ids))]


def load_or_fetch(path: Path, pages: int = 20, refresh: bool = False) -> pd.DataFrame:
    """Load cached matches from parquet, fetching from the API if absent."""
    if path.exists() and not refresh:
        print(f"Loading cached matches from {path}")
        return pd.read_parquet(path)
    print(f"Fetching {pages} pages of pro matches from OpenDota...")
    df = fetch_pro_matches(pages=pages)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} matches to {path}")
    return df
