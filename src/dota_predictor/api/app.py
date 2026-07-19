"""FastAPI service exposing the prediction model and the LLM preview agent.

Run:
    uvicorn dota_predictor.api.app:app --port 8123

The match data and model load once at startup (from the parquet caches
populated by `python -m dota_predictor.pipeline`).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from dota_predictor.llm import preview as preview_mod


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warms the shared ContextStore singleton also used by the agent tools.
    app.state.store = preview_mod.store()
    yield


app = FastAPI(
    title="dota-predictor",
    description="Win probability and LLM-generated previews for pro Dota 2 matches",
    version="0.4.0",
    lifespan=lifespan,
)


def _resolve_or_404(name: str) -> int:
    tid, name_or_suggestions = app.state.store.find_team(name)
    if tid is None:
        raise HTTPException(
            status_code=404,
            detail={"error": f"team '{name}' not found", "suggestions": name_or_suggestions},
        )
    return tid


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "matches_loaded": len(app.state.store.matches)}


@app.get("/teams")
def teams(query: str = Query(min_length=2)) -> dict:
    """Resolve a team name; returns the match or fuzzy suggestions."""
    tid, name_or_suggestions = app.state.store.find_team(query)
    if tid is None:
        return {"found": False, "suggestions": name_or_suggestions}
    return {"found": True, "team": app.state.store.team_overview(tid)}


@app.get("/predict")
def predict(team_a: str, team_b: str) -> dict:
    """Pre-draft win probability for team_a against team_b."""
    tid_a, tid_b = _resolve_or_404(team_a), _resolve_or_404(team_b)
    return {
        "prediction": app.state.store.predict(tid_a, tid_b),
        "team_a": app.state.store.team_overview(tid_a),
        "team_b": app.state.store.team_overview(tid_b),
        "head_to_head": app.state.store.head_to_head(tid_a, tid_b),
    }


@app.get("/preview")
def preview(team_a: str, team_b: str) -> dict:
    """LLM-generated analytical preview (requires Anthropic API credentials)."""
    _resolve_or_404(team_a)  # fail fast before spending an LLM call
    _resolve_or_404(team_b)
    try:
        text = preview_mod.generate_preview(team_a, team_b)
    except Exception as exc:  # missing credentials, API errors
        raise HTTPException(status_code=503, detail=f"LLM preview unavailable: {exc}")
    return {"team_a": team_a, "team_b": team_b, "preview": text}
