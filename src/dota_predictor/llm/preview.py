"""LLM agent that writes an analytical preview for an upcoming match.

Claude gathers context itself via tools (team stats, head-to-head, model
prediction) and produces the preview. Uses the SDK tool runner, which
drives the tool-call loop automatically.

Usage:
    python -m dota_predictor.llm.preview "Team Spirit" "Team Falcons"

Requires Anthropic API credentials (ANTHROPIC_API_KEY or an `ant auth
login` profile).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anthropic
from anthropic import beta_tool

from dota_predictor.llm.context import ContextStore

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

SYSTEM_PROMPT = """\
Ты — аналитик профессиональной Dota 2. Твоя задача — написать превью \
предстоящего матча между двумя командами.

Собери контекст инструментами: обзор обеих команд, личные встречи, прогноз \
модели. Затем напиши превью на русском со структурой:
- Заголовок матча и главная интрига (1-2 предложения)
- Форма и положение каждой команды
- Личные встречи, если они есть
- Прогноз: вероятность от модели и твоя интерпретация — с чем модель может \
ошибаться (она не видит замены в составах, драфты и мету патча)

Опирайся только на данные из инструментов, не выдумывай факты. Данные \
покрывают ~2 года про-матчей; Elo и форма считаются по ним. Если команда не \
нашлась, сообщи и предложи варианты из подсказок инструмента."""

_store: ContextStore | None = None


def store() -> ContextStore:
    global _store
    if _store is None:
        print("Загружаю данные и обучаю модель...", file=sys.stderr)
        _store = ContextStore(DATA_DIR)
    return _store


def _resolve(team: str) -> tuple[int | None, str]:
    tid, name_or_suggestions = store().find_team(team)
    if tid is None:
        return None, json.dumps(
            {"error": f"team '{team}' not found", "suggestions": name_or_suggestions},
            ensure_ascii=False,
        )
    return tid, name_or_suggestions


@beta_tool
def get_team_overview(team: str) -> str:
    """Get a team's current Elo rating, rank, and recent form, plus its last 10 matches.

    Args:
        team: Team name, e.g. "Team Spirit". Fuzzy matching is applied.
    """
    tid, err = _resolve(team)
    if tid is None:
        return err
    data = store().team_overview(tid)
    data["last_matches"] = store().recent_matches(tid)
    return json.dumps(data, ensure_ascii=False)


@beta_tool
def get_head_to_head(team_a: str, team_b: str) -> str:
    """Get the head-to-head record between two teams in the data sample.

    Args:
        team_a: First team name.
        team_b: Second team name.
    """
    tid_a, err_a = _resolve(team_a)
    if tid_a is None:
        return err_a
    tid_b, err_b = _resolve(team_b)
    if tid_b is None:
        return err_b
    return json.dumps(store().head_to_head(tid_a, tid_b), ensure_ascii=False)


@beta_tool
def get_model_prediction(team_a: str, team_b: str) -> str:
    """Get the ML model's win probability for team_a against team_b (pre-draft).

    Args:
        team_a: First team name.
        team_b: Second team name.
    """
    tid_a, err_a = _resolve(team_a)
    if tid_a is None:
        return err_a
    tid_b, err_b = _resolve(team_b)
    if tid_b is None:
        return err_b
    return json.dumps(store().predict(tid_a, tid_b), ensure_ascii=False)


def generate_preview(team_a: str, team_b: str) -> str:
    client = anthropic.Anthropic()
    runner = client.beta.messages.tool_runner(
        model="claude-opus-4-8",
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        tools=[get_team_overview, get_head_to_head, get_model_prediction],
        messages=[
            {"role": "user", "content": f"Напиши превью матча: {team_a} против {team_b}."}
        ],
    )
    final = runner.until_done()
    return "".join(block.text for block in final.content if block.type == "text")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team_a")
    parser.add_argument("team_b")
    args = parser.parse_args()
    print(generate_preview(args.team_a, args.team_b))


if __name__ == "__main__":
    main()
