# dota-predictor

Предсказание вероятности победы в профессиональных матчах Dota 2.

## Быстрый старт

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python -m dota_predictor.pipeline --pages 20
```

Пайплайн скачивает последние про-матчи с [OpenDota API](https://docs.opendota.com/)
(кэширует в `data/raw/`), считает Elo-рейтинги и форму команд, обучает
логистическую регрессию и сравнивает её с наивными baseline'ами по
accuracy / log loss / Brier score на временном сплите (train — прошлое,
test — будущее).

## Структура

```
src/dota_predictor/
├── ingest/      # клиенты внешних API (OpenDota)
├── features/    # Elo, форма — только pre-match информация, без утечек
├── models/      # baseline и метрики
└── pipeline.py  # точка входа
```

## Roadmap

- [x] v0: Elo + форма, логистическая регрессия
- [x] v1: драфт-фичи (винрейты героев, bag-of-heroes), CatBoost
      (драфты тянутся пачками через `/explorer` SQL endpoint)
- [x] v1.1: фильтр по тиру лиги (`--tiers premium,professional` по умолчанию);
      фичи считаются по всем матчам, обучение/оценка — только по выбранным тирам
- [x] v1.2: глубокая история через `/explorer` по таблице `matches`
      (`--history-days 730` по умолчанию, ~54 тыс. матчей); затухание
      винрейтов героев с half-life 90 дней под смену патчей
- [x] v2: калибровка (Platt/isotonic, ECE, reliability-таблица); сравнение
      с кэфами через `--odds-csv` (формат `match_id,odds_radiant,odds_dire`,
      вилка снимается нормализацией, есть симуляция флэт-ставок)
- [x] v3: LLM-агент превью матчей — Claude (Anthropic SDK, tool runner) сам
      собирает контекст инструментами (обзор команд, личные встречи, прогноз
      модели) и пишет аналитическое превью:
      ```powershell
      $env:ANTHROPIC_API_KEY = "sk-ant-..."
      python -m dota_predictor.llm.preview "Team Spirit" "Team Falcons"
      ```
- [x] v4: FastAPI-сервис:
      ```powershell
      .venv\Scripts\python -m uvicorn dota_predictor.api.app:app --port 8123
      ```
      Эндпоинты: `GET /health`, `GET /teams?query=...` (поиск команды),
      `GET /predict?team_a=...&team_b=...` (вероятность + форма + личные
      встречи), `GET /preview?...` (LLM-превью; нужен ANTHROPIC_API_KEY,
      иначе 503). Интерактивная документация — `/docs`.
- [x] v5: мета патча и контекст турнира — винрейты/популярность героев
      внутри текущего патча (сброс на границе патча), форма команды на
      текущем турнире до матча, винрейты героев на турнире. Форма на
      турнире — вторая по важности фича модели после Elo.
- [x] v5.1: баны из `picks_bans` — сила и «метовость» забаненных героев,
      целевые баны (насколько соперник выцеливает сигнатурных героев
      команды в текущем патче).
