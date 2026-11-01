"""Клиент закрытого торгового модуля. Без ключа или при сбое отдаёт записанный ответ и прямо об этом пишет."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)
REPLAY = Path(__file__).resolve().parents[1] / "demo" / "replay.json"


def _replay(ticker: str, reason: str) -> Optional[Dict[str, Any]]:
    data = json.loads(REPLAY.read_text(encoding="utf-8"))
    for row in data["items"]:
        if row["ticker"] == ticker:
            return {**row, "source": "replay", "notice": f"Демо: записанный ответ, не живая работа модели ({reason})."}
    return None


def live_configured() -> bool:
    return bool(os.environ.get("STRATEGY_API_URL") and os.environ.get("STRATEGY_API_KEY"))


def decide(ticker: str, headlines: List[Dict[str, Any]], transport: Optional[httpx.BaseTransport] = None) -> Optional[Dict[str, Any]]:
    if not live_configured():
        return _replay(ticker, "ключ API не задан")
    try:
        with httpx.Client(transport=transport, timeout=120) as c:
            r = c.post(os.environ["STRATEGY_API_URL"].rstrip("/") + "/v1/decide", json={"ticker": ticker, "headlines": headlines},
                       headers={"X-API-Key": os.environ["STRATEGY_API_KEY"]})
        r.raise_for_status()
        return r.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("торговый модуль не ответил: %s", type(e).__name__)
        return _replay(ticker, "сервис не ответил")
