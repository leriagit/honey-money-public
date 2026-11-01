"""Веб-приложение: лента, карточки бумаг и запрос решения у торгового модуля."""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from . import scorer, strategy_client
from .news import pipeline
from .news.claim_log import ClaimLog
from .schemas import NewsItem

logger = logging.getLogger(__name__)
WEB = Path(__file__).resolve().parents[1] / "web"
REFRESH_S = 180


class Feed:
    def __init__(self, offline: bool):
        self.offline = offline
        self.cfg = pipeline.load_config()
        self.tickers: List[str] = self.cfg["tickers"]
        self.items: List[NewsItem] = []
        self.cards: Dict[str, Dict[str, Any]] = {}
        self.updated_at = None
        self.lock = threading.Lock()

    def refresh(self) -> None:
        if self.offline:
            raw = pipeline.load_demo()
            items = pipeline.select(raw, limit=200, now=max(it.published_at for it in raw))
        else:
            sources = getattr(self, "_sources", None) or pipeline.build_sources(self.cfg)
            self._sources = sources
            log = ClaimLog(Path(os.environ.get("DATA_DIR", "./data")))
            items = pipeline.select(pipeline.collect(sources, claim_log=log), limit=200)
        scores = scorer.score(items, self.tickers)
        with self.lock:
            self.items, self.cards = items, scores
            self.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def loop(self) -> None:
        while True:
            try:
                self.refresh()
            except Exception:  # noqa: BLE001 — следующий проход повторит сбор
                logger.exception("сбор новостей")
            time.sleep(REFRESH_S)

    def about(self, ticker: str) -> List[NewsItem]:
        with self.lock:
            return [it for it in self.items if ticker in it.tickers]


def create_app(offline: bool | None = None) -> FastAPI:
    offline = os.environ.get("NEWS_OFFLINE", "") == "1" if offline is None else offline
    feed = Feed(offline)
    if offline:
        feed.refresh()
    else:
        threading.Thread(target=feed.loop, daemon=True).start()
    app = FastAPI(title="Honey Money")

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {"ok": True, "news": "demo" if offline else "live", "strategy": "live" if strategy_client.live_configured() else "replay",
                "updated_at": feed.updated_at}

    @app.get("/api/cards")
    def cards() -> List[Dict[str, Any]]:
        rows = []
        for t, v in feed.cards.items():
            about = feed.about(t)
            rows.append({"ticker": t, **v, "headlines": [{"title": it.title, "source": it.source_id, "url": it.url,
                                                          "published_at": it.published_at.isoformat()} for it in about[:5]]})
        return sorted(rows, key=lambda r: -abs(r["score"]))

    @app.post("/api/decision/{ticker}")
    def decision(ticker: str) -> Dict[str, Any]:
        if ticker not in feed.tickers:
            raise HTTPException(404, "бумаги нет в списке")
        hs = [{"title": it.title[:300], "source": it.source_id, "kind": it.source_kind} for it in feed.about(ticker)[:20]]
        res = strategy_client.decide(ticker, hs)
        if res is None:
            raise HTTPException(503, "Торговый модуль недоступен, записанного ответа по этой бумаге нет.")
        return res

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB / "index.html")

    return app

