"""Схема новости."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class NewsItem(BaseModel):
    """Одна новость из источника (используется парсерами)."""
    source_id: str
    title: str
    summary: str | None = None
    url: str | None = None
    published_at: datetime
    tickers: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    sentiment: float = Field(ge=-1.0, le=1.0, default=0.0)
    ru_relevance: float = Field(ge=0.0, le=1.0, default=0.0)
    trump_mention: bool = False
    raw_score: float | None = None
    regulator_event: bool = False
    # Тип и вес источника из config/news_sources.yaml
    source_kind: str = "news"
    source_weight: float = 0.5
    source_group: str | None = None      # редакция / пара «лента + TG-дубль»
    forwarded_from: str | None = None    # TG: пост переслан из другого канала

