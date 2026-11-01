"""Журнал новостей: каждая новая запись пишется один раз в DATA_DIR/news/claims_YYYY-MM-DD.jsonl.

Хранится ссылка и короткий заголовок, не полный текст (авторские права).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Set

from ..schemas import NewsItem

logger = logging.getLogger(__name__)

TITLE_MAX = 200


def claim_id(item: NewsItem) -> str:
    key = f"{item.source_id}|{item.url or item.title[:120]}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


class ClaimLog:
    def __init__(self, data_dir: Optional[Path] = None, keep_days: int = 3, subdir: str = "news") -> None:
        base = Path(data_dir or os.environ.get("DATA_DIR", "./data"))
        self.dir = base / subdir                  # news — старый бот, news_live — сборщик новостей
        self.keep_days = keep_days
        self._lock = threading.Lock()
        self._seen: Set[str] = set()
        self._load_recent_ids()

    def _path(self, day: datetime) -> Path:
        return self.dir / f"claims_{day:%Y-%m-%d}.jsonl"

    def _load_recent_ids(self) -> None:
        """После рестарта не писать повторно то, что уже в журнале за последние дни."""
        today = datetime.now(timezone.utc)
        for d in range(self.keep_days):
            p = self._path(today - timedelta(days=d))
            if not p.exists():
                continue
            try:
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        try:
                            self._seen.add(json.loads(line)["claim_id"])
                        except (ValueError, KeyError):
                            continue
            except OSError as e:
                logger.warning("claim_log: не прочитан %s: %s", p, e)

    def record(self, items: Iterable[NewsItem], now: Optional[datetime] = None) -> int:
        """Дописать новые записи. Возвращает их число. Никогда не бросает наружу."""
        return len(self.record_rows(items, now))

    def record_rows(self, items: Iterable[NewsItem], now: Optional[datetime] = None) -> List[dict]:
        """То же, что record, но возвращает записанные строки."""
        now = now or datetime.now(timezone.utc)
        rows = []
        with self._lock:
            for it in items:
                cid = claim_id(it)
                if cid in self._seen:
                    continue
                self._seen.add(cid)
                rows.append({
                    "claim_id": cid,
                    "event_id": None,                 # склейка в событие — следующий этап
                    "source": it.source_id,
                    "source_kind": it.source_kind,
                    "source_weight": it.source_weight,
                    "source_group": it.source_group,
                    "forwarded_from": it.forwarded_from,
                    "published_at": it.published_at.isoformat(),
                    "ts_first_seen": now.isoformat(),
                    "lead_sec_vs_first": None,
                    "claim_type": None,               # fact / forecast / opinion — разбор позже
                    "tickers": it.tickers,
                    "title": it.title[:TITLE_MAX],
                    "origin_url": it.url,
                    "sentiment": it.sentiment,
                    "ru_relevance": it.ru_relevance,
                    "regulator_event": it.regulator_event,
                    "status": "provisional",
                    "confirmed_by": None,
                    "contradicted_by": None,
                    "abnormal_return_1d": None,
                    "abnormal_return_3d": None,
                })
            if not rows:
                return []
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
                with open(self._path(now), "a", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            except OSError as e:
                logger.warning("claim_log: запись не удалась: %s", e)
                return []
        return rows
