"""Загрузка лент из конфига, сбор и отбор заголовков."""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ..schemas import NewsItem
from .base import NewsSource, SourceConfig
from .claim_log import ClaimLog

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "news_sources.yaml"
KIND_MULT = {"official": 1.0, "news": 1.0, "analysis": 0.8}
HALF_LIFE_H = 6.0
MAX_PER_SOURCE = 30
SURGUT = re.compile(r"сургутнефтегаз|surgutneftegas|sngs", re.I)


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_sources(cfg: Optional[dict] = None) -> List[NewsSource]:
    from .parsers._rss_common import DOMAIN_GUARD, RSSSource
    cfg = cfg or load_config()
    DOMAIN_GUARD.configure(cfg.get("domain_limits"), cfg.get("defaults"))
    out: List[NewsSource] = []
    for s in cfg["sources"]:
        out.append(RSSSource(SourceConfig(
            id=s["id"], name=s.get("name", s["id"]), weight=float(s.get("weight", 0.5)), kind=s.get("kind", "news"),
            rss_url=s.get("rss_url"), urls=s.get("urls"), ru_specialized=bool(s.get("ru_specialized")),
            market_filter=bool(s.get("market_filter")), group=s.get("group"), tz_fix_hours=float(s.get("tz_fix_hours", 0)),
            drop_keywords=s.get("drop_keywords"), extra=s)))
    return out


def collect(sources: List[NewsSource], window_h: int = 24, claim_log: Optional[ClaimLog] = None) -> List[NewsItem]:
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=window_h)

    def one(src: NewsSource) -> List[NewsItem]:
        try:
            return src.fetch(since) or []
        except Exception as e:  # noqa: BLE001 — сломанная лента не останавливает остальные
            logger.warning("лента %s: %s", src.id, e)
            return []

    with ThreadPoolExecutor(max_workers=8) as ex:
        items = [it for chunk in ex.map(one, sources) for it in chunk]
    seen, out = set(), []
    for it in items:
        key = (it.source_id, it.url or it.title[:80])
        if key not in seen:
            seen.add(key)
            if "SNGSP" in it.tickers and not SURGUT.search(it.title):      # «Сургут» в спортивной новости — не компания
                it = it.model_copy(update={"tickers": [t for t in it.tickers if t != "SNGSP"]})
            out.append(it)
    if claim_log:
        claim_log.record(out, now=now)
    return out


def load_demo(path: Path = ROOT / "demo" / "news.json") -> List[NewsItem]:
    """Записанная лента для запуска без сети."""
    return [NewsItem(**r) for r in json.loads(path.read_text(encoding="utf-8"))]


def score(it: NewsItem, now: datetime) -> float:
    """вес источника × тип × свежесть, с надбавкой за названную бумагу."""
    age_h = max(0.0, (now - it.published_at).total_seconds() / 3600)
    return it.source_weight * KIND_MULT.get(it.source_kind, 0.0) * 0.5 ** (age_h / HALF_LIFE_H) * (1.5 if it.tickers else 1.0)


def select(items: List[NewsItem], limit: int = 40, now: Optional[datetime] = None) -> List[NewsItem]:
    """Лучшие заголовки: перепечатки склеиваются, от одной редакции не больше MAX_PER_SOURCE."""
    now = now or datetime.now(timezone.utc)
    ranked = sorted((it for it in items if score(it, now) > 0), key=lambda it: score(it, now), reverse=True)
    per: Dict[str, int] = {}
    titles, out = set(), []
    for it in ranked:
        key = re.sub(r"\W+", " ", it.title.lower()).strip()[:70]
        group = it.source_group or it.source_id
        if key in titles or per.get(group, 0) >= MAX_PER_SOURCE:
            continue
        titles.add(key)
        per[group] = per.get(group, 0) + 1
        out.append(it)
        if len(out) >= limit:
            break
    return out
