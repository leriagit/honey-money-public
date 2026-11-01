"""
Общий RSS/Atom-парсер. Используется всеми источниками пула с transport rss/atom.

- у источника может быть несколько лент (urls), запасные (fallback_urls —
  только если основные ничего не дали) и ленты через рыночный фильтр (filtered_urls);
- ограничения на домен из news_sources.yaml → domain_limits (DOMAIN_GUARD):
  пауза между запросами, 403/429 → пауза домена 60 мин, 3 сбоя подряд → 30 мин.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from ...schemas import NewsItem
from ..base import NewsSource, market_relevant

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; HoneyMoneyNews/1.0)"


class DomainGuard:
    """Очередь и паузы по доменам. Один экземпляр на процесс (DOMAIN_GUARD)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._host_locks: Dict[str, threading.Lock] = {}
        self._last_ts: Dict[str, float] = {}
        self._paused_until: Dict[str, float] = {}
        self._fails: Dict[str, int] = {}
        self.limits: Dict[str, dict] = {}
        self.failures_before_pause = 3
        self.pause_after_failures_sec = 30 * 60
        self.pause_on_block_sec = 60 * 60
        self.user_agent = DEFAULT_USER_AGENT
        self.timeout: Tuple[float, float] = (3.0, 8.0)

    def configure(self, domain_limits: Optional[dict], defaults: Optional[dict]) -> None:
        self.limits = dict(domain_limits or {})
        d = defaults or {}
        self.failures_before_pause = int(d.get("failures_before_pause", 3))
        self.pause_after_failures_sec = float(d.get("pause_after_failures_min", 30)) * 60
        self.user_agent = d.get("user_agent", DEFAULT_USER_AGENT)
        self.timeout = (float(d.get("timeout_connect_sec", 3)), float(d.get("timeout_total_sec", 8)))

    def _host_lock(self, host: str) -> threading.Lock:
        with self._lock:
            return self._host_locks.setdefault(host, threading.Lock())

    def min_interval(self, host: str) -> float:
        lim = self.limits.get(host) or self.limits.get("default") or {}
        return float(lim.get("min_interval_sec", 0))

    def paused_for(self, host: str) -> float:
        return max(0.0, self._paused_until.get(host, 0.0) - time.time())

    def acquire(self, host: str) -> Optional[threading.Lock]:
        """Дождаться очереди домена. None — домен на паузе, запрос не делать.
        Возвращает захваченный lock домена: освободить через release()."""
        if self.paused_for(host) > 0:
            return None
        lock = self._host_lock(host)
        lock.acquire()
        wait = self._last_ts.get(host, 0.0) + self.min_interval(host) - time.time()
        if wait > 0:
            time.sleep(wait)
        return lock

    def release(self, host: str, lock: threading.Lock, status: Optional[int]) -> None:
        """status: HTTP-код или None при сетевой ошибке."""
        self._last_ts[host] = time.time()
        try:
            if status in (403, 429):
                self._paused_until[host] = time.time() + self.pause_on_block_sec
                self._fails[host] = 0
                logger.warning("news: %s ответил %s — пауза домена %d мин",
                               host, status, int(self.pause_on_block_sec / 60))
            elif status is not None and status < 400:
                self._fails[host] = 0
            else:
                n = self._fails.get(host, 0) + 1
                self._fails[host] = n
                if n >= self.failures_before_pause:
                    self._paused_until[host] = time.time() + self.pause_after_failures_sec
                    self._fails[host] = 0
                    logger.warning("news: %s — %d сбоя подряд, пауза %d мин",
                                   host, n, int(self.pause_after_failures_sec / 60))
        finally:
            lock.release()


DOMAIN_GUARD = DomainGuard()


def fetch_feed(url: str, timeout: Optional[Tuple[float, float]] = None):
    """Скачать ленту с учётом DOMAIN_GUARD и распарсить. None — не получилось/пауза.

    feedparser.parse(url) делает HTTP сам и без таймаута — зависший хост
    навсегда блокировал поток сборщика новостей, поэтому качаем requests-ом.
    """
    import feedparser
    import requests

    host = urlparse(url).hostname or ""
    lock = DOMAIN_GUARD.acquire(host)
    if lock is None:
        logger.debug("news: %s на паузе ещё %.0f с — %s пропущен", host, DOMAIN_GUARD.paused_for(host), url)
        return None
    status: Optional[int] = None
    try:
        r = requests.get(url, timeout=timeout or DOMAIN_GUARD.timeout,
                         headers={"User-Agent": DOMAIN_GUARD.user_agent})
        status = r.status_code
        if status != 200:
            logger.warning("news: HTTP %s %s", status, url)
            return None
        return feedparser.parse(r.content)
    except Exception as e:  # noqa: BLE001 — сетевые ошибки только логируем
        logger.warning("news: fetch failed %s: %s", url, e)
        return None
    finally:
        DOMAIN_GUARD.release(host, lock, status)


def _parse_feed_with_timeout(feedparser, url: str, timeout: float = 10.0):
    """Совместимость со старыми парсерами (moex/alphavantage не используют)."""
    feed = fetch_feed(url, timeout=(min(3.0, timeout), timeout))
    if feed is None:
        raise RuntimeError(f"feed unavailable: {url}")
    return feed


def entry_datetime(entry, tz_fix_hours: float = 0.0) -> Optional[datetime]:
    """Время записи в UTC. tz_fix_hours — поправка для лент с неверным смещением
    (kremlin.ru пишет +04:00 вместо +03:00 → прибавить 1 час)."""
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    return datetime(*parsed[:6], tzinfo=timezone.utc) + timedelta(hours=tz_fix_hours)


def entry_source_title(entry) -> str:
    """Поле источника в лентах Google News (<source>) и Bing News (<News:Source>)."""
    src = entry.get("source")
    if isinstance(src, dict) and src.get("title"):
        return str(src["title"])
    return str(entry.get("news_source") or "")


class RSSSource(NewsSource):
    """Базовая реализация для любого источника с RSS/Atom-лентами."""

    MAX_ENTRIES_PER_FEED = 50

    def _timeout(self) -> Tuple[float, float]:
        extra = self.cfg.extra or {}
        connect, total = DOMAIN_GUARD.timeout
        return (float(extra.get("timeout_connect_sec", connect)),
                float(extra.get("timeout_total_sec", total)))

    def fetch(self, since: datetime) -> List[NewsItem]:
        try:
            import feedparser  # noqa: F401 — отложенный импорт, чтобы не падал тест на схемах
        except ImportError:
            logger.warning("feedparser not installed — RSS источник %s пропущен", self.id)
            return []

        items: List[NewsItem] = []
        seen: set = set()
        for url in self.cfg.all_urls():
            items += self._fetch_url(url, since, seen, force_market_filter=False)
        for url in self.cfg.filtered_urls or []:
            items += self._fetch_url(url, since, seen, force_market_filter=True)
        if not items:
            for url in self.cfg.fallback_urls or []:
                items += self._fetch_url(url, since, seen, force_market_filter=False)
                if items:
                    break
        return items

    def _fetch_url(self, url: str, since: datetime, seen: set,
                   force_market_filter: bool) -> List[NewsItem]:
        feed = fetch_feed(url, timeout=self._timeout())
        if feed is None:
            return []
        items: List[NewsItem] = []
        for entry in feed.entries[: self.MAX_ENTRIES_PER_FEED]:
            try:
                title = entry.get("title", "") or ""
                summary = entry.get("summary", "") or entry.get("description", "") or ""
                link = entry.get("link")
                key = link or title[:80]
                if key in seen:
                    continue
                pub = entry_datetime(entry, self.cfg.tz_fix_hours) or datetime.now(timezone.utc)
                if pub < since:
                    continue
                if self._filtered_out(entry, title, summary):
                    continue
                if (force_market_filter or self.cfg.market_filter) and not market_relevant(f"{title} {summary}"):
                    continue
                seen.add(key)
                items.append(self.parse_item(title=title, summary=summary, url=link, published_at=pub))
            except Exception as e:  # noqa: BLE001
                logger.debug("rss entry parse error in %s: %s", self.id, e)
        return items

    # ─── Фильтры из конфига ───

    def _filtered_out(self, entry, title: str, summary: str) -> bool:
        """True — запись отбрасывается. Фильтры опциональны и мягкие:
        нет поля в конфиге — нет фильтра; нет категорий у записи —
        keep_categories её не режет (лучше шум, чем молчание)."""
        cats = [t.get("term", "") or "" for t in (entry.get("tags") or [])]
        drop = self.cfg.drop_categories
        if drop and cats and any(d.lower() in c.lower() for c in cats for d in drop):
            return True
        keep = self.cfg.keep_categories
        if keep and cats and not any(k.lower() in c.lower() for c in cats for k in keep):
            return True
        if self.cfg.dropped(f"{title} {summary}"):
            return True
        kws = self.cfg.keep_keywords
        if kws:
            text = f"{title} {summary}".lower()
            if not any(k.lower() in text for k in kws):
                return True
        allowed = self.cfg.keep_if_source_in
        if allowed:
            src = entry_source_title(entry)
            ok = any(a.lower() == src.lower() for a in allowed) if src else \
                any(title.rstrip().lower().endswith(f"- {a.lower()}") for a in allowed)
            if not ok:
                return True
        return False
