"""
Базовый класс NewsSource.

Контракт:
- fetch(since) -> list[NewsItem]
- классифицирует ru_relevance и trump_mention внутри parse_item()
- никогда не падает наружу — ошибки логируются и возвращается []
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from ..schemas import NewsItem

logger = logging.getLogger(__name__)


class NewsFetchError(Exception):
    """Сбой при получении новостей. НЕ выбрасывается наружу — только логируется."""


# Ключевые слова, по которым отмечаем РФ-релевантность.
# Тикеры голубых фишек + сектора + страна/события.
RU_KEYWORDS = [
    # тикеры
    "sber", "сбер", "gazp", "газпром", "lkoh", "лукойл", "rosn", "роснефт",
    "vtbr", "втб", "yandex", "yndx", "ydex", "яндекс", "plzl", "полюс",
    "novatek", "новатэк", "nvtk", "gmkn", "норникель", "norilsk",
    "magnit", "магнит", "mgnt", "alrs", "алроса", "aeroflot", "аэрофлот", "aflt",
    "x5", "пятёрочка", "пятерочка", "chmf", "северсталь", "severstal",
    "nlmk", "нлмк", "moex", "мосбиржа", "sngs", "сургут", "сургутнефтегаз",
    "mtss", "мтс", "pikk", "пик",
    # рынок / макро
    "moscow exchange", "московская биржа", "russian stock", "russian equities",
    "ruble", "рубль", "rosneft", "central bank of russia", "цб рф", "bank of russia",
    "kremlin", "кремль", "putin", "путин",
    # сырьё связанное
    "urals", "юралс", "ormuz", "ормуз", "strait of hormuz", "nord stream", "северный поток",
    # макро
    "russia", "россия", "russian", "moscow", "москва", "санкции", "sanctions",
]

TRUMP_KEYWORDS = ["trump", "трамп", "elon musk", "маск", "jd vance", "вэнс",
                  "tucker carlson", "карлсон"]

# Очень примитивный sentiment-лексикон (на старте). Заменяется RuBERT/BGE-M3.
POSITIVE_WORDS = [
    "rally", "gain", "surge", "jump", "rise", "boost", "beat", "record high",
    "soar", "outperform", "strong", "growth", "recovery", "deal", "agreement",
    "ралл", "рост", "взлет", "рекорд", "опередил", "укрепил", "увеличил",
]
NEGATIVE_WORDS = [
    "fall", "drop", "plunge", "crash", "loss", "miss", "downgrade", "ban",
    "sanction", "decline", "slump", "warning", "default", "investigation",
    "падение", "обвал", "потеря", "провал", "санкции", "запрет", "снижение",
    "расследование", "нарушение",
]


def heuristic_sentiment(text: str) -> float:
    """
    Грубая baseline-оценка. После подключения RuBERT эта функция не используется
    для русскоязычных новостей.
    Возвращает [-1, 1].
    """
    if not text:
        return 0.0
    t = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in t)
    neg = sum(1 for w in NEGATIVE_WORDS if w in t)
    if pos + neg == 0:
        return 0.0
    score = (pos - neg) / (pos + neg)
    return max(-1.0, min(1.0, score))


def detect_ru_relevance(text: str) -> float:
    """Доля совпадений ключевых слов РФ в тексте → [0, 1]."""
    if not text:
        return 0.0
    t = text.lower()
    hits = sum(1 for kw in RU_KEYWORDS if kw in t)
    if hits == 0:
        return 0.0
    # Сатурация: 1 совпадение → 0.4, 3+ → 1.0
    return min(1.0, 0.4 + 0.2 * (hits - 1))


# Рыночный словарь для общих потоков (government.ru, kremlin.ru, @tass_agency,
# @bbbreaking …): запись проходит, если в ней есть тикер или одно из этих слов.
MARKET_KEYWORDS = [
    "ставк", "рубл", "курс", "валют", "нефт", "газ", "спг", "уголь", "металл",
    "золот", "санкц", "бюджет", "налог", "пошлин", "экспорт", "импорт", "дивиденд",
    "акци", "облигац", "офз", "биржа", "биржи", "банк", "инфляц", "ввп", "минфин",
    "указ", "постановлен", "распоряжен", "национализац", "конфиск", "приватизац",
    "тариф", "демпфер", "бензин", "топлив", "электроэнерг", "госкомпан", "госкорпорац",
    "инвестиц", "кредит", "ипотек", "опек", "ормуз", "экономик", "промышлен",
    "oil", "sanction", "tariff", "ruble", "opec",
]


def market_relevant(text: str) -> bool:
    """Есть ли в тексте тикер из universe или рыночное слово."""
    if not text:
        return False
    if detect_tickers(text):
        return True
    t = text.lower()
    return any(kw in t for kw in MARKET_KEYWORDS)


def detect_trump_mention(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(kw in t for kw in TRUMP_KEYWORDS)


# Тикер-mention helper (точный, без false-positive на коротких словах)
TICKER_MAP = {
    "SBER": ["sber", "сбер", "сбербанк", "sberbank"],
    "GAZP": ["gazp", "газпром", "gazprom"],
    "LKOH": ["lkoh", "лукойл", "lukoil"],
    "ROSN": ["rosn", "роснефт", "rosneft"],
    "VTBR": ["vtbr", "втб", "vtb"],
    "YDEX": ["yandex", "yndx", "ydex", "яндекс"],
    "PLZL": ["plzl", "полюс", "polyus"],
    "T":    ["т-банк", "tinkoff", "тинькофф"],
    "NVTK": ["novatek", "новатэк", "nvtk"],
    "X5":   ["x5", "пятёрочка", "пятерочка", "перекрёсток"],
    "GMKN": ["gmkn", "норникель", "norilsk", "nornickel"],
    "MGNT": ["magnit", "магнит", "mgnt"],
    "ALRS": ["alrosa", "алроса", "alrs"],
    "AFLT": ["aeroflot", "аэрофлот", "aflt"],
    "CHMF": ["severstal", "северсталь", "chmf"],
    "NLMK": ["nlmk", "нлмк"],
    "MOEX": ["moex", "мосбиржа", "moscow exchange"],
    "SNGSP": ["сургут", "surgut", "sngs"],
    "MTSS": ["мтс", "mts ", "mtss"],
    "PIKK": ["пик", "pik group", "pikk"],
}


def detect_tickers(text: str) -> List[str]:
    if not text:
        return []
    t = text.lower()
    found = []
    for ticker, kws in TICKER_MAP.items():
        if any(kw in t for kw in kws):
            found.append(ticker)
    return found


# ───────────────────────── Базовый класс ─────────────────────────


@dataclass
class SourceConfig:
    id: str
    name: str
    weight: float
    requires_key: bool = False
    env_key: Optional[str] = None
    rss_url: Optional[str] = None
    ru_specialized: bool = False
    trump_priority: bool = False
    trust_tier: str = "verified"
    regulator_triggers: Optional[List[str]] = None
    # username Telegram-канала для generic-источника
    channel: Optional[str] = None
    # Фильтры содержимого ленты
    keep_categories: Optional[List[str]] = None
    drop_categories: Optional[List[str]] = None
    keep_keywords: Optional[List[str]] = None
    extra: Optional[dict] = None
    # Тип источника, ленты, дубли, редакции
    kind: str = "news"
    urls: Optional[List[str]] = None
    fallback_urls: Optional[List[str]] = None
    filtered_urls: Optional[List[str]] = None   # ленты, которые идут через market_filter
    keep_if_source_in: Optional[List[str]] = None
    market_filter: bool = False
    duplicate_of: Optional[str] = None
    group: Optional[str] = None
    tz_fix_hours: float = 0.0
    drop_keywords: Optional[List[str]] = None   # рутина источника: запись с этими словами отбрасывается

    def dropped(self, text: str) -> bool:
        if not self.drop_keywords or not text:
            return False
        low = text.lower()
        return any(k.lower() in low for k in self.drop_keywords)

    @property
    def group_key(self) -> str:
        """Ключ «одного источника»: TG-дубль ленты и каналы одной редакции схлопываются."""
        return self.group or self.duplicate_of or self.id

    def all_urls(self) -> List[str]:
        urls = list(self.urls or [])
        if self.rss_url and self.rss_url not in urls:
            urls.insert(0, self.rss_url)
        return urls


class NewsSource(ABC):
    """
    Базовый класс источника. Конкретные классы переопределяют fetch().
    Sentiment/ru_relevance/tickers заполняем в parse_item() общей логикой,
    если источник не отдаёт свою разметку.
    """

    def __init__(self, config: SourceConfig) -> None:
        self.cfg = config
        self.id = config.id

    # ---- основной метод ----
    @abstractmethod
    def fetch(self, since: datetime) -> List[NewsItem]:
        """Получить новости с момента `since`. Должен НЕ выбрасывать наружу."""
        ...

    # ---- общая обработка ----
    def parse_item(
        self,
        title: str,
        summary: Optional[str],
        url: Optional[str],
        published_at: datetime,
        explicit_sentiment: Optional[float] = None,
        explicit_tickers: Optional[List[str]] = None,
        explicit_ru: Optional[float] = None,
        raw_score: Optional[float] = None,
    ) -> NewsItem:
        full_text = f"{title or ''} {summary or ''}"

        sentiment = explicit_sentiment if explicit_sentiment is not None \
            else heuristic_sentiment(full_text)
        ru = explicit_ru if explicit_ru is not None else detect_ru_relevance(full_text)
        if self.cfg.ru_specialized:
            # официальные РФ (ЦБ, Мосбиржа, Правительство, Кремль) — всегда про рынок РФ
            ru = max(ru, 1.0 if self.cfg.kind == "official" else 0.5)
        tickers = explicit_tickers or detect_tickers(full_text)
        trump = detect_trump_mention(full_text)

        # На trump-priority источниках поднимаем порог детекта
        if self.cfg.trump_priority and trump:
            ru = max(ru, 0.3)

        # Регуляторные триггеры — ЦБ, Кремль, MOEX
        regulator_event = False
        if self.cfg.regulator_triggers:
            text_lower = full_text.lower()
            regulator_event = any(t.lower() in text_lower for t in self.cfg.regulator_triggers)

        return NewsItem(
            source_id=self.id,
            title=title or "",
            summary=summary,
            url=url,
            published_at=published_at,
            tickers=tickers,
            sentiment=float(max(-1.0, min(1.0, sentiment))),
            ru_relevance=float(max(0.0, min(1.0, ru))),
            trump_mention=trump,
            regulator_event=regulator_event,
            raw_score=raw_score,
            source_kind=self.cfg.kind,
            source_weight=self.cfg.weight,
            source_group=self.cfg.group_key,
        )

    @staticmethod
    def safe_fetch(fn, default=None):
        """Декоратор-как-функция — оборачивает fetch, никогда не падает наружу."""
        try:
            return fn()
        except Exception as e:
            logger.warning("news source error: %s", e)
            return default if default is not None else []
