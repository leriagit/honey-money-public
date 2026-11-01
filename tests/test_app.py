"""Запись новостей, отбор, оценка и API приложения. Сеть не используется."""
from datetime import datetime, timedelta, timezone

import httpx
from fastapi.testclient import TestClient

from honey_money import strategy_client
from honey_money.app import create_app
from honey_money.news import pipeline
from honey_money.schemas import NewsItem
from honey_money.scorer import score_by_words

NOW = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)


def item(title, source="tass", kind="news", weight=0.7, age_h=0, tickers=("SBER",)):
    return NewsItem(source_id=source, title=title, published_at=NOW - timedelta(hours=age_h), tickers=list(tickers),
                    source_kind=kind, source_weight=weight)


def test_config_builds_sources():
    assert len(pipeline.build_sources()) == 10


def test_select_prefers_fresh_and_merges_reprints():
    got = pipeline.select([item("Старая новость", age_h=48), item("Свежая новость"), item("Свежая новость!", source="rbc")], now=NOW)
    assert [i.title for i in got] == ["Свежая новость", "Старая новость"]


def test_select_limits_one_source():
    got = pipeline.select([item(f"Заголовок {'x' * i}") for i in range(pipeline.MAX_PER_SOURCE + 5)], now=NOW)
    assert len(got) == pipeline.MAX_PER_SOURCE


def test_collect_drops_ticker_from_non_market_news():
    class Src:
        id = "tass"

        def fetch(self, since):
            return [item("Клуб из Сургута вышел в финал по водному поло", tickers=("SNGSP",)), item("Прибыль Сбербанка выросла")]

    got = {i.title: i.tickers for i in pipeline.collect([Src()])}
    assert got["Клуб из Сургута вышел в финал по водному поло"] == [] and got["Прибыль Сбербанка выросла"] == ["SBER"]


def test_words_score_range():
    res = score_by_words([item("Сбербанк рост прибыль")], ["SBER"])
    assert -1 <= res["SBER"]["score"] <= 1 and res["SBER"]["method"] == "words"


def test_offline_app_end_to_end(monkeypatch):
    monkeypatch.delenv("STRATEGY_API_URL", raising=False)
    monkeypatch.delenv("STRATEGY_API_KEY", raising=False)
    for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    c = TestClient(create_app(offline=True))
    assert c.get("/api/health").json()["strategy"] == "replay"
    assert c.get("/api/cards").json()
    r = c.post("/api/decision/SBER").json()
    assert r["source"] == "replay" and "не живая" in r["notice"]
    assert c.post("/api/decision/AAPL").status_code == 404
    assert "Honey Money" in c.get("/").text


def test_live_call_and_fallback(monkeypatch):
    monkeypatch.setenv("STRATEGY_API_URL", "https://api.test")
    monkeypatch.setenv("STRATEGY_API_KEY", "k")

    def ok(req):
        assert req.headers["x-api-key"] == "k" and req.url.path == "/v1/decide"
        return httpx.Response(200, json={"ticker": "SBER", "action": "no_entry", "source": "live"})

    assert strategy_client.decide("SBER", [], transport=httpx.MockTransport(ok))["source"] == "live"
    down = strategy_client.decide("SBER", [], transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    assert down["source"] == "replay" and "не ответил" in down["notice"]
