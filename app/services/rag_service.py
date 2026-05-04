"""RAG 진입점.

라우터(`app/api/v1/endpoints.py`) / 모델(`app/models/predictor.py`) 가 호출하는 고수준 API.

  - ingest(items)        : 임베딩 + 감성점수 (벡터DB + 캐시 영속)
  - features(ticker, ...) : 일별 sentiment 피처 (Sentiment_lag1, Sentiment_rolling_3)
  - search(query, ...)   : RAG 시맨틱 검색
"""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Iterable, Sequence

import pandas as pd
from langchain_core.documents import Document

from app.core.config import Settings, get_settings
from app.database.chroma_db import NewsVectorStore
from app.services.feature_builder import add_rolling_3, build_daily_sentiment
from app.services.schema import NewsItem, from_any
from app.services.sentiment_chain import SentimentScorer


class RAGService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.vs = NewsVectorStore(self.settings)
        self.scorer = SentimentScorer(self.settings)

    # ── 1) 입력 ──
    def ingest(
        self,
        rows: Iterable[NewsItem | dict],
        score: bool = True,
    ) -> dict:
        """뉴스 ingest. dict / NewsItem 둘 다 받음."""
        items: list[NewsItem] = []
        for r in rows:
            items.append(r if isinstance(r, NewsItem) else from_any(r))

        added = self.vs.add(items)
        scored = self.scorer.score_batch(items, show_progress=False) if score else []
        return {"received": len(items), "indexed": added, "scored": len(scored)}

    # ── 2) 피처 ──
    def features(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        docs = self.vs.fetch_by_window(ticker, start, end)
        records = self._docs_to_scored_records(docs)
        daily = build_daily_sentiment(records, market=self.settings.market)
        return add_rolling_3(daily)

    def latest_sentiment(self, ticker: str, end: datetime) -> dict:
        """end 시점 기준 최신 거래일의 sentiment_lag1 / rolling_3 단일 dict 반환.

        모델 (`predictor.py`) 의 추론 시점에 호출하기 좋은 형태.
        """
        from datetime import timedelta
        start = end - timedelta(days=10)  # rolling_3 계산용 여유
        df = self.features(ticker, start, end)
        if df.empty:
            return {
                "ticker": ticker,
                "sentiment_lag1": 0.0,
                "sentiment_rolling_3": 0.0,
                "n_articles": 0,
            }
        last = df.iloc[-1]
        return {
            "ticker": ticker,
            "trade_date": str(last["trade_date"].date()),
            "sentiment_lag1": float(last["sentiment_lag1"]),
            "sentiment_rolling_3": float(last["sentiment_rolling_3"]),
            "n_articles": int(last["n_articles"]),
        }

    # ── 3) RAG 검색 ──
    def search(
        self,
        query: str,
        ticker: str | None = None,
        k: int = 5,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[dict]:
        results = self.vs.search(query, ticker=ticker, k=k, start=start, end=end)
        out = []
        for doc, dist in results:
            m = doc.metadata or {}
            out.append({
                "score": float(dist),
                "title": m.get("title"),
                "url": m.get("url"),
                "source": m.get("source"),
                "published_at": m.get("published_at"),
                "ticker": m.get("ticker"),
                "snippet": doc.page_content[:300],
            })
        return out

    # ── 내부 ──
    def _docs_to_scored_records(self, docs: Sequence[Document]) -> list[dict]:
        records: list[dict] = []
        for d in docs:
            news_id = (d.metadata or {}).get("news_id")
            if not news_id:
                continue
            cached = self.scorer.get_cached(news_id)
            if cached is None:
                continue
            records.append(cached)
        return records


@lru_cache
def get_rag_service() -> RAGService:
    """FastAPI Depends 용 싱글톤."""
    return RAGService()
