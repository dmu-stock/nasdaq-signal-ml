"""동적 단건 뉴스 수집 (요약 체인용).

기존 news_crawler.py = 유니버스 배치 + 헤드라인만.
여기 = 티커 1개의 최신 N건을 title+summary+url까지 살려서 반환.
"""
from __future__ import annotations
from dataclasses import dataclass
import yfinance as yf


@dataclass
class NewsDoc:
    title: str
    summary: str
    date: str
    source: str
    url: str


def fetch_recent_news(ticker: str, limit: int = 10) -> list[NewsDoc]:
    try:
        items = yf.Ticker(ticker).news or []
    except Exception as error:
        print(f"[news_fetch] {ticker} 수집 실패: {error}")
        return []

    docs: list[NewsDoc] = []
    for item in items[:limit]:
        content = item.get("content", {})
        title = content.get("title") or ""
        if not title:
            continue
        summary = content.get("summary") or content.get("description") or ""
        pub_date = content.get("pubDate") or ""
        docs.append(NewsDoc(
            title=title,
            summary=summary,
            date=pub_date[:10],                 # ISO "2026-09-01T..." → "2026-09-01"
            source=(content.get("provider") or {}).get("displayName", ""),
            url=(content.get("canonicalUrl") or {}).get("url", ""),
        ))
    return docs
