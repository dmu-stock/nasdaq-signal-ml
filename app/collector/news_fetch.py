"""동적 단건 뉴스 수집 (요약 체인용).

기존 news_crawler.py = 유니버스 배치 + 헤드라인만.
여기 = 티커 1개의 최신 N건을 title+summary+url까지 살려서 반환.
"""
from __future__ import annotations
from dataclasses import dataclass
import yfinance as yf

@dataclass
class NewsDoc:
    title :str
    summary :str
    date :str
    source :str
    url :str

def fetch_recent_news(ticker,limit : int = 10) -> list[NewsDoc]:
    try:
        items = yf.Ticker(ticker).news or []
    except Exception as e:
        print(f"[news_fetch] {ticker} 수집 실패: {e}")
        return []

    docs: list[NewsDoc] = []
    for item in items[:limit]:
        c = item.get("content", {})
        title = c.get("title") or ""
        if not title:
            continue
        summary = c.get("summary") or c.get("description") or ""
        pub = c.get("pubDate") or ""
        docs.append(NewsDoc(
            title=title,
            summary=summary,
            date=pub[:10],                      # ISO "2026-09-01T..." → "2026-09-01"
            source=(c.get("provider") or {}).get("displayName", ""),
            url=(c.get("canonicalUrl") or {}).get("url", ""),
        ))
    return docs
