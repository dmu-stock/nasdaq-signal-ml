"""뉴스 데이터 입력 인터페이스.

수집은 `app/collector/news_crawler.py` (다른 팀원) 담당.
이쪽은 그 결과를 NewsItem 으로 받기만 한다.
입력 포맷: dict / pandas Row / CSV → from_any 로 통일.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Iterable

import pandas as pd
from pydantic import BaseModel, Field


@dataclass
class NewsItem:
    id: str
    ticker: str
    title: str
    content: str
    url: str
    source: str
    published_at: datetime  # tz-aware (UTC)

    def to_metadata(self) -> dict:
        return {
            "news_id": self.id,
            "ticker": self.ticker,
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "published_at": self.published_at.astimezone(timezone.utc).isoformat(),
            "published_ts": int(self.published_at.timestamp()),
        }

    def text(self) -> str:
        return f"{self.title}\n\n{self.content}".strip()


class NewsItemIn(BaseModel):
    """FastAPI 요청 바디용 (POST /api/v1/rag/ingest)."""
    ticker: str = Field(..., description="종목 식별자 (예: AAPL)")
    title: str
    published_at: str = Field(..., description="ISO8601 또는 RFC822 또는 unix epoch")
    content: str | None = ""
    url: str | None = ""
    source: str | None = "unknown"
    id: str | None = None


def _hash_id(url: str, title: str) -> str:
    return hashlib.sha1(f"{url}::{title}".encode("utf-8")).hexdigest()[:16]


def _to_datetime(v: Any) -> datetime:
    if isinstance(v, datetime):
        dt = v
    elif isinstance(v, (int, float)):
        dt = datetime.fromtimestamp(float(v), tz=timezone.utc)
    elif isinstance(v, str):
        s = v.strip()
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            try:
                dt = parsedate_to_datetime(s)
            except Exception:
                dt = pd.to_datetime(s, utc=True).to_pydatetime()
    else:
        raise TypeError(f"published_at 변환 불가: {type(v).__name__}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def from_any(row: Any) -> NewsItem:
    """dict / pandas Row / Pydantic / dataclass → NewsItem."""
    if hasattr(row, "model_dump"):  # pydantic v2
        d = row.model_dump()
    elif hasattr(row, "to_dict"):
        d = row.to_dict()
    elif isinstance(row, dict):
        d = dict(row)
    else:
        raise TypeError(f"지원 안 함: {type(row).__name__}")

    lower = {str(k).lower(): v for k, v in d.items()}

    def pick(*keys: str, default: Any = "") -> Any:
        for k in keys:
            if k in lower and lower[k] is not None:
                return lower[k]
        return default

    title = str(pick("title", "headline", default="")).strip()
    content = str(pick("content", "description", "summary", "body", default="")).strip()
    url = str(pick("url", "link", "originallink", default="")).strip()
    ticker = str(pick("ticker", "symbol", default="")).strip()
    source = str(pick("source", default="")).strip() or "unknown"
    published_at = _to_datetime(pick("published_at", "pubdate", "datetime", "date"))

    given_id = pick("id", "news_id", default="")
    item_id = str(given_id).strip() or _hash_id(url, title)

    if not ticker or not title:
        raise ValueError(f"NewsItem 필수 누락 (ticker/title): {d}")

    return NewsItem(
        id=item_id,
        ticker=ticker,
        title=title,
        content=content,
        url=url,
        source=source,
        published_at=published_at,
    )


def from_iter(rows: Iterable[Any]) -> list[NewsItem]:
    return [from_any(r) for r in rows]


def to_dict(item: NewsItem) -> dict:
    d = asdict(item)
    d["published_at"] = item.published_at.isoformat()
    return d
