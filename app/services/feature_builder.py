"""감성 점수 → 일별 피처.

  - Sentiment_lag1[T]      = mean(scores in (T-1 close, T close])
  - Sentiment_rolling_3[T] = mean( Sentiment_lag1[T-2..T] )

장마감 시각:
  - us: 16:00 ET (DST 무시 근사)
  - kr: 15:30 KST
"""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Iterable

import pandas as pd
import pytz


KST = pytz.timezone("Asia/Seoul")
ET = pytz.timezone("America/New_York")


def market_close_dt(d: datetime, market: str) -> datetime:
    if market == "kr":
        local = KST.localize(datetime.combine(d.date(), time(15, 30)))
    elif market == "us":
        local = ET.localize(datetime.combine(d.date(), time(16, 0)))
    else:
        raise ValueError(f"unknown market: {market}")
    return local.astimezone(timezone.utc)


def trading_day_window(t_day: datetime, market: str) -> tuple[datetime, datetime]:
    end = market_close_dt(t_day, market)
    start = end - timedelta(days=1)
    return start, end


def _assign_trade_date(ts_utc: pd.Timestamp, market: str) -> pd.Timestamp:
    """published_at(UTC) 가 속하는 거래일 T 를 반환.

    ts 가 (close[T-1], close[T]] 사이면 T.
    """
    if market == "kr":
        local = ts_utc.tz_convert(KST)
        target = (
            local.date()
            if (local.hour, local.minute) <= (15, 30)
            else (local + pd.Timedelta(days=1)).date()
        )
    elif market == "us":
        local = ts_utc.tz_convert(ET)
        target = (
            local.date()
            if (local.hour, local.minute) <= (16, 0)
            else (local + pd.Timedelta(days=1)).date()
        )
    else:
        raise ValueError(f"unknown market: {market}")
    return pd.Timestamp(target)


def build_daily_sentiment(
    sentiment_records: Iterable[dict],
    market: str,
) -> pd.DataFrame:
    """감성 레코드 → 일별 평균 (sentiment_lag1).

    레코드 스키마: {id, ticker, published_at(iso), score, ...}
    반환 컬럼: trade_date, ticker, sentiment_lag1, n_articles
    """
    df = pd.DataFrame(list(sentiment_records))
    if df.empty:
        return pd.DataFrame(
            columns=["trade_date", "ticker", "sentiment_lag1", "n_articles"]
        )

    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df["trade_date"] = df["published_at"].apply(
        lambda ts: _assign_trade_date(ts, market)
    )

    grouped = (
        df.groupby(["trade_date", "ticker"], as_index=False)
        .agg(sentiment_lag1=("score", "mean"), n_articles=("score", "size"))
        .sort_values(["ticker", "trade_date"])
        .reset_index(drop=True)
    )
    return grouped


def add_rolling_3(daily: pd.DataFrame) -> pd.DataFrame:
    """ticker 별 sentiment_lag1 의 3-거래일 rolling 평균 추가."""
    if daily.empty:
        out = daily.copy()
        out["sentiment_rolling_3"] = pd.Series(dtype="float64")
        return out

    parts = []
    for _, sub in daily.groupby("ticker", as_index=False):
        sub = sub.sort_values("trade_date").copy()
        sub["sentiment_rolling_3"] = (
            sub["sentiment_lag1"].rolling(window=3, min_periods=1).mean()
        )
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    return out[
        ["trade_date", "ticker", "sentiment_lag1", "sentiment_rolling_3", "n_articles"]
    ]
