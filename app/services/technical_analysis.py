"""기술적 분석 — 지지/저항/추천 매수가/추세 계산.

yfinance 의 일봉 OHLCV 를 기반으로 다음을 계산:
  - current_price       : 최근 종가
  - pivot               : 클래식 피벗 포인트 (H+L+C)/3
  - resistance_1, _2    : 1차/2차 저항선 (피벗 R1, 20일 고가 중 큰 값)
  - support             : 지지선 (피벗 S1, 20일 저가 중 작은 값)
  - recommended_buy     : 추천 매수가 (현재가와 지지선 사이의 중간 — 보수적 진입)
  - sma_5/10/20/60      : 이동평균
  - trend               : "up" | "down" | "sideways"
  - confidence_score    : 신호 정합도 (지표들이 얼마나 같은 방향인지) 0~1
  - probability_up      : 휴리스틱 상승확률 (sentiment 합산 가능)
  - rsi_14              : RSI(14) (참고용)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _safe_round(x: float, digits: int = 2) -> float:
    try:
        return float(np.round(float(x), digits))
    except Exception:
        return 0.0


def fetch_ohlcv(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """yfinance 일봉 데이터. 비어있을 수 있음."""
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    # 표준화
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    return df


def compute_rsi(closes: pd.Series, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    delta = closes.diff().dropna()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    if pd.isna(val):
        return 50.0
    return float(val)


def compute_levels(df: pd.DataFrame) -> Optional[dict]:
    """피벗 + 스윙 고저 결합 레벨."""
    if df is None or df.empty:
        return None

    last = df.iloc[-1]
    H, L, C = float(last["High"]), float(last["Low"]), float(last["Close"])
    P = (H + L + C) / 3.0
    R1 = 2 * P - L
    S1 = 2 * P - H
    R2 = P + (H - L)

    # 최근 20일 스윙
    recent20 = df.tail(20)
    swing_high_20 = float(recent20["High"].max())
    swing_low_20 = float(recent20["Low"].min())
    swing_high_60 = float(df.tail(60)["High"].max())

    resistance_1 = max(R1, swing_high_20)
    resistance_2 = max(R2, swing_high_60)
    if resistance_2 <= resistance_1:
        resistance_2 = resistance_1 * 1.045
    support = min(S1, swing_low_20)
    # 추천 매수가: 현재가와 지지선 사이 중간 (≈ 보수 진입)
    recommended_buy = (C + support) / 2.0

    return {
        "current_price": _safe_round(C),
        "pivot": _safe_round(P),
        "resistance_1": _safe_round(resistance_1),
        "resistance_2": _safe_round(resistance_2),
        "support": _safe_round(support),
        "recommended_buy": _safe_round(recommended_buy),
    }


def compute_signal(df: pd.DataFrame, sentiment_score: float = 0.0) -> dict:
    """이동평균 정합 + RSI + sentiment 로 신호/확률/신뢰도 산정 (휴리스틱)."""
    closes = df["Close"].astype(float)
    last = float(closes.iloc[-1])

    sma_5 = float(closes.tail(5).mean())
    sma_10 = float(closes.tail(10).mean())
    sma_20 = float(closes.tail(20).mean())
    sma_60 = float(closes.tail(60).mean()) if len(closes) >= 60 else float(closes.mean())

    rsi = compute_rsi(closes, 14)

    score = 0
    flags = []
    if last > sma_5:   score += 1; flags.append("close>SMA5")
    if last > sma_20:  score += 1; flags.append("close>SMA20")
    if last > sma_60:  score += 1; flags.append("close>SMA60")
    if sma_5 > sma_20: score += 1; flags.append("SMA5>SMA20")
    if sma_20 > sma_60: score += 1; flags.append("SMA20>SMA60")
    if rsi > 55:       score += 1; flags.append("RSI>55")
    if rsi < 30:       score -= 1; flags.append("RSI<30 (oversold)")
    if rsi > 75:       score -= 1; flags.append("RSI>75 (overbought)")
    if sentiment_score > 0.15:  score += 1; flags.append("sentiment+")
    if sentiment_score < -0.15: score -= 1; flags.append("sentiment-")

    # score 범위 대략 -3 ~ +7
    if score >= 5:
        signal = "up"
        probability = 0.85
    elif score >= 3:
        signal = "up"
        probability = 0.70
    elif score >= 1:
        signal = "up"
        probability = 0.58
    elif score >= -1:
        signal = "neutral"
        probability = 0.50
    elif score >= -3:
        signal = "down"
        probability = 0.35
    else:
        signal = "down"
        probability = 0.20

    # 신뢰도: 모든 지표가 같은 방향이면 높음. abs(score) / max_score
    confidence = min(0.95, 0.55 + abs(score) * 0.06)

    # trend
    if sma_5 > sma_20 > sma_60:
        trend = "up"
    elif sma_5 < sma_20 < sma_60:
        trend = "down"
    else:
        trend = "sideways"

    return {
        "signal": signal,
        "probability_up": _safe_round(probability, 3),
        "confidence": _safe_round(confidence, 3),
        "trend": trend,
        "sma_5":  _safe_round(sma_5),
        "sma_10": _safe_round(sma_10),
        "sma_20": _safe_round(sma_20),
        "sma_60": _safe_round(sma_60),
        "rsi_14": _safe_round(rsi, 1),
        "flags": flags,
    }


def full_technical(ticker: str, sentiment_score: float = 0.0) -> dict:
    """레벨 + 신호 + 메타. 데이터 없으면 빈 dict."""
    df = fetch_ohlcv(ticker, period="6mo")
    if df.empty:
        return {}
    levels = compute_levels(df) or {}
    signal = compute_signal(df, sentiment_score=sentiment_score)
    return {**levels, **signal, "as_of": str(df.index[-1].date())}
