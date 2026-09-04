"""매수시그널 CSV 리더 (경량, torch 불필요) — API·챗봇이 이걸 사용한다.

signal_service.save_signals()가 로컬에서 계산해 저장한 날짜별 CSV를 읽어 픽을 산출한다.
클라우드(torch 없는 환경)에서도 동작한다.
- 저장: data/signals/signal_YYYY-MM-DD.csv (signal_service, 로컬)
- 읽기: 오늘 파일 우선, 없으면 가장 최근 파일 (as_of로 실제 날짜 표시)
"""
from __future__ import annotations
import glob
import os
from datetime import date

import pandas as pd

SIGNAL_DIR = "data/signals"
GBM_MIN, TOP_N_GBM = 0.50, 8


def _resolve_path(day: str | None = None) -> str | None:
    """오늘 파일 우선, 없으면 가장 최근 파일 경로."""
    day = day or str(date.today())
    today_path = os.path.join(SIGNAL_DIR, f"signal_{day}.csv")
    if os.path.exists(today_path):
        return today_path
    files = sorted(glob.glob(os.path.join(SIGNAL_DIR, "signal_*.csv")))
    return files[-1] if files else None


def _load(day: str | None = None) -> pd.DataFrame | None:
    path = _resolve_path(day)
    if not path:
        return None
    df = pd.read_csv(path)
    if "extra" in df.columns:  # CSV의 True/False 문자열을 진짜 bool로 (문자열 truthy 함정 방지)
        df["extra"] = df["extra"].astype(str).str.lower().isin(["true", "1"])
    return df


def get_buy_picks(top_n: int = 3) -> dict:
    """오늘(또는 최신) 매수 top-N. GBM≥0.5 상위8 → LSTM 상위N. VIX≥30이면 매수 중단."""
    df = _load()
    if df is None or df.empty:
        return {"picks": [], "vix": None, "as_of": None,
                "reason": "시그널 파일이 없습니다. 로컬에서 갱신하세요."}
    as_of = str(df["date"].iloc[0])
    vix = float(df["vix"].iloc[0]) if "vix" in df.columns else None
    if vix is not None and vix >= 30:
        return {"picks": [], "vix": vix, "as_of": as_of,
                "reason": f"VIX {vix:.1f} 극공포 → 매수 중단"}
    cand = df[df["prob_lgb"] >= GBM_MIN].sort_values("prob_lgb", ascending=False).head(TOP_N_GBM)
    picks = cand.sort_values("prob_lstm", ascending=False).head(top_n)
    cols = ["ticker", "prob_lgb", "prob_lstm", "final_prob"]
    if "extra" in picks.columns:
        cols.append("extra")   # 확장종목(참고) 구분용
    return {
        "picks": picks[cols].to_dict("records"),
        "vix": vix,
        "as_of": as_of,
        "reason": "" if not picks.empty else f"GBM {GBM_MIN} 이상 없음 — 현금 권장",
    }


def get_ticker_signal(ticker: str) -> dict:
    """특정 종목의 시그널 점수 (CSV에서)."""
    df = _load()
    if df is None or df.empty:
        return {"ticker": ticker.upper(), "error": "시그널 파일 없음"}
    row = df[df["ticker"] == ticker.upper()]
    if row.empty:
        return {"ticker": ticker.upper(), "error": "미갱신 또는 학습 유니버스 밖"}
    r = row.iloc[0]
    return {
        "ticker": ticker.upper(),
        "prob_lgb": float(r["prob_lgb"]),
        "prob_lstm": float(r["prob_lstm"]),
        "final_prob": float(r["final_prob"]),
        "buy": float(r["prob_lgb"]) >= GBM_MIN,
        "extra": bool(r["extra"]) if "extra" in df.columns else False,  # 확장종목(참고)
        "as_of": str(r["date"]),
    }
