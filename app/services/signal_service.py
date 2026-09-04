"""ML 매수시그널 서비스 (챗봇 도구용).

inference_pipeline.py(스크립트)를 함수+캐싱으로 재작성.
- print/exit 제거 → 반환값으로
- 모델 1회 로드(캐시), 시그널 하루 1회 계산(첫 호출 캐시)
"""
from __future__ import annotations
from datetime import date
from functools import lru_cache
import os

import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.preprocessing import StandardScaler

from app.config.config import GBM_FEATURE_COLS, LSTM_FEATURE_COLS, TICKERS
from app.models.lstm_model import DualLSTMModel
from app.collector.price_yfinance import fetch_all_stocks_price_data
from app.features.processor import FeatureProcessorGBM
from app.features.processor_lstm import FeatureProcessorLSTM

GBM_MIN, TOP_N_GBM = 0.50, 8
SEQ_LEN_20, SEQ_LEN_60 = 20, 60

# 학습 유니버스(41) 외에 앱에서 시그널을 낼 추가 종목 (재학습 없이 추론만, out-of-distribution)
EXTRA_TICKERS = ["SNOW", "SOFI", "RIVN", "ABNB", "SHOP","MSTR","QCOM","MDB","IREN","TSM","MRVL","LITE"]
ALL_TICKERS = TICKERS + EXTRA_TICKERS


@lru_cache(maxsize=1)
def _load_models():
    """모델·스케일러 로드 (프로세스당 1회)."""
    lgb = joblib.load("artifacts/models/best_lgbm_model.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load("artifacts/models/best_multi_input_lstm.pt", map_location=device)
    lstm = DualLSTMModel(ckpt["num_features"]).to(device)
    lstm.load_state_dict(ckpt["model_state_dict"])
    lstm.eval()
    scalers = joblib.load("artifacts/models/ticker_scalers.pkl")
    return lgb, lstm, scalers, device


def _compute_signals() -> dict:
    """전 종목 추론 → {ok, vix, signals[], reason}. print·exit 없음."""
    lgb_model, lstm_model, scalers, device = _load_models()

    df_raw = fetch_all_stocks_price_data(tickers=ALL_TICKERS, period="2y")
    if df_raw.empty:
        return {"ok": False, "reason": "시장 데이터 수집 실패", "signals": []}

    vix_now = float(df_raw["vix"].iloc[-1])
    # VIX 극공포 가드레일은 리더(signal_store)에서 적용 — 여기선 전 종목을 항상 계산·저장한다.

    gbm_proc, lstm_proc = FeatureProcessorGBM(), FeatureProcessorLSTM()
    df_gbm = gbm_proc.calc_technical_indicators(df_raw.copy(), is_inference=True).replace([np.inf, -np.inf], np.nan)
    df_lstm = lstm_proc.calc_technical_indicators(df_raw.copy(), is_inference=True).replace([np.inf, -np.inf], np.nan)

    results = []
    for ticker in ALL_TICKERS:
        tg = df_gbm[df_gbm["ticker"] == ticker].sort_values("date")
        if tg.empty:
            continue
        prob_lgb = lgb_model.predict_proba(tg[GBM_FEATURE_COLS].iloc[[-1]])[0][1]

        tl = df_lstm[df_lstm["ticker"] == ticker].sort_values("date")
        if len(tl) < SEQ_LEN_60:
            continue
        sc = scalers.get(ticker)
        if sc is None:                                       # 학습에 없던 종목: 즉석 스케일러 fit
            sc = StandardScaler().fit(tl[LSTM_FEATURE_COLS])
        seq20 = sc.transform(pd.DataFrame(tl[LSTM_FEATURE_COLS].iloc[-SEQ_LEN_20:].values, columns=LSTM_FEATURE_COLS))
        seq60 = sc.transform(pd.DataFrame(tl[LSTM_FEATURE_COLS].iloc[-SEQ_LEN_60:].values, columns=LSTM_FEATURE_COLS))
        t20 = torch.tensor(seq20, dtype=torch.float32).unsqueeze(0).to(device)
        t60 = torch.tensor(seq60, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            prob_lstm = float(torch.sigmoid(lstm_model(t20, t60)).cpu().item())

        final_prob = 2 * (prob_lgb * prob_lstm) / (prob_lgb + prob_lstm + 1e-9)
        results.append({"ticker": ticker, "prob_lgb": round(float(prob_lgb), 4),
                        "prob_lstm": round(prob_lstm, 4), "final_prob": round(final_prob, 4),
                        "extra": ticker not in TICKERS})   # 학습 유니버스 밖(확장종목) 표시

    return {"ok": True, "vix": round(vix_now, 2), "signals": results,
            "date": str(date.today()), "reason": ""}


@lru_cache(maxsize=1)
def _cached(day: str) -> dict:          # ← 첫 호출 캐시 (day가 키)
    return _compute_signals()


def get_signals() -> dict:
    """오늘 시그널. 하루 1회만 계산(첫 호출 느림), 이후 캐시. 날짜 바뀌면 재계산."""
    return _cached(str(date.today()))


def get_buy_picks(top_n: int = 3) -> dict:
    """오늘 매수 top-N (GBM≥0.5 상위8 → LSTM 상위N)."""
    data = get_signals()
    if not data["ok"] or not data["signals"]:
        return {"picks": [], "reason": data.get("reason", "시그널 없음"), "vix": data.get("vix")}
    df = pd.DataFrame(data["signals"])
    cand = df[df["prob_lgb"] >= GBM_MIN].sort_values("prob_lgb", ascending=False).head(TOP_N_GBM)
    picks = cand.sort_values("prob_lstm", ascending=False).head(top_n)
    return {"picks": picks.to_dict("records"), "vix": data["vix"],
            "reason": "" if not picks.empty else f"GBM {GBM_MIN} 이상 없음 — 현금 권장"}


def get_ticker_signal(ticker: str) -> dict:
    """특정 종목 시그널 점수."""
    for s in get_signals().get("signals", []):
        if s["ticker"] == ticker.upper():
            return {**s, "buy": s["prob_lgb"] >= GBM_MIN}
    return {"ticker": ticker.upper(), "error": "학습 유니버스에 없거나 데이터 부족"}


def save_signals(dir_path: str = "data/signals") -> str:
    """모델 추론 → 날짜별 CSV 저장 (data/signals/signal_YYYY-MM-DD.csv). 로컬에서 실행(torch 필요)."""
    data = _compute_signals()
    if not data.get("signals"):
        return f"저장 안 함: {data.get('reason', '시그널 없음')}"
    day = data["date"]
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, f"signal_{day}.csv")
    df = pd.DataFrame(data["signals"])
    df.insert(0, "date", day)
    df["vix"] = data["vix"]
    df.to_csv(path, index=False, encoding="utf-8")
    return f"저장: {path} ({len(df)}종목, VIX {data['vix']}, {day})"