"""
RankIC / RankICIR 표 — 횡단면 순위 예측력 (주요 결과 지표)
==========================================================
전역 AUC(전종목 이진분류)를 보완하여, 같은 날 종목 간 상대순위 예측력을
RankIC(일별 예측·실현수익 순위상관, Spearman)와 그 정보비율 RankICIR로 측정.

입력: ensemble_wf_predictions_seed42.csv (표준 실행, 결정론적) + db/adv_ai_nasdaq.db
지표:
  RankIC   = mean_t( Spearman( pred_t, fwd5_t ) )   (일별 IC의 평균)
  RankICIR = mean_t(IC) / std_t(IC)                 (정보비율, 안정성)
  양(+)일% = IC>0 인 영업일 비율
모델별(GBM prob_lgb / LSTM prob_lstm) · 폴드별 + 전체를 한 표로.
"""
import sys
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "ensemble_wf_predictions_seed42.csv"
if not PRED.exists():
    PRED = ROOT / "ensemble_wf_predictions.csv"
DB = ROOT / "db" / "adv_ai_nasdaq.db"
FWD = 5           # 실현수익 지평(영업일) — §4.1 서술과 동일
MIN_N = 5         # 순위상관 산출 최소 종목 수


def load():
    sig = pd.read_csv(PRED, parse_dates=["date"])
    con = sqlite3.connect(DB)
    px = pd.read_sql("SELECT ticker,date,adj_close FROM stock_prices", con)
    con.close()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["ticker", "date"])
    px["fwd"] = px.groupby("ticker")["adj_close"].shift(-FWD) / px["adj_close"] - 1
    m = sig.merge(px[["ticker", "date", "fwd"]], on=["ticker", "date"], how="left")
    return m.dropna(subset=["fwd"])


def daily_ic(df, col):
    """날짜별 Spearman 순위상관 시계열."""
    out = []
    for _, g in df.groupby("date"):
        if len(g) < MIN_N:
            continue
        ic = spearmanr(g[col], g["fwd"])[0]
        if not np.isnan(ic):
            out.append(ic)
    return np.asarray(out)


def stats(ic):
    return dict(rankic=ic.mean(), icir=ic.mean() / ic.std() if ic.std() > 0 else np.nan,
                pos=(ic > 0).mean(), n=len(ic))


def fold_label(f):
    y = f.split("~")[0][:7]
    q = {"2024-07": "2024 H2", "2025-01": "2025 H1",
         "2025-07": "2025 H2", "2026-01": "2026 H1"}
    return q.get(y, y)


def main():
    m = load()
    folds = list(dict.fromkeys(m["fold"]))
    models = [("GBM (prob_lgb)", "prob_lgb"), ("Dual-LSTM (prob_lstm)", "prob_lstm")]

    print("=" * 70)
    print(f"  RankIC / RankICIR — 횡단면 순위 예측력  (실현수익 {FWD}영업일, seed 42)")
    print("=" * 70)
    print(f"  {'모델 · 구간':<26}{'RankIC':>10}{'RankICIR':>11}{'양(+)일%':>10}{'일수':>7}")
    print("-" * 70)
    for name, col in models:
        for f in folds:
            s = stats(daily_ic(m[m["fold"] == f], col))
            print(f"  {name+' · '+fold_label(f):<26}{s['rankic']:>+10.4f}"
                  f"{s['icir']:>11.3f}{s['pos']*100:>9.1f}%{s['n']:>7}")
        s = stats(daily_ic(m, col))
        print(f"  {name+' · 전체':<26}{s['rankic']:>+10.4f}"
              f"{s['icir']:>11.3f}{s['pos']*100:>9.1f}%{s['n']:>7}")
        print("-" * 70)
    print("참고: 횡단면 랭킹 문헌(S&P500 등)에서 RankIC는 대략 0.01~0.14 수준으로 보고됨.")
    print("      값의 범위는 유니버스 크기·기간·리밸런싱 주기에 따라 달라짐(직접 비교 유의).")


if __name__ == "__main__":
    main()
