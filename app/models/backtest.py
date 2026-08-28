"""
백테스트 (방식 B, 1차) — 앙상블 Top3 · 3일 후 종가 청산
=========================================================
설계 확정본
  [선정] 매 영업일 final_prob 내림차순 Top3
         (가드레일: LGBM >= 0.54 & LSTM >= 0.49 통과 종목 중)
  [진입] 당일 종가(adj_close), 자본 균등 배분
  [청산] 3영업일 후 종가 (return_t3 실현)   ← 청산규칙 A
  [비용] 왕복 0.1% / 0.2% 각각 산출
  [지표] CAGR · Sharpe(연율화) · MDD · 누적수익 vs 나스닥 buy&hold

방식 B(근사): 3영업일 논오버랩 거래. 시작오프셋 0/1/2 세 코호트를
평균내 시작일 의존성을 제거한다 (자본 중첩·암묵 레버리지 없음).
"""
import sys
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows cp949 콘솔 대응
except Exception:
    pass

# --- 경로 (이 파일 기준 repo 루트 = app의 상위) --------------------
ROOT = Path(__file__).resolve().parents[2]        # C:\develop\dmu_adv_ai
CSV  = ROOT / "ensemble_prediction_result.csv"
DB   = ROOT / "db" / "adv_ai_nasdaq.db"           # 41종목 나스닥 유니버스

LGBM_THRESHOLD = 0.54
LSTM_THRESHOLD = 0.49
TOPN           = 3
HOLD           = 3            # 3영업일 보유
TRADING_DAYS   = 252


# --- 데이터 로드 --------------------------------------------------
def load_data():
    sig = pd.read_csv(CSV, parse_dates=["date"])

    con = sqlite3.connect(DB)
    px = pd.read_sql(
        "SELECT ticker, date, adj_close, nasdaq_close FROM stock_prices", con
    )
    con.close()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["ticker", "date"])

    # t+3 종가 수익률 (라벨 계산부와 동일 정의: shift(-3))
    px["ret_t3"] = px.groupby("ticker")["adj_close"].shift(-HOLD) / px["adj_close"] - 1

    sig = sig.merge(
        px[["ticker", "date", "ret_t3"]], on=["ticker", "date"], how="left"
    )
    return sig, px


# --- Top3 선정 (가드레일 필터 후) ---------------------------------
def pick_topn(day_group):
    filt = day_group[
        (day_group["prob_lgb"]  >= LGBM_THRESHOLD)
        & (day_group["prob_lstm"] >= LSTM_THRESHOLD)
    ].sort_values("final_prob", ascending=False)
    return filt.head(TOPN)


# --- 지표 계산 ----------------------------------------------------
def metrics_from_trades(trade_returns, calendar_days):
    """trade_returns: 각 거래(HOLD일 보유)의 순수익률 시퀀스."""
    tr = np.asarray(trade_returns, dtype=float)
    equity = np.cumprod(1 + tr)
    total_return = equity[-1] - 1 if len(equity) else 0.0

    years = calendar_days / 365.25
    cagr = equity[-1] ** (1 / years) - 1 if len(equity) and years > 0 else np.nan

    # Sharpe: HOLD일(3영업일) 주기 수익률 → 연율화 √(252/HOLD)
    if tr.std(ddof=1) > 0 and len(tr) > 1:
        sharpe = tr.mean() / tr.std(ddof=1) * np.sqrt(TRADING_DAYS / HOLD)
    else:
        sharpe = np.nan

    peak = np.maximum.accumulate(equity) if len(equity) else np.array([1.0])
    mdd = ((equity - peak) / peak).min() if len(equity) else 0.0

    return dict(
        total_return=total_return, cagr=cagr, sharpe=sharpe, mdd=mdd,
        n_trades=len(tr), final_equity=equity[-1] if len(equity) else 1.0,
    )


def run_strategy(sig, roundtrip_cost):
    """방식 B: 오프셋 0/1/2 세 코호트 평균."""
    all_dates = np.sort(sig["date"].unique())
    daily_pick = {d: pick_topn(g) for d, g in sig.groupby("date")}

    span_days = (pd.Timestamp(all_dates[-1]) - pd.Timestamp(all_dates[0])).days

    cohort_metrics = []
    pooled_trades = []
    for offset in range(HOLD):
        trade_days = all_dates[offset::HOLD]
        trs = []
        for d in trade_days:
            picks = daily_pick[d]
            picks = picks.dropna(subset=["ret_t3"])   # 청산가 없는 말미 제외
            if picks.empty:
                trs_r = 0.0                            # 신호 0 → 현금 보유
            else:
                gross = picks["ret_t3"].mean()         # 균등배분 = 단순평균
                trs_r = gross - roundtrip_cost         # 왕복비용 차감
            trs.append(trs_r)
        cohort_metrics.append(metrics_from_trades(trs, span_days))
        pooled_trades.extend(trs)

    # 세 코호트 지표 평균
    avg = {k: np.nanmean([m[k] for m in cohort_metrics])
           for k in ["total_return", "cagr", "sharpe", "mdd"]}
    avg["n_trades"] = int(np.mean([m["n_trades"] for m in cohort_metrics]))
    # 참고용: 전체 거래 풀의 평균 거래수익 / 승률
    pt = np.asarray(pooled_trades)
    avg["avg_trade_ret"] = pt.mean()
    avg["win_rate"] = (pt > 0).mean()
    return avg


# --- 벤치마크: 나스닥 buy&hold ------------------------------------
def benchmark_nasdaq(px, sig):
    d0, d1 = sig["date"].min(), sig["date"].max()
    nq = (
        px[["date", "nasdaq_close"]].drop_duplicates("date")
        .sort_values("date")
    )
    nq = nq[(nq["date"] >= d0) & (nq["date"] <= d1)].reset_index(drop=True)
    close = nq["nasdaq_close"].to_numpy(dtype=float)
    daily = close[1:] / close[:-1] - 1

    total_return = close[-1] / close[0] - 1
    years = (d1 - d0).days / 365.25
    cagr = (close[-1] / close[0]) ** (1 / years) - 1
    sharpe = daily.mean() / daily.std(ddof=1) * np.sqrt(TRADING_DAYS)
    eq = close / close[0]
    peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min()
    return dict(total_return=total_return, cagr=cagr, sharpe=sharpe, mdd=mdd)


# --- 리포트 -------------------------------------------------------
def pct(x):
    return f"{x*100:+.2f}%" if pd.notna(x) else "  n/a"


def main():
    sig, px = load_data()
    d0, d1 = sig["date"].min().date(), sig["date"].max().date()
    n_days = sig["date"].nunique()

    print("=" * 60)
    print("  백테스트 (방식 B) — 앙상블 Top3 · 3일 후 종가 청산")
    print("=" * 60)
    print(f"기간        : {d0} ~ {d1}  ({n_days} 영업일)")
    print(f"유니버스    : {sig['ticker'].nunique()} 종목 (나스닥)")
    print(f"가드레일    : LGBM>={LGBM_THRESHOLD}, LSTM>={LSTM_THRESHOLD}")
    print(f"청산규칙    : 3영업일 후 종가 (A)")
    print("-" * 60)

    bench = benchmark_nasdaq(px, sig)
    print("\n[벤치마크] 나스닥 Buy & Hold")
    print(f"  누적수익 {pct(bench['total_return'])} | CAGR {pct(bench['cagr'])} "
          f"| Sharpe {bench['sharpe']:.2f} | MDD {pct(bench['mdd'])}")

    print("\n[전략] 앙상블 Top3")
    header = f"  {'비용(왕복)':<10}{'누적수익':>12}{'CAGR':>10}{'Sharpe':>9}{'MDD':>10}{'승률':>9}{'거래수':>7}"
    print(header)
    for cost in [0.001, 0.002]:
        r = run_strategy(sig, cost)
        print(f"  {cost*100:>4.1f}%     "
              f"{pct(r['total_return']):>12}{pct(r['cagr']):>10}"
              f"{r['sharpe']:>9.2f}{pct(r['mdd']):>10}"
              f"{r['win_rate']*100:>8.1f}%{r['n_trades']:>7}")
    print(f"\n  (거래당 평균수익·승률은 3코호트 풀 기준, 비용 전 gross)")
    print("=" * 60)


if __name__ == "__main__":
    main()
