"""
다중 시드 재현성 검증 — 시드별 백테스트 → 평균±표준편차
=========================================================
입력: ensemble_wf_predictions_seed{S}.csv  (여러 시드)
각 시드 예측을 자본배분 B·고정 임계값 0.54/0.49로 백테스트 →
전체 OOS 지표(누적/CAGR/Sharpe/MDD)를 시드별로 모아 mean±std.
"""
import sys
import glob
import re
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
DB   = ROOT / "db" / "adv_ai_nasdaq.db"
# 원고 채택 재정렬 방식
GBM_MIN, N_CAND, TOPN, HOLD, TRADING_DAYS, COST = 0.50, 8, 3, 3, 252, 0.001


def load_prices():
    con = sqlite3.connect(DB)
    px = pd.read_sql("SELECT ticker,date,open,adj_close,close,nasdaq_close FROM stock_prices", con)
    con.close()
    px["date"] = pd.to_datetime(px["date"]); px = px.sort_values(["ticker", "date"])
    px["adj_open"] = px["open"] * px["adj_close"] / px["close"]      # 배당·분할 조정 시가
    px["ret_t3"] = (px.groupby("ticker")["adj_open"].shift(-(HOLD+1))
                    / px.groupby("ticker")["adj_open"].shift(-1) - 1)  # 다음날 시가→+3일 시가
    return px


def day_return(picks):
    r = picks["ret_t3"].to_numpy(dtype=float)
    if len(r) == 0:
        return 0.0
    return r.sum() / TOPN - COST * (len(r) / TOPN)


def backtest(sig):
    dates = np.sort(sig["date"].unique())
    daily = {}
    for d, g in sig.groupby("date"):
        pool = g[g["prob_lgb"] >= GBM_MIN]
        cand = pool.sort_values("prob_lgb", ascending=False).head(N_CAND)
        daily[d] = cand.sort_values("prob_lstm", ascending=False).head(TOPN).dropna(subset=["ret_t3"])
    grid = pd.DatetimeIndex(dates)
    curves = []
    for off in range(HOLD):
        eq, cur = {}, 1.0
        for d in dates[off::HOLD]:
            cur *= (1 + day_return(daily[d])); eq[d] = cur
        curves.append(pd.Series(eq).reindex(grid).ffill().fillna(1.0))
    port = pd.concat(curves, axis=1).mean(axis=1)
    days = (port.index[-1] - port.index[0]).days
    total = port.iloc[-1] - 1
    cagr = port.iloc[-1] ** (365.25/days) - 1
    mdd = (port / port.cummax() - 1).min()
    rets = port.iloc[::HOLD].pct_change().dropna()
    sharpe = rets.mean()/rets.std(ddof=1)*np.sqrt(TRADING_DAYS/HOLD)
    return dict(total=total, cagr=cagr, sharpe=sharpe, mdd=mdd)


def main():
    files = sorted(glob.glob(str(ROOT / "ensemble_wf_predictions_seed*.csv")))
    if not files:
        print("[대기] 시드별 예측 CSV 없음 — 시드 런 먼저 실행")
        return
    px = load_prices()
    rows = []
    for f in files:
        seed = int(re.search(r"seed(\d+)", f).group(1))
        sig = pd.read_csv(f, parse_dates=["date"])
        sig = sig.merge(px[["ticker", "date", "ret_t3"]], on=["ticker", "date"], how="left")
        m = backtest(sig); m["seed"] = seed
        rows.append(m)
    df = pd.DataFrame(rows).sort_values("seed")

    print("=" * 62)
    print(f"  다중 시드 재현성 검증  ({len(df)} seeds)  자본배분 B · 비용 0.1%")
    print("=" * 62)
    print(f"  {'seed':<6}{'누적수익':>10}{'CAGR':>10}{'Sharpe':>9}{'MDD':>10}")
    print("-" * 62)
    for _, r in df.iterrows():
        print(f"  {int(r.seed):<6}{r.total*100:>+9.1f}%{r.cagr*100:>+9.1f}%"
              f"{r.sharpe:>9.2f}{r.mdd*100:>+9.1f}%")
    print("-" * 62)

    def ms(col, pctsign=True):
        mu, sd = df[col].mean(), df[col].std(ddof=1)
        if pctsign:
            return f"{mu*100:+.1f}% ± {sd*100:.1f}%"
        return f"{mu:.2f} ± {sd:.2f}"
    print(f"  {'평균±SD':<6}{'':>1}")
    print(f"    누적수익 : {ms('total')}")
    print(f"    CAGR     : {ms('cagr')}")
    print(f"    Sharpe   : {ms('sharpe', False)}")
    print(f"    MDD      : {ms('mdd')}")
    print("-" * 62)
    print(f"    변동계수(CAGR): {df['cagr'].std(ddof=1)/df['cagr'].mean()*100:.1f}%  "
          f"(낮을수록 시드에 안정적)")
    print("=" * 62)


if __name__ == "__main__":
    main()
