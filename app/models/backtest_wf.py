"""
Walk-Forward 다중폴드 백테스트 — 앙상블 Top3 · 3일 후 종가 청산
================================================================
입력: ensemble_wf_predictions.csv  (ensemble_wf.py가 생성한 4폴드 OOS 예측)
      4폴드를 이어 붙이면 2024-07 ~ 2026-07 연속 OOS 구간.

목적: 단일폴드(+183%) 결과가 '강세장 한 국면 운빨'이 아님을 입증.
      각 폴드가 서로 다른 시장 국면 → 폴드별 + 전체 수익지표 산출.

선정/청산/지표: backtest.py(단일폴드)와 100% 동일 → 직접 비교 가능.
  [선정] 재정렬: GBM>=0.50 통과 → GBM 상위 8 → LSTM 상위 3
  [청산] 3영업일 후 종가 (ret_t3)
  [방식] B — 오프셋 0/1/2 논오버랩 코호트 평균
"""
import sys
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "ensemble_wf_predictions.csv"
DB   = ROOT / "db" / "adv_ai_nasdaq.db"

# 원고 채택 방식(재정렬): GBM_MIN 통과 → GBM 상위 N_CAND 후보 → LSTM 상위 TOPN
GBM_MIN   = 0.50
N_CAND    = 8
TOPN, HOLD, TRADING_DAYS = 3, 3, 252


def load(stops=(0.10, 0.05, 0.03)):
    sig = pd.read_csv(PRED, parse_dates=["date"])
    con = sqlite3.connect(DB)
    px = pd.read_sql("SELECT ticker,date,open,low,adj_close,close,nasdaq_close FROM stock_prices", con)
    con.close()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["ticker", "date"])
    adjf = px["adj_close"] / px["close"]                             # 배당·분할 조정계수
    px["adj_open"] = px["open"] * adjf                              # 조정 시가
    px["adj_low"] = px["low"] * adjf                                # 조정 저가(손절 판정용)
    g = px.groupby("ticker")
    p0 = g["adj_open"].shift(-1)                                    # 진입가 = 다음날(t+1) 시가
    pex = g["adj_open"].shift(-(HOLD + 1))                          # 청산가 = t+4 시가
    px["ret_t3"] = pex / p0 - 1                                     # 무손절 3일 수익
    # 손절: 보유 3일(t+1~t+3) 저가가 진입가 대비 -s 이하로 내려가면 -s에 체결
    low_min = pd.concat([g["adj_low"].shift(-k) for k in range(1, HOLD + 1)], axis=1).min(axis=1)
    for s in stops:
        hit = low_min <= p0 * (1 - s)
        px[f"ret_sl{int(round(s*100))}"] = np.where(hit, -s, px["ret_t3"])
    ret_cols = ["ret_t3"] + [f"ret_sl{int(round(s*100))}" for s in stops]
    sig = sig.merge(px[["ticker", "date"] + ret_cols], on=["ticker", "date"], how="left")
    return sig, px


def pick_topn(g):
    """원고 채택 재정렬 방식: GBM_MIN 통과 → GBM 상위 N_CAND → LSTM 상위 TOPN."""
    pool = g[g["prob_lgb"] >= GBM_MIN]
    cand = pool.sort_values("prob_lgb", ascending=False).head(N_CAND)
    return cand.sort_values("prob_lstm", ascending=False).head(TOPN)


def day_return(picks, roundtrip_cost, alloc, ret_col="ret_t3"):
    """그날 픽들의 포트폴리오 순수익.
    alloc='A': 통과 종목에 자본 균등배분(1종목이면 100%) — 공격적
    alloc='B': 3개 슬롯 고정 1/3, 빈 슬롯은 현금 — 보수적.
    ret_col: 수익 컬럼(ret_t3=무손절, ret_sl5=-5% 손절 등)."""
    rets = picks[ret_col].to_numpy(dtype=float)
    n = len(rets)
    if n == 0:
        return 0.0
    if alloc == "A":
        return rets.mean() - roundtrip_cost                     # 풀투자, 균등
    # B: 채운 슬롯만 1/3씩 투자, 비용도 투자분에만
    deployed = n / TOPN
    return rets.sum() / TOPN - roundtrip_cost * deployed


def build_equity(sig, roundtrip_cost, alloc="B", ret_col="ret_t3"):
    """방식 B(오프셋 0/1/2 코호트)를 일 단위 계단 자본곡선으로 전개 후 평균.
    → 차트(backtest_wf_curve.py)와 100% 동일한 합성 자본곡선.
    alloc은 종목 부족일 자본배분 방식(A/B), ret_col은 수익(손절 여부) 컬럼."""
    dates = np.sort(sig["date"].unique())
    daily = {d: pick_topn(g) for d, g in sig.groupby("date")}
    grid = pd.DatetimeIndex(dates)
    curves, pooled = [], []
    for off in range(HOLD):
        eq, cur = {}, 1.0
        for d in dates[off::HOLD]:
            picks = daily[d].dropna(subset=[ret_col])
            r = day_return(picks, roundtrip_cost, alloc, ret_col)
            cur *= (1 + r)
            eq[d] = cur
            pooled.append(r)
        curves.append(pd.Series(eq).reindex(grid).ffill().fillna(1.0))
    port = pd.concat(curves, axis=1).mean(axis=1)   # 3코호트 평균 = 실제 포트폴리오
    return port, np.asarray(pooled)


def run_strategy(sig, roundtrip_cost, alloc="B", ret_col="ret_t3"):
    """합성 자본곡선에서 직접 지표 산출 (표·그림 완전 일치)."""
    port, pooled = build_equity(sig, roundtrip_cost, alloc, ret_col)
    if len(port) < 2:
        return dict(total_return=0, cagr=np.nan, sharpe=np.nan, mdd=0,
                    win_rate=np.nan, n_trades=0)
    days = (port.index[-1] - port.index[0]).days
    years = days / 365.25
    total_return = port.iloc[-1] - 1
    cagr = port.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    mdd = (port / port.cummax() - 1).min()
    # Sharpe: 합성곡선을 HOLD일 간격으로 샘플링한 주기수익 → 연율화 √(252/HOLD)
    rets = port.iloc[::HOLD].pct_change().dropna()
    sharpe = (rets.mean() / rets.std(ddof=1) * np.sqrt(TRADING_DAYS / HOLD)
              if len(rets) > 1 and rets.std(ddof=1) > 0 else np.nan)
    return dict(total_return=total_return, cagr=cagr, sharpe=sharpe, mdd=mdd,
                win_rate=(pooled > 0).mean(), n_trades=len(pooled) // HOLD)


def benchmark(px, d0, d1):
    nq = px[["date", "nasdaq_close"]].drop_duplicates("date").sort_values("date")
    nq = nq[(nq["date"] >= d0) & (nq["date"] <= d1)].reset_index(drop=True)
    c = nq["nasdaq_close"].to_numpy(float)
    if len(c) < 2:
        return dict(total_return=np.nan, cagr=np.nan, sharpe=np.nan, mdd=np.nan)
    daily = c[1:] / c[:-1] - 1
    years = (pd.Timestamp(d1) - pd.Timestamp(d0)).days / 365.25
    eq = c / c[0]
    peak = np.maximum.accumulate(eq)
    return dict(total_return=c[-1]/c[0]-1, cagr=(c[-1]/c[0])**(1/years)-1,
                sharpe=daily.mean()/daily.std(ddof=1)*np.sqrt(TRADING_DAYS),
                mdd=((eq-peak)/peak).min())


def pct(x):
    return f"{x*100:+.1f}%" if pd.notna(x) else "   n/a"


def row(label, m, extra=""):
    sh = f"{m['sharpe']:.2f}" if pd.notna(m.get("sharpe")) else " n/a"
    return (f"  {label:<22}{pct(m['total_return']):>10}{pct(m['cagr']):>9}"
            f"{sh:>8}{pct(m['mdd']):>9}{extra}")


def main():
    if not PRED.exists():
        print(f"[대기] {PRED.name} 아직 없음 — ensemble_wf.py 완료 후 실행하세요.")
        return
    sig, px = load()
    folds = list(dict.fromkeys(sig["fold"]))  # 순서 보존
    d0, d1 = sig["date"].min(), sig["date"].max()

    print("=" * 74)
    print("  Walk-Forward 다중폴드 백테스트 — 앙상블 Top3 · 3일 종가청산 (비용 0.1%)")
    print("=" * 74)
    print(f"전체 OOS 구간: {d0.date()} ~ {d1.date()}  |  {sig['date'].nunique()} 영업일  |  {len(folds)} 폴드")
    print(f"선정(재정렬): GBM>={GBM_MIN} 통과 → GBM 상위 {N_CAND} → LSTM 상위 {TOPN}")
    print("-" * 74)
    print(f"  {'구간':<22}{'누적수익':>10}{'CAGR':>9}{'Sharpe':>8}{'MDD':>9}{'  vs 나스닥'}")
    print("-" * 74)

    COST = 0.001
    # 폴드별 (각 폴드 = 한 시장 국면)
    for f in folds:
        sub = sig[sig["fold"] == f]
        s = run_strategy(sub, COST)
        b = benchmark(px, sub["date"].min(), sub["date"].max())
        tag = f.replace("2024-", "").replace("2025-", "").replace("2026-", "")
        # 연도 라벨 복원
        yr0 = f.split("~")[0][:7]
        extra = f"   나스닥 {pct(b['total_return'])}"
        print(row(yr0 + " 폴드", s, extra))

    print("-" * 74)
    # 전체 (4폴드 이어붙인 연속 OOS)
    s_all = run_strategy(sig, COST)
    b_all = benchmark(px, d0, d1)
    print(row("★ 전체 OOS", s_all, f"   나스닥 {pct(b_all['total_return'])}"))
    print("-" * 74)
    print(f"  전체 승률 {s_all['win_rate']*100:.1f}%  |  나스닥 B&H: "
          f"CAGR {pct(b_all['cagr'])}, Sharpe {b_all['sharpe']:.2f}, MDD {pct(b_all['mdd'])}")

    # 비용 0.2% 전체만
    s2 = run_strategy(sig, 0.002)
    print(f"  (비용 0.2% 전체: 누적 {pct(s2['total_return'])}, CAGR {pct(s2['cagr'])}, "
          f"Sharpe {s2['sharpe']:.2f}, MDD {pct(s2['mdd'])})")
    print("=" * 74)


if __name__ == "__main__":
    main()
