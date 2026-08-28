"""
(c) 검증셋 기준 임계값 재산출 — soft-leak 제거
==================================================
각 폴드의 val에서 Top3 적중률 최대화하는 (TH_lgb, TH_lstm)를 그리드서치로
선택(=test 미참조) → 그 폴드의 test에 적용 → 백테스트.
고정 임계값(0.54/0.49, test 분포 참고해 정한 값)과 비교.

입력: ensemble_wf_predictions.csv (test), ensemble_wf_val_predictions.csv (val)
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
TEST = ROOT / "ensemble_wf_predictions.csv"
VAL  = ROOT / "ensemble_wf_val_predictions.csv"
DB   = ROOT / "db" / "adv_ai_nasdaq.db"

TOPN, HOLD, TRADING_DAYS, COST = 3, 3, 252, 0.001
GRID_LGB  = [0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
GRID_LSTM = [0.45, 0.47, 0.49, 0.51, 0.53]
MIN_ENTRIES_PER_DAY = 1.0   # val에서 하루 평균 최소 진입수 (너무 빡센 임계 방지)


def add_final(df):
    df["final_prob"] = (2 * df["prob_lgb"] * df["prob_lstm"]
                        / (df["prob_lgb"] + df["prob_lstm"] + 1e-9))
    return df


def val_hit(vfold, th_lgb, th_lstm):
    """val에서 가드레일+Top3 적중률과 하루평균 진입수."""
    hits, n_days, n_ent = [], 0, 0
    for _, g in vfold.groupby("date"):
        n_days += 1
        f = g[(g["prob_lgb"] >= th_lgb) & (g["prob_lstm"] >= th_lstm)]
        top = f.sort_values("final_prob", ascending=False).head(TOPN)
        if not top.empty:
            hits.extend(top["label"].tolist())
            n_ent += len(top)
    hit = np.mean(hits) if hits else 0.0
    return hit, (n_ent / n_days if n_days else 0.0)


def tune_thresholds(vfold):
    best = None
    for tl in GRID_LGB:
        for ts in GRID_LSTM:
            hit, epd = val_hit(vfold, tl, ts)
            if epd < MIN_ENTRIES_PER_DAY:
                continue
            key = (hit, epd)      # 적중률 우선, 동률이면 진입수 많은 쪽
            if best is None or key > best[0]:
                best = (key, tl, ts, hit, epd)
    if best is None:             # 제약 만족 못하면 가장 느슨하게
        return GRID_LGB[0], GRID_LSTM[0], *val_hit(vfold, GRID_LGB[0], GRID_LSTM[0])
    return best[1], best[2], best[3], best[4]


# ── 백테스트 (자본배분 B, 3코호트) ──
def day_return(picks):
    r = picks["ret_t3"].to_numpy(dtype=float)
    if len(r) == 0:
        return 0.0
    return r.sum() / TOPN - COST * (len(r) / TOPN)


def build_equity(sig, th_map):
    """th_map: fold -> (th_lgb, th_lstm). 폴드별 임계값 적용."""
    dates = np.sort(sig["date"].unique())
    daily = {}
    for d, g in sig.groupby("date"):
        fold = g["fold"].iloc[0]
        tl, ts = th_map[fold]
        f = g[(g["prob_lgb"] >= tl) & (g["prob_lstm"] >= ts)]
        daily[d] = f.sort_values("final_prob", ascending=False).head(TOPN).dropna(subset=["ret_t3"])
    grid = pd.DatetimeIndex(dates)
    curves = []
    for off in range(HOLD):
        eq, cur = {}, 1.0
        for d in dates[off::HOLD]:
            cur *= (1 + day_return(daily[d]))
            eq[d] = cur
        curves.append(pd.Series(eq).reindex(grid).ffill().fillna(1.0))
    return pd.concat(curves, axis=1).mean(axis=1)


def metrics(port):
    days = (port.index[-1] - port.index[0]).days
    total = port.iloc[-1] - 1
    cagr = port.iloc[-1] ** (365.25 / days) - 1
    mdd = (port / port.cummax() - 1).min()
    rets = port.iloc[::HOLD].pct_change().dropna()
    sharpe = rets.mean() / rets.std(ddof=1) * np.sqrt(TRADING_DAYS / HOLD)
    return total, cagr, sharpe, mdd


def main():
    if not VAL.exists():
        print(f"[대기] {VAL.name} 없음 — ensemble_wf.py 재실행 필요")
        return
    test = add_final(pd.read_csv(TEST, parse_dates=["date"]))
    val  = add_final(pd.read_csv(VAL,  parse_dates=["date"]))

    con = sqlite3.connect(DB)
    px = pd.read_sql("SELECT ticker,date,adj_close FROM stock_prices", con); con.close()
    px["date"] = pd.to_datetime(px["date"]); px = px.sort_values(["ticker", "date"])
    px["ret_t3"] = px.groupby("ticker")["adj_close"].shift(-HOLD) / px["adj_close"] - 1
    test = test.merge(px[["ticker", "date", "ret_t3"]], on=["ticker", "date"], how="left")

    folds = list(dict.fromkeys(test["fold"]))

    # 폴드별 val 튜닝
    print("=" * 74)
    print("  (c) 검증셋 기준 임계값 재산출 — 폴드별 val 그리드서치")
    print("=" * 74)
    print(f"  {'폴드':<22}{'val 선택 (LGBM/LSTM)':<22}{'val적중률':<12}{'val진입/일'}")
    print("-" * 74)
    th_map = {}
    for f in folds:
        vf = val[val["fold"] == f]
        tl, ts, hit, epd = tune_thresholds(vf)
        th_map[f] = (tl, ts)
        print(f"  {f.split('~')[0][:7]+' 폴드':<22}{f'{tl:.2f} / {ts:.2f}':<22}{hit:<12.4f}{epd:.2f}")
    print("-" * 74)

    # 두 세팅 백테스트
    fixed_map = {f: (0.54, 0.49) for f in folds}
    p_fixed = build_equity(test, fixed_map)
    p_val   = build_equity(test, th_map)

    def line(name, port):
        t, c, s, m = metrics(port)
        return f"  {name:<28}{t*100:>+8.1f}%{c*100:>+9.1f}%{s:>8.2f}{m*100:>+9.1f}%"

    print("\n  === 전체 OOS 성적 비교 (자본배분 B, 비용 0.1%) ===")
    print(f"  {'세팅':<28}{'누적수익':>9}{'CAGR':>10}{'Sharpe':>8}{'MDD':>10}")
    print(line("고정 0.54/0.49 (기존)", p_fixed))
    print(line("val 재산출 (soft-leak 제거)", p_val))
    print("=" * 74)


if __name__ == "__main__":
    main()
