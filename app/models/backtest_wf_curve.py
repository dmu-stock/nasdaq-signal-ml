"""
Walk-Forward 자본곡선 차트 (논문 그림용)
==========================================
상단: 전략(앙상블 Top3) vs 나스닥 Buy&Hold 자본곡선 + 4폴드 경계
하단: 언더워터(낙폭) 플롯 — 전략 MDD 시각화
입력: ensemble_wf_predictions.csv (+ db/adv_ai_nasdaq.db)
출력: backtest_wf_curve.png / .pdf (repo 루트)
"""
import os, sys, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# 언어: 환경변수 BAND_LANG = 'ko'(기본) 또는 'en'
LANG = os.environ.get("BAND_LANG", "ko")
plt.rcParams["font.family"] = "Times New Roman" if LANG == "en" else "Batang"  # 논문체 세리프
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.edgecolor"] = "#3a3f45"
plt.rcParams["axes.linewidth"] = 0.9

T = {
    "ko": dict(
        strat="앙상블 Top3 전략", bench="나스닥 Buy & Hold", ylab="누적 자본 (초기=1.0)",
        ddlab="낙폭 %", loss="손실 국면",
        title="그림. 앙상블 Top3 전략의 자본곡선과 낙폭 (Walk-Forward OOS, 중앙값 시드, 비용 0.1%)",
        cap="전체 OOS: 누적 +{tot:.0f}%  ·  CAGR +{cagr:.0f}%  ·  MDD {mdd:.1f}%   |   "
            "나스닥: +{nq:.0f}%   |   회색 점선 = 폴드(재학습) 경계",
    ),
    "en": dict(
        strat="Ensemble Top-3 strategy", bench="Nasdaq Buy & Hold",
        ylab="Cumulative capital (start = 1.0)", ddlab="Drawdown %", loss="Loss regime",
        title="Fig. Equity curve and drawdown of the ensemble Top-3 strategy "
              "(Walk-Forward OOS, median seed, 0.1% cost)",
        cap="Full OOS: cum. +{tot:.0f}%  ·  CAGR +{cagr:.0f}%  ·  MDD {mdd:.1f}%   |   "
            "Nasdaq: +{nq:.0f}%   |   dashed grey = fold (retrain) boundaries",
    ),
}[LANG]

ROOT = Path(__file__).resolve().parents[2]
# 표준 실행 = seed 42 + 결정론 (원고 시드, 재현 가능). 없으면 기본 파일로 폴백.
PRED = ROOT / "ensemble_wf_predictions_seed42.csv"
if not PRED.exists():
    PRED = ROOT / "ensemble_wf_predictions.csv"
DB   = ROOT / "db" / "adv_ai_nasdaq.db"
OUT  = ROOT / ("backtest_wf_curve_en" if LANG == "en" else "backtest_wf_curve")

# 원고 채택 재정렬 방식
GBM_MIN, N_CAND = 0.50, 8
TOPN, HOLD, COST = 3, 3, 0.001

# ── 배색 ──────────────────────────────────────────
C_STRAT = "#e8443b"   # 전략 (레드)
C_BENCH = "#5b6b7a"   # 나스닥 (그레이블루)
C_DD    = "#e8443b"
C_GRID  = "#d9dde2"
C_FOLD  = "#b0b7bf"
C_BAD   = "#f4c8c4"   # 손실 폴드 음영


def load():
    sig = pd.read_csv(PRED, parse_dates=["date"])
    con = sqlite3.connect(DB)
    px = pd.read_sql("SELECT ticker,date,open,adj_close,close,nasdaq_close FROM stock_prices", con)
    con.close()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["ticker", "date"])
    px["adj_open"] = px["open"] * px["adj_close"] / px["close"]      # 배당·분할 조정 시가
    px["ret_t3"] = (px.groupby("ticker")["adj_open"].shift(-(HOLD+1))
                    / px.groupby("ticker")["adj_open"].shift(-1) - 1)  # 다음날 시가→+3일 시가
    sig = sig.merge(px[["ticker", "date", "ret_t3"]], on=["ticker", "date"], how="left")
    return sig, px


def pick(g):
    """재정렬: GBM_MIN 통과 → GBM 상위 N_CAND → LSTM 상위 TOPN."""
    pool = g[g["prob_lgb"] >= GBM_MIN]
    cand = pool.sort_values("prob_lgb", ascending=False).head(N_CAND)
    return cand.sort_values("prob_lstm", ascending=False).head(TOPN)


def day_return(picks):
    """자본배분 B: 3개 슬롯 고정 1/3, 빈 슬롯은 현금 (부족일 보수적)."""
    r = picks["ret_t3"].to_numpy(dtype=float)
    if len(r) == 0:
        return 0.0
    return r.sum() / TOPN - COST * (len(r) / TOPN)


def strategy_equity(sig):
    """오프셋 0/1/2 코호트를 일 단위 계단 자본곡선으로 전개 후 평균 (자본배분 B)."""
    dates = np.sort(sig["date"].unique())
    daily = {d: pick(g) for d, g in sig.groupby("date")}
    grid = pd.DatetimeIndex(dates)
    cohort_curves = []
    for off in range(HOLD):
        tdays = dates[off::HOLD]
        eq, cur = {}, 1.0
        for d in tdays:
            picks = daily[d].dropna(subset=["ret_t3"])
            cur *= (1 + day_return(picks))
            eq[d] = cur                       # 이 거래 실현 후 자본
        s = pd.Series(eq).reindex(grid).ffill().fillna(1.0)
        cohort_curves.append(s)
    port = pd.concat(cohort_curves, axis=1).mean(axis=1)   # 3코호트 평균
    return port


def nasdaq_equity(px, grid):
    nq = px[["date", "nasdaq_close"]].drop_duplicates("date").set_index("date").sort_index()
    nq = nq.reindex(grid).ffill()
    return nq["nasdaq_close"] / nq["nasdaq_close"].iloc[0]


def main():
    sig, px = load()
    port = strategy_equity(sig)
    grid = port.index
    bench = nasdaq_equity(px, grid)

    dd = port / port.cummax() - 1.0
    mdd_i = dd.idxmin()

    folds = list(dict.fromkeys(sig["fold"]))
    fold_bounds = [pd.Timestamp(f.split("~")[0]) for f in folds] + [grid[-1]]
    # 손실 폴드(2024 H2) 구간
    bad0, bad1 = pd.Timestamp("2024-07-01"), pd.Timestamp("2025-01-01")

    fig, (ax, axd) = plt.subplots(
        2, 1, figsize=(11, 6.6), height_ratios=[3, 1], sharex=True,
        gridspec_kw={"hspace": 0.08})
    fig.patch.set_facecolor("white")

    # ── 상단: 자본곡선 ──
    ax.set_facecolor("#fbfcfd")
    ax.axvspan(bad0, bad1, color=C_BAD, alpha=0.5, lw=0, zorder=0)
    for b in fold_bounds[1:-1]:
        ax.axvline(b, color=C_FOLD, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.plot(grid, bench.values, color=C_BENCH, lw=1.8, label=T["bench"], zorder=3)
    ax.plot(grid, port.values, color=C_STRAT, lw=2.3, label=T["strat"], zorder=4)

    # 종점 라벨
    ax.annotate(f"×{port.iloc[-1]:.2f}  (+{(port.iloc[-1]-1)*100:.0f}%)",
                xy=(grid[-1], port.iloc[-1]), xytext=(8, 0), textcoords="offset points",
                va="center", color=C_STRAT, fontweight="bold", fontsize=10)
    ax.annotate(f"×{bench.iloc[-1]:.2f}  (+{(bench.iloc[-1]-1)*100:.0f}%)",
                xy=(grid[-1], bench.iloc[-1]), xytext=(8, 0), textcoords="offset points",
                va="center", color=C_BENCH, fontsize=9)
    # MDD 지점
    ax.scatter([mdd_i], [port.loc[mdd_i]], color=C_STRAT, s=28, zorder=5,
               edgecolor="white", linewidth=1)

    # 폴드 라벨 (상단) — 첫 폴드는 범례와 겹치므로 음영 하단에 배치
    fold_names = ["2024 H2", "2025 H1", "2025 H2", "2026 H1"]
    for i, name in enumerate(fold_names):
        mid = fold_bounds[i] + (fold_bounds[i+1] - fold_bounds[i]) / 2
        if i == 0:
            ax.text(mid, 0.10, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8.5, color="#c0392b", fontweight="bold")
            ax.text(mid, 0.045, T["loss"], transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, color="#c0392b", style="italic")
        else:
            ax.text(mid, 0.965, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8.5, color="#7a828b")

    ax.set_ylabel(T["ylab"], fontsize=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"×{v:.1f}"))
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=10,
              edgecolor="#e0e0e0")
    ax.grid(True, color=C_GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title(T["title"], fontsize=12.5, fontweight="normal", pad=12, loc="center")
    ax.set_ylim(0.7, None)

    # ── 하단: 언더워터(낙폭) ──
    axd.set_facecolor("#fbfcfd")
    axd.fill_between(grid, dd.values * 100, 0, color=C_DD, alpha=0.28, lw=0)
    axd.plot(grid, dd.values * 100, color=C_DD, lw=1.2)
    axd.axhline(dd.min()*100, color=C_STRAT, lw=0.9, ls=":", alpha=0.8)
    axd.annotate(f"MDD {dd.min()*100:.1f}%", xy=(mdd_i, dd.min()*100),
                 xytext=(6, 4), textcoords="offset points",
                 color="#c0392b", fontsize=9, fontweight="bold")
    for b in fold_bounds[1:-1]:
        axd.axvline(b, color=C_FOLD, lw=1, ls=(0, (4, 3)))
    axd.set_ylabel(T["ddlab"], fontsize=10)
    axd.grid(True, color=C_GRID, lw=0.6, alpha=0.7)
    axd.set_axisbelow(True)
    for s in ["top", "right"]:
        axd.spines[s].set_visible(False)
    axd.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # 하단 요약 캡션
    fin = port.iloc[-1]
    cagr = fin ** (365.25 / (grid[-1]-grid[0]).days) - 1
    fig.text(0.125, -0.02,
             T["cap"].format(tot=(fin-1)*100, cagr=cagr*100, mdd=dd.min()*100,
                             nq=(bench.iloc[-1]-1)*100),
             fontsize=9, color="#5a626b")

    fig.savefig(str(OUT) + ".png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(str(OUT) + ".pdf", bbox_inches="tight", facecolor="white")
    print(f"저장: {OUT}.png / .pdf")
    print(f"전략 종점 ×{fin:.3f} (+{(fin-1)*100:.1f}%) | CAGR {cagr*100:.1f}% | MDD {dd.min()*100:.1f}%")
    print(f"나스닥 종점 ×{bench.iloc[-1]:.3f} (+{(bench.iloc[-1]-1)*100:.1f}%)")


if __name__ == "__main__":
    main()
