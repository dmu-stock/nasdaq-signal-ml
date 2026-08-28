"""
5시드 밴드 자본곡선 (논문 그림용)
====================================
5개 시드 각각의 자본곡선을 그린 뒤:
  · min~max 밴드(음영) = 시드에 따른 흔들림
  · 중앙값(굵은 선)      = 대표 경로
  · 나스닥 Buy&Hold      = 벤치마크
"모든 시드가 항상 나스닥 위" 를 한 장에.
입력: ensemble_wf_predictions_seed{S}.csv
출력: backtest_wf_band.png / .pdf
"""
import os, glob, re, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, FixedLocator

# 언어: 환경변수 BAND_LANG = 'ko'(기본) 또는 'en'
LANG = os.environ.get("BAND_LANG", "ko")
plt.rcParams["font.family"] = "Times New Roman" if LANG == "en" else "Batang"  # 논문체 세리프
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.edgecolor"] = "#3a3f45"
plt.rcParams["axes.linewidth"] = 0.9

T = {
    "ko": dict(
        band="5시드 min~max 범위", median="전략 중앙값 (5시드)", bench="나스닥 Buy & Hold",
        ylab="누적 자본 (초기=1.0)", median_tag="중앙값", loss="2024 H2\n손실국면",
        title="그림. 앙상블 Top3 전략의 다중시드 자본곡선과 나스닥 벤치마크 비교",
        sub="Walk-Forward OOS  ·  {d0}–{d1}  ·  5 seeds  ·  거래비용 0.1%",
        cap="5시드 종점 ×{lo:.1f}~×{hi:.1f} (중앙값 ×{md:.1f}, 로그축)  ·  절대 배수는 유니버스 "
            "구성에 크게 기인(한계 참조)  ·  시드 간 순위·위험조정 안정  |  나스닥 ×{nq:.2f}(참고)",
    ),
    "en": dict(
        band="5-seed min–max range", median="Strategy median (5 seeds)", bench="Nasdaq Buy & Hold",
        ylab="Cumulative capital (start = 1.0)", median_tag="Median", loss="2024 H2\nLoss regime",
        title="Fig. Multi-seed equity curves of the ensemble Top-3 strategy vs. Nasdaq benchmark",
        sub="Walk-Forward OOS  ·  {d0}–{d1}  ·  5 seeds  ·  0.1% cost",
        cap="5-seed endpoints ×{lo:.1f}–×{hi:.1f} (median ×{md:.1f}, log axis)  ·  absolute multiples "
            "largely reflect universe composition (see limitations)  ·  rank & risk-adjusted stable  |  Nasdaq ×{nq:.2f} (ref.)",
    ),
}[LANG]

ROOT = Path(__file__).resolve().parents[2]
DB   = ROOT / "db" / "adv_ai_nasdaq.db"
OUT  = ROOT / ("backtest_wf_band_en" if LANG == "en" else "backtest_wf_band")
GBM_MIN, N_CAND, TOPN, HOLD, COST = 0.50, 8, 3, 3, 0.001   # 원고 채택 재정렬

C_STRAT, C_BAND, C_BENCH = "#e8443b", "#f3b0ab", "#5b6b7a"
C_GRID, C_FOLD, C_BAD = "#d9dde2", "#b0b7bf", "#f4c8c4"


def prices():
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
    return 0.0 if len(r) == 0 else r.sum()/TOPN - COST*(len(r)/TOPN)


def equity(sig, grid):
    dates = np.sort(sig["date"].unique())
    daily = {}
    for d, g in sig.groupby("date"):
        pool = g[g["prob_lgb"] >= GBM_MIN]
        cand = pool.sort_values("prob_lgb", ascending=False).head(N_CAND)
        daily[d] = cand.sort_values("prob_lstm", ascending=False).head(TOPN).dropna(subset=["ret_t3"])
    curves = []
    for off in range(HOLD):
        eq, cur = {}, 1.0
        for d in dates[off::HOLD]:
            cur *= (1 + day_return(daily[d])); eq[d] = cur
        curves.append(pd.Series(eq).reindex(pd.DatetimeIndex(dates)).ffill().fillna(1.0))
    return pd.concat(curves, axis=1).mean(axis=1).reindex(grid).ffill()


def main():
    files = sorted(glob.glob(str(ROOT / "ensemble_wf_predictions_seed*.csv")))
    px = prices()
    grid = pd.DatetimeIndex(np.sort(pd.read_csv(files[0], parse_dates=["date"])["date"].unique()))

    seeds = {}
    for f in files:
        s = int(re.search(r"seed(\d+)", f).group(1))
        sig = pd.read_csv(f, parse_dates=["date"]).merge(
            px[["ticker", "date", "ret_t3"]], on=["ticker", "date"], how="left")
        seeds[s] = equity(sig, grid)
    E = pd.DataFrame(seeds).sort_index(axis=1)      # columns = seeds
    lo, hi, mid = E.min(axis=1), E.max(axis=1), E.median(axis=1)

    nq = px[["date", "nasdaq_close"]].drop_duplicates("date").set_index("date").sort_index().reindex(grid).ffill()
    bench = nq["nasdaq_close"] / nq["nasdaq_close"].iloc[0]

    fold_bounds = [pd.Timestamp(x) for x in ["2025-01-01", "2025-07-01", "2026-01-01"]]
    bad0, bad1 = pd.Timestamp("2024-07-01"), pd.Timestamp("2025-01-01")

    fig, ax = plt.subplots(figsize=(11, 6.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("#fbfcfd")
    ax.axvspan(bad0, bad1, color=C_BAD, alpha=0.45, lw=0, zorder=0)
    for b in fold_bounds:
        ax.axvline(b, color=C_FOLD, lw=1, ls=(0, (4, 3)), zorder=1)

    # 밴드 + 개별 시드(옅게) + 중앙값
    ax.fill_between(grid, lo.values, hi.values, color=C_BAND, alpha=0.6, lw=0,
                    zorder=2, label=T["band"])
    for s in E.columns:
        ax.plot(grid, E[s].values, color=C_STRAT, lw=0.7, alpha=0.35, zorder=3)
    ax.plot(grid, mid.values, color=C_STRAT, lw=2.6, zorder=5, label=T["median"])
    ax.plot(grid, bench.values, color=C_BENCH, lw=1.9, zorder=4, label=T["bench"])

    # 종점 라벨
    ax.annotate(f"{T['median_tag']} ×{mid.iloc[-1]:.1f}", xy=(grid[-1], mid.iloc[-1]),
                xytext=(8, 0), textcoords="offset points", va="center",
                color=C_STRAT, fontweight="bold", fontsize=9.5)
    ax.annotate(f"×{bench.iloc[-1]:.2f}", xy=(grid[-1], bench.iloc[-1]),
                xytext=(8, 0), textcoords="offset points", va="center",
                color=C_BENCH, fontsize=9)

    # 폴드 라벨 — 각 폴드 구간의 정중앙에 배치
    end = grid[-1]
    labelspots = [(bad0+(bad1-bad0)/2, T["loss"], "#8a1f16"),
                  (fold_bounds[0]+(fold_bounds[1]-fold_bounds[0])/2, "2025 H1", "#5a626b"),
                  (fold_bounds[1]+(fold_bounds[2]-fold_bounds[1])/2, "2025 H2", "#5a626b"),
                  (fold_bounds[2]+(end-fold_bounds[2])/2, "2026 H1", "#5a626b")]
    for x, name, col in labelspots:
        is_bad = "2024" in name
        y = 0.55 if is_bad else 0.955          # 손실국면 라벨은 음영 중간 빈공간
        va = "center" if is_bad else "top"
        ax.text(x, y, name, transform=ax.get_xaxis_transform(), ha="center", va=va,
                fontsize=9 if is_bad else 8.5, color=col,
                fontweight="bold" if is_bad else "normal",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, alpha=0.85, lw=1) if is_bad else None)

    ax.set_ylabel(T["ylab"] + ("  ·  log" ), fontsize=10)
    # 로그 스케일: 극단적 절대수익을 완만하게, 벤치마크 대비 배수 구조 부각
    ax.set_yscale("log")
    _tk = [1, 2, 3, 5, 10, 20, 30]
    ax.yaxis.set_major_locator(FixedLocator(_tk))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"×{v:g}"))
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=9.5, edgecolor="#e0e0e0")
    ax.grid(True, color=C_GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(labelsize=9.5, colors="#2b2f34")
    d0s, d1s = grid[0].strftime("%Y.%m"), grid[-1].strftime("%Y.%m")
    ax.set_title(T["title"], fontsize=13, fontweight="normal", pad=16, loc="center")
    ax.text(0.5, 1.015, T["sub"].format(d0=d0s, d1=d1s),
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9.5, color="#5a626b", style="italic")
    ax.set_ylim(0.8, float(hi.max()) * 1.25)

    fin = E.iloc[-1]
    fig.text(0.125, 0.005,
             T["cap"].format(lo=fin.min(), hi=fin.max(), md=fin.median(), nq=bench.iloc[-1]),
             fontsize=8.5, color="#5a626b")

    fig.savefig(str(OUT)+".png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(str(OUT)+".pdf", bbox_inches="tight", facecolor="white")
    print(f"저장: {OUT}.png/.pdf")
    print(f"종점 배수 seed별: {[round(v,2) for v in sorted(fin.values)]}")
    print(f"나스닥 종점 ×{bench.iloc[-1]:.2f}  | 최저 시드도 나스닥 초과? "
          f"{fin.min() > bench.iloc[-1]}")


if __name__ == "__main__":
    main()
