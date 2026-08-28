"""
벤치마크 비교 + 레짐 분석 (논문 핵심 그림·표)
================================================
동일 유니버스 내 4개 벤치마크와 앙상블을 같은 프레임(3코호트 균등, 비용 0.1%)에서 비교.
  · 앙상블(재정렬)   : GBM≥0.50 → GBM 상위8 → LSTM 상위3
  · 단순 모멘텀      : 과거 20일 수익 상위3  ← 원고 이론(모멘텀 국면 불안정성)의 대상
  · 무작위 3종목     : 선정력 하한선 (20회 평균)
  · 동일가중         : 유니버스 전체 (선정 없음)
  · 나스닥 B&H       : 참고용(시장)

핵심 발견: 2025 H1에서 단순 모멘텀이 붕괴(+8%)하나 앙상블은 견고(+104%)
          → 모멘텀 국면 불안정성을 앙상블이 극복함을 실증.

출력: backtest_wf_bench(_en).png/.pdf + 콘솔 표
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

LANG = os.environ.get("BAND_LANG", "ko")
plt.rcParams["font.family"] = "Times New Roman" if LANG == "en" else "Batang"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.edgecolor"] = "#3a3f45"
plt.rcParams["axes.linewidth"] = 0.9

T = {
    "ko": dict(ens="앙상블(재정렬)", mom="단순 모멘텀", rnd="무작위 3종목",
               ew="동일가중", nq="나스닥",
               ylab="폴드 누적수익 %", xlab="",
               title="그림. 국면별 단순 모멘텀 vs 앙상블 — 2025 H1 모멘텀 붕괴와 앙상블의 강건성",
               note="2025 H1: 단순 모멘텀 붕괴({mom:+.0f}%) ↔ 앙상블 견고({ens:+.0f}%)"),
    "en": dict(ens="Ensemble (rerank)", mom="Simple momentum", rnd="Random-3",
               ew="Equal-weight", nq="Nasdaq",
               ylab="Fold cumulative return %", xlab="",
               title="Fig. Simple momentum vs. ensemble by regime — momentum collapse in 2025 H1 and ensemble robustness",
               note="2025 H1: momentum collapses ({mom:+.0f}%) while ensemble stays robust ({ens:+.0f}%)"),
}[LANG]

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "ensemble_wf_predictions_seed42.csv"   # 표준 실행 (seed 42 + 결정론)
DB   = ROOT / "db" / "adv_ai_nasdaq.db"
OUT  = ROOT / ("backtest_wf_bench_en" if LANG == "en" else "backtest_wf_bench")
GBM_MIN, N_CAND, TOPN, HOLD, COST = 0.50, 8, 3, 3, 0.001
N_RANDOM = 20


def load():
    sig = pd.read_csv(PRED, parse_dates=["date"])
    con = sqlite3.connect(DB)
    px = pd.read_sql("SELECT ticker,date,open,adj_close,close,nasdaq_close FROM stock_prices", con)
    con.close()
    px["date"] = pd.to_datetime(px["date"]); px = px.sort_values(["ticker", "date"])
    px["adj_open"] = px["open"] * px["adj_close"] / px["close"]      # 배당·분할 조정 시가
    px["ret_t3"] = (px.groupby("ticker")["adj_open"].shift(-(HOLD+1))
                    / px.groupby("ticker")["adj_open"].shift(-1) - 1)  # 다음날 시가→+3일 시가
    px["mom20"] = (px.groupby("ticker")["adj_close"].transform(lambda s: s.pct_change(20))
                   .groupby(px["ticker"]).shift(1))            # 인과적 과거 모멘텀
    sig = sig.merge(px[["ticker", "date", "ret_t3", "mom20"]], on=["ticker", "date"], how="left")
    return sig, px


def curve_total(daily_ret, dates):
    """3코호트 균등 복리 → 최종 누적수익."""
    grid = pd.DatetimeIndex(dates); cs = []
    for off in range(HOLD):
        cur, eq = 1.0, {}
        for d in dates[off::HOLD]:
            cur *= (1 + daily_ret.get(d, 0.0)); eq[d] = cur
        cs.append(pd.Series(eq).reindex(grid).ffill().fillna(1.0))
    return pd.concat(cs, axis=1).mean(axis=1).iloc[-1] - 1


def strat_returns(sub, kind, rng=None):
    """그날 선택 3종목의 균등 평균수익(비용 차감) 딕셔너리."""
    out = {}
    for d, g in sub.groupby("date"):
        if kind == "ens":
            pool = g[g["prob_lgb"] >= GBM_MIN]
            cand = pool.sort_values("prob_lgb", ascending=False).head(N_CAND)
            r = cand.sort_values("prob_lstm", ascending=False).head(TOPN)["ret_t3"].dropna()
        elif kind == "mom":
            r = g.dropna(subset=["mom20"]).sort_values("mom20", ascending=False).head(TOPN)["ret_t3"].dropna()
        elif kind == "ew":
            r = g["ret_t3"].dropna()
        elif kind == "rnd":
            rv = g["ret_t3"].dropna().values
            r = pd.Series(rng.choice(rv, min(TOPN, len(rv)), replace=False)) if len(rv) else pd.Series([], dtype=float)
        out[d] = (r.mean() - COST) if len(r) else 0.0
    return out


def total_for(sub, kind):
    dates = np.sort(sub["date"].unique())
    if kind == "rnd":
        rng = np.random.default_rng(0)
        return np.mean([curve_total(strat_returns(sub, "rnd", rng), dates) for _ in range(N_RANDOM)])
    if kind == "nq":
        con = sqlite3.connect(DB); px = pd.read_sql("SELECT date,nasdaq_close FROM stock_prices", con); con.close()
        px["date"] = pd.to_datetime(px["date"])
        nq = px.drop_duplicates("date").sort_values("date")
        nq = nq[(nq["date"] >= sub["date"].min()) & (nq["date"] <= sub["date"].max())]
        return nq["nasdaq_close"].iloc[-1] / nq["nasdaq_close"].iloc[0] - 1
    return curve_total(strat_returns(sub, kind), dates)


def main():
    sig, px = load()
    folds = list(dict.fromkeys(sig["fold"]))
    fold_lbl = [f.split("~")[0][:7] for f in folds]
    kinds = ["ens", "mom", "rnd", "ew", "nq"]
    names = {"ens": T["ens"], "mom": T["mom"], "rnd": T["rnd"], "ew": T["ew"], "nq": T["nq"]}

    # 폴드별 + 전체
    res = {k: [] for k in kinds}
    for f in folds:
        sub = sig[sig["fold"] == f]
        for k in kinds:
            res[k].append(total_for(sub, k))
    overall = {k: total_for(sig, k) for k in kinds}

    # 콘솔 표
    print("=" * 78)
    print("  벤치마크 비교 (동일 유니버스, 3코호트 균등, 비용 0.1%, 표준 seed 42)")
    print("=" * 78)
    hdr = f"  {'구간':<9}" + "".join(f"{names[k]:>13}" for k in kinds)
    print(hdr); print("-" * 78)
    for i, f in enumerate(folds):
        print(f"  {fold_lbl[i]:<9}" + "".join(f"{res[k][i]*100:>+12.0f}%" for k in kinds))
    print("-" * 78)
    print(f"  {'전체':<9}" + "".join(f"{overall[k]*100:>+12.0f}%" for k in kinds))
    print("=" * 78)
    # 선정 알파 & 모멘텀 붕괴 지표
    print("  [선정 알파] 앙상블 − 무작위 (폴드별):",
          " / ".join(f"{(res['ens'][i]-res['rnd'][i])*100:+.0f}%p" for i in range(len(folds))))
    print("  [모멘텀 국면] 단순모멘텀 폴드별:",
          " / ".join(f"{res['mom'][i]*100:+.0f}%" for i in range(len(folds))),
          "→ 2025 H1 붕괴")

    # ── 그림: 폴드별 그룹 막대 (앙상블/모멘텀/무작위) ──
    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.patch.set_facecolor("white"); ax.set_facecolor("#fbfcfd")
    x = np.arange(len(folds)); w = 0.26
    bars = [("ens", "#e8443b", -w), ("mom", "#e0913b", 0.0), ("rnd", "#8a94a0", w)]
    for k, c, off in bars:
        vals = [res[k][i]*100 for i in range(len(folds))]
        b = ax.bar(x + off, vals, w, color=c, label=names[k], zorder=3,
                   edgecolor="white", linewidth=0.6)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 4, f"{v:+.0f}", ha="center", va="bottom",
                    fontsize=8, color=c, fontweight="bold")
    # 2025 H1 강조
    if "2025-01" in fold_lbl:
        j = fold_lbl.index("2025-01")
        ax.axvspan(j-0.45, j+0.45, color="#fff2cc", alpha=0.6, zorder=0)
        ax.annotate(T["note"].format(mom=res["mom"][j]*100, ens=res["ens"][j]*100),
                    xy=(j, res["mom"][j]*100),
                    xytext=(j, max(res['ens'])*100*0.72), ha="center",
                    fontsize=9, color="#b5651d",
                    arrowprops=dict(arrowstyle="->", color="#b5651d", lw=1.2))
    ax.set_xticks(x); ax.set_xticklabels(["2024 H2", "2025 H1", "2025 H2", "2026 H1"], fontsize=10)
    ax.set_ylabel(T["ylab"], fontsize=10.5)
    ax.axhline(0, color="#888", lw=0.8)
    ax.legend(fontsize=10, frameon=True, framealpha=0.95, edgecolor="#e0e0e0", loc="upper right")
    ax.grid(True, axis="y", color="#d9dde2", lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title(T["title"], fontsize=12, fontweight="normal", pad=12, loc="center")
    fig.savefig(str(OUT)+".png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(str(OUT)+".pdf", bbox_inches="tight", facecolor="white")
    print(f"\n저장: {OUT}.png/.pdf")


if __name__ == "__main__":
    main()
