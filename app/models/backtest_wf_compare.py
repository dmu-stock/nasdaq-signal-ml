"""자본배분 A vs B 비교 — 종목 부족일 처리 방식 결정용."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from app.models.backtest_wf import load, run_strategy, benchmark
import pandas as pd


def pf(x):
    return f"{x*100:+.1f}%" if pd.notna(x) else "n/a"


def pair(a, b, key, f=pf):
    if key == "sharpe":
        return f"{a[key]:.2f} / {b[key]:.2f}"
    return f"{f(a[key])} / {f(b[key])}"


def main():
    sig, px = load()
    folds = list(dict.fromkeys(sig["fold"]))
    print("=" * 82)
    print("  자본배분  A(있는만큼 균등·공격적)  vs  B(슬롯고정 1/3·보수적)   비용 0.1%")
    print("=" * 82)
    print(f"  {'구간':<10}{'누적수익 A/B':<24}{'CAGR A/B':<22}{'Sharpe A/B':<16}{'MDD A/B'}")
    print("-" * 82)
    for f in folds:
        sub = sig[sig["fold"] == f]
        a = run_strategy(sub, 0.001, "A")
        b = run_strategy(sub, 0.001, "B")
        yr = f.split("~")[0][:7]
        print(f"  {yr:<10}{pair(a,b,'total_return'):<24}{pair(a,b,'cagr'):<22}"
              f"{pair(a,b,'sharpe'):<16}{pair(a,b,'mdd')}")
    print("-" * 82)
    a = run_strategy(sig, 0.001, "A")
    b = run_strategy(sig, 0.001, "B")
    print(f"  {'★ 전체':<9}{pair(a,b,'total_return'):<24}{pair(a,b,'cagr'):<22}"
          f"{pair(a,b,'sharpe'):<16}{pair(a,b,'mdd')}")
    print("-" * 82)
    bench = benchmark(px, sig["date"].min(), sig["date"].max())
    print(f"  나스닥 B&H: 누적 {pf(bench['total_return'])}, CAGR {pf(bench['cagr'])}, "
          f"Sharpe {bench['sharpe']:.2f}, MDD {pf(bench['mdd'])}")
    print("=" * 82)
    print(f"  참고 — 하루 평균 매수 종목수 2.09개 / 0종목(현금)일 19.8% / 1종목 몰빵일 11.3%")


if __name__ == "__main__":
    main()
