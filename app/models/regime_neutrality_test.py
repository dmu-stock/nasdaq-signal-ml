"""
레짐 중립성 검증 실험
가설: 강도 피처(변동성·수급)는 추세장/반전장 두 국면에서 부호(방향)가 유지되고,
      방향 피처(모멘텀)는 국면에 따라 부호가 뒤집힌다.

방법:
1) mom_factor(모멘텀 전략 당일 손익)로 각 날짜를 추세장(>0)/반전장(<0)으로 라벨
2) 각 국면에서 [모멘텀] vs [강도 피처들]의 5일 후 수익률 순위상관(Spearman) 비교
3) 모멘텀은 부호 반전, 강도는 부호 유지 → '레짐 저민감' 실증

실행: python -m app.models.regime_neutrality_test
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

os.environ.setdefault('DB_FILE', 'db/adv_ai_nasdaq.db')
from app.database.sqlite_db import get_connection

# ---- 1. 가격 → 모멘텀·미래수익률 ----
conn = get_connection()
px = pd.read_sql("SELECT date, ticker, adj_close FROM stock_prices", conn); conn.close()
px['date'] = pd.to_datetime(px['date'])
px = px.sort_values(['ticker', 'date'])
px['ret1'] = px.groupby('ticker')['adj_close'].pct_change()
px['mom20'] = px.groupby('ticker')['adj_close'].pct_change(20)          # 과거 20일 모멘텀
px['mom20_lag'] = px.groupby('ticker')['mom20'].shift(1)                # 인과적 (어제까지)
px['fwd5'] = px.groupby('ticker')['adj_close'].shift(-5) / px['adj_close'] - 1  # 5일 후 수익률

# ---- 2. mom_factor: 그날 모멘텀 상위30% 수익률 - 하위30% (추세장/반전장 지표) ----
def day_factor(g):
    v = g['mom20_lag'].notna()
    if v.sum() < 5:
        return np.nan
    hi = g['mom20_lag'].quantile(0.7); lo = g['mom20_lag'].quantile(0.3)
    hi_r = g.loc[g['mom20_lag'] >= hi, 'ret1'].mean()
    lo_r = g.loc[g['mom20_lag'] <= lo, 'ret1'].mean()
    return hi_r - lo_r
fac = px.groupby('date').apply(day_factor, include_groups=False)
fac20 = fac.rolling(20).sum()   # 20일 누적 → 국면 판단 (make_fig와 동일)
regime = pd.DataFrame({'date': fac20.index, 'regime_val': fac20.values})
regime['regime'] = np.where(regime['regime_val'] > 0, '추세장', '반전장')

# ---- 3. 강도 피처(LSTM CSV) 병합 ----
feat = pd.read_csv("feature__indicator_lstm20260623.csv")
feat['date'] = pd.to_datetime(feat['date'])
STRENGTH = ['volatility_regime_20', 'volume_zscore_20', 'candle_body', 'high_low_spread']

d = px[['date', 'ticker', 'mom20_lag', 'fwd5']].merge(
        feat[['date', 'ticker'] + STRENGTH], on=['date', 'ticker'], how='inner')
d = d.merge(regime[['date', 'regime']], on='date', how='left')
d = d.dropna(subset=['fwd5', 'regime'])

# ---- 4. 국면별 피처-미래수익률 순위상관 ----
def corr_by_regime(col):
    out = {}
    for rg in ['추세장', '반전장']:
        g = d[d['regime'] == rg].dropna(subset=[col])
        if len(g) < 30:
            out[rg] = np.nan; continue
        out[rg] = spearmanr(g[col], g['fwd5'])[0]
    return out

print("===== 레짐 중립성 검증 (피처 vs 5일 후 수익률 순위상관) =====")
print(f"추세장 일수: {(d.groupby('date')['regime'].first()=='추세장').sum()}  "
      f"반전장 일수: {(d.groupby('date')['regime'].first()=='반전장').sum()}\n")
print(f"{'피처':<22}{'추세장':>10}{'반전장':>10}{'부호 유지?':>12}")
print("-" * 54)

# 방향 피처 (모멘텀)
mc = corr_by_regime('mom20_lag')
same = "반전(X)" if (mc['추세장'] * mc['반전장'] < 0) else "유지(O)"
print(f"{'[방향] mom20':<22}{mc['추세장']:>10.4f}{mc['반전장']:>10.4f}{same:>12}")
print("-" * 54)
# 강도 피처들
for c in STRENGTH:
    sc = corr_by_regime(c)
    same = "반전(X)" if (sc['추세장'] * sc['반전장'] < 0) else "유지(O)"
    print(f"{'[강도] '+c:<22}{sc['추세장']:>10.4f}{sc['반전장']:>10.4f}{same:>12}")

print("\n해석: 모멘텀은 두 국면서 부호 반전(레짐 의존), "
      "강도 피처는 부호 유지면 '레짐 저민감' 실증.")
