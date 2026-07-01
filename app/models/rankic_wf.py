"""
RankIC 계산 (관련논문 지표로 정직 비교용)
LSTM 예측(prob_lstm)을 실제 5일 후 수익률과 날짜별 순위상관(Spearman)으로 측정.
RankIC = mean_t( corr_rank(pred_t, ret_t) ),  RankICIR = mean/std
실행: python -m app.models.rankic_wf
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from app.database.sqlite_db import get_connection
from app.models.ensemble_wf import lstm_fold, FOLDS  # 기존 LSTM 예측 재사용

# 1) 실제 5일 후 수익률 (나스닥 DB)
os.environ.setdefault('DB_FILE', 'db/adv_ai_nasdaq.db')
conn = get_connection()
px = pd.read_sql("SELECT date, ticker, adj_close FROM stock_prices", conn)
conn.close()
px['date'] = pd.to_datetime(px['date'])
px = px.sort_values(['ticker', 'date'])
px['fwd5'] = px.groupby('ticker')['adj_close'].shift(-5) / px['adj_close'] - 1
ret_map = px[['date', 'ticker', 'fwd5']]

# 2) 각 폴드에서 LSTM 예측 모으기
preds = []
for tr_end, te_s, te_e in FOLDS:
    out = lstm_fold(tr_end, te_e)
    if out is None:
        continue
    l = out.copy()
    l['date'] = pd.to_datetime(l['date'])
    # 테스트 구간만
    l = l[(l['date'] >= pd.Timestamp(te_s)) & (l['date'] < pd.Timestamp(te_e))]
    preds.append(l[['date', 'ticker', 'prob_lstm']])
pred = pd.concat(preds, ignore_index=True)

# 3) 예측 + 실제수익률 병합
m = pred.merge(ret_map, on=['date', 'ticker'], how='left').dropna(subset=['fwd5'])

# 4) 날짜별 RankIC (Spearman)
rics = []
for d, g in m.groupby('date'):
    if len(g) < 5:
        continue
    ic, _ = spearmanr(g['prob_lstm'], g['fwd5'])
    if not np.isnan(ic):
        rics.append(ic)
rics = np.array(rics)

print("===== LSTM RankIC (실제 5일 수익률 기준) =====")
print(f"평가 일수: {len(rics)}")
print(f"RankIC   : {rics.mean():.4f}")
print(f"RankICIR : {rics.mean()/rics.std():.4f}  (평균/표준편차)")
print(f"양(+) 비율: {(rics>0).mean():.1%}")
print("\n참고: 관련논문(S&P500 등) RankIC 0.01~0.14 수준")
