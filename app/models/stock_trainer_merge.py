import pandas as pd
import numpy as np

# ---------------------------------------------------
# 두 모델 예측 결과 로드 & 병합
# ---------------------------------------------------
lgb_res  = pd.read_csv("prediction_result.csv")
lstm_res = pd.read_csv("lstm_prediction_result.csv")

m1 = lgb_res[['date', 'ticker', 'pred_prob', 'actual']].rename(columns={'pred_prob': 'prob_lgb'})
m2 = lstm_res[['date', 'ticker', 'pred_prob']].rename(columns={'pred_prob': 'prob_lstm'})

ensemble_df = pd.merge(m1, m2, on=['date', 'ticker'], how='inner')

# 조화평균 앙상블 점수 (둘 다 높아야 높은 점수)
ensemble_df['final_prob'] = (
    2 * (ensemble_df['prob_lgb'] * ensemble_df['prob_lstm'])
    / (ensemble_df['prob_lgb'] + ensemble_df['prob_lstm'] + 1e-9)
)

# ---------------------------------------------------
# 절대 확률 분포 확인 (임계치 튜닝 참고용)
# ---------------------------------------------------
print("===== Probability Distribution =====")
print("  [LGBM]")
for t in [0.40, 0.43, 0.45, 0.48, 0.50, 0.52]:
    n = (ensemble_df['prob_lgb'] >= t).sum()
    print(f"    >= {t:.2f}: {n:>5}개  ({n/len(ensemble_df)*100:.1f}%)")
print("  [LSTM]")
for t in [0.45, 0.50, 0.55, 0.58, 0.60, 0.65]:
    n = (ensemble_df['prob_lstm'] >= t).sum()
    print(f"    >= {t:.2f}: {n:>5}개  ({n/len(ensemble_df)*100:.1f}%)")

# ---------------------------------------------------
# AND 게이트 앙상블
# ---------------------------------------------------
LGBM_THRESHOLD = 0.55
LSTM_THRESHOLD = 0.60

baseline = ensemble_df['actual'].mean()

prob_top3_actuals = []
prob_top5_actuals = []

for date, group in ensemble_df.groupby('date'):
    filtered = group[
        (group['prob_lgb']  >= LGBM_THRESHOLD) &
        (group['prob_lstm'] >= LSTM_THRESHOLD)
    ].sort_values('final_prob', ascending=False)

    top3 = filtered.head(3)
    top5 = filtered.head(5)
    if not top3.empty:
        prob_top3_actuals.extend(top3['actual'].tolist())
    if not top5.empty:
        prob_top5_actuals.extend(top5['actual'].tolist())

p_top3 = np.mean(prob_top3_actuals) if prob_top3_actuals else 0.0
p_top5 = np.mean(prob_top5_actuals) if prob_top5_actuals else 0.0

print(f"\n===== AND Gate Ensemble =====")
print(f"가드레일: LGBM >= {LGBM_THRESHOLD}, LSTM >= {LSTM_THRESHOLD}")
print(f"Top3 타율: {p_top3:.4f}  ({len(prob_top3_actuals)}회)  베이스라인 대비 {p_top3/baseline:.2f}배")
print(f"Top5 타율: {p_top5:.4f}  ({len(prob_top5_actuals)}회)  베이스라인 대비 {p_top5/baseline:.2f}배")
print(f"\n베이스라인: {baseline:.4f}")

# ---------------------------------------------------
# 결과 저장
# ---------------------------------------------------
out = ensemble_df[['date', 'ticker', 'prob_lgb', 'prob_lstm', 'final_prob', 'actual']]
out.to_csv("ensemble_prediction_result.csv", index=False)
print(f"\nensemble_prediction_result.csv 저장 완료  ({len(out)}행)")
