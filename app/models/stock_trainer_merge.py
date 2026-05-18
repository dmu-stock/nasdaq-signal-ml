import pandas as pd

# 두 모델의 예측 결과 로드 (date, ticker 기준)
lgb_res = pd.read_csv("prediction_result.csv")  # lgb 결과 (pred_prob 포함 필수)
lstm_res = pd.read_csv("lstm_prediction_result.csv")  # lstm 결과 (pred_prob 포함 필수)

# 필요한 컬럼만 정리해서 병합 (suffix로 구분)
m1 = lgb_res[['date', 'ticker', 'pred_prob', 'actual']].rename(columns={'pred_prob': 'prob_lgb'})
m2 = lstm_res[['date', 'ticker', 'pred_prob']].rename(columns={'pred_prob': 'prob_lstm'})

ensemble_df = pd.merge(m1, m2, on=['date', 'ticker'], how='inner')

# 3. 두 모델의 확률 합치기 
ensemble_df['final_prob'] = (ensemble_df['prob_lgb'] * 0.6) + (ensemble_df['prob_lstm'] * 0.4)

# 합쳐진 점수를 기준으로 다시 Top 50 뽑아서 성능 확인해보기
top_50_ensemble = ensemble_df.sort_values(by='final_prob', ascending=False).drop_duplicates(subset=['ticker']).head(50)

print("===== 앙상블 모델 Top 50 상승 비율 =====")
print(top_50_ensemble['actual'].mean())