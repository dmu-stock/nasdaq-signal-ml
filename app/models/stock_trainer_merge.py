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

print("\n==============================================")
print("     🔥 앙상블 모델 Top 50 최종 성능 리포트 🔥     ")
print("==============================================")

# 2. 찐 타율 계산 및 출력
final_hit_rate = top_50_ensemble['actual'].mean()
print(f"▶ Top 50 찐 타율 (상승 비율) : {final_hit_rate:.4f} ({final_hit_rate*100:.2f}%)")
print("----------------------------------------------")

# 3. 실전 매매용 최상위 TOP 5 종목 화면에 뿌려주기
print("주먹 봇 추천 내일 장 살만한 TOP 5 종목:")
print(top_50_ensemble[['ticker', 'date', 'actual', 'final_prob']].head(5))
print("==============================================")

# 4. CSV 파일로 깔끔하게 저장 완료하기
output_filename = "ensemble_top50.csv"
top_50_ensemble.to_csv(output_filename, index=False)
print(f"[성공] 앙상블 리스트가 '{output_filename}'로 저장되었습니다.")
