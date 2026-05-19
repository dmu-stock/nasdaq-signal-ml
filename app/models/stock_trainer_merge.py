import pandas as pd
import numpy as np

# 1. 두 모델의 예측 결과 로드
lgb_res = pd.read_csv("prediction_result.csv")
lstm_res = pd.read_csv("lstm_prediction_result.csv")

# 2. 필요한 컬럼만 정리해서 병합
m1 = lgb_res[['date', 'ticker', 'pred_prob', 'actual']].rename(columns={'pred_prob': 'prob_lgb'})
m2 = lstm_res[['date', 'ticker', 'pred_prob']].rename(columns={'pred_prob': 'prob_lstm'})

ensemble_df = pd.merge(m1, m2, on=['date', 'ticker'], how='inner')

ensemble_df['final_prob'] = (ensemble_df['prob_lgb'] * 0.7) + (ensemble_df['prob_lstm'] * 0.3)

print(f"앙상블 완료된 데이터 수: {len(ensemble_df)}개")

# ---------------------------------------------------
# 3. 앙상블 모델 Daily Top-K 평가 (0.65 가드레일)
# ---------------------------------------------------
print("\n===== Ensemble Daily Top-K Performance =====")

ensemble_top3_actuals = []
CONFIDENCE_THRESHOLD = 0.71  # 앙상블 결합 확률이 0.65 이상일 때만 베팅

# 날짜순으로 정렬 후 그룹화
ensemble_df = ensemble_df.sort_values(['date', 'final_prob'], ascending=[True, False])

for date, group in ensemble_df.groupby('date'):
    
    # 두 모델의 시너지가 반영된 final_prob 기준으로 당일 탑 3 추출
    group_sorted = group.sort_values('final_prob', ascending=False)
    top3 = group_sorted.head(3)
    
    # 둘 다 동의해서 확률이 0.65를 뚫어낸 알짜배기 종목만 필터링
    valid_picks = top3[top3['final_prob'] >= CONFIDENCE_THRESHOLD]
    
    if not valid_picks.empty:
        ensemble_top3_actuals.extend(valid_picks['actual'].tolist())

# 최종 앙상블 타율 계산
ensemble_accuracy = np.mean(ensemble_top3_actuals) if ensemble_top3_actuals else 0

print(f"가드레일 기준: 앙상블 결합 확률 {CONFIDENCE_THRESHOLD} 이상")
print(f"매일 랭킹 Top 3 매수 시 진짜 타율 : {ensemble_accuracy:.4f}")
print(f"총 매수 진입 횟수 (종목 수)       : {len(ensemble_top3_actuals)}개")
print(f"시장 평균 상승 비율 (Baseline)     : {ensemble_df['actual'].mean():.4f}")

# ---------------------------------------------------
# 앙상블 최종 결과 CSV 저장 로직 추가
# ---------------------------------------------------

ensemble_final_df = ensemble_df[[
    'date', 'ticker', 'prob_lgb', 'prob_lstm', 'final_prob', 'actual'
]]

# 날짜는 오름차순, 그날의 앙상블 확률은 내림차순으로 칼정렬해서 저장
ensemble_final_df = ensemble_final_df.sort_values(
    ['date', 'final_prob'], 
    ascending=[True, False]
)

ensemble_final_df.to_csv(
    "ensemble_prediction_result.csv", 
    index=False
)

print("\n[안내] ensemble_prediction_result.csv 저장 완료!")
print(f"최종 저장된 행(Row) 수: {len(ensemble_final_df)}개")