import pandas as pd
import os
import numpy as np
import joblib
from app.config.config import GBM_FEATURE_COLS
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score
)

# -----------------------------
# 데이터 로드
# -----------------------------
df_tech = pd.read_csv("feature__indicator_20260522.csv")
# df_news = pd.read_csv("news_sentiment_20260513.csv")

# final_df = pd.merge(df_tech,[['ticker', 'date', 'sentiment_score']],on=['ticker', 'date'], how='left')

# final_df['sentiment_score'] = final_df['sentiment_score'].fillna(0)

# 날짜 타입 변환
df_tech['date'] = pd.to_datetime(df_tech['date'])
# final_df = final_df[abs(final_df['sentiment_score'] - 0.15) > 0.05]

# -----------------------------
# Feature 선택
# -----------------------------
feature_cols = GBM_FEATURE_COLS

target_col = 'label'

# -----------------------------
# Train / Test Split
# 시계열이라 날짜 기준 분리
# -----------------------------
train = df_tech[df_tech['date'] < '2025-07-01']
test = df_tech[df_tech['date'] >= '2025-07-01']

X_train = train[feature_cols]
y_train = train[target_col]

X_test = test[feature_cols]
y_test = test[target_col]

print(f"Train Size: {len(train)}")
print(f"Test Size: {len(test)}")

# -----------------------------
# LightGBM 모델 생성
# -----------------------------
model = LGBMClassifier(
    n_estimators=1000,
    objective='binary',
    boosting_type='gbdt',

    learning_rate=0.01,

    max_depth=5,
    num_leaves=31,
    min_data_in_leaf= 50,
    feature_fraction= 0.8,
    force_col_wise= True,
    subsample=0.8,
    colsample_bytree=0.7,

    random_state=42,
    class_weight='balanced',
    verbose= -1,
)

# -----------------------------
# 학습
# -----------------------------
model.fit(X_train, y_train)
joblib.dump(model, 'best_lgbm_model.pkl')

# -----------------------------
# 예측
# -----------------------------
pred = model.predict(X_test)

# 상승 확률
pred_prob = model.predict_proba(X_test)[:, 1]
threshold = 0.55
pred = (pred_prob >= threshold).astype(int)

# -----------------------------
# 평가
# -----------------------------
print("\n===== Classification Report =====")
print(classification_report(y_test, pred))

print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, pred_prob):.4f}")

# -----------------------------
# Feature Importance
# -----------------------------
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\n===== Feature Importance =====")
print(importance_df)

# -----------------------------
# 예측 결과 저장
# -----------------------------
result_df = test[['ticker', 'date']].copy()

result_df['actual'] = y_test.values
result_df['pred'] = pred
result_df['pred_prob'] = pred_prob

# -----------------------------
# 날짜별 상대 점수(Z-score)
# -----------------------------
result_df['rank_score'] = (
    result_df.groupby('date')['pred_prob']
    .transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )
)
# -----------------------------
# 날짜별 Top50 성능
# -----------------------------
daily_scores = []

for date, group in result_df.groupby("date"):

    top50 = group.sort_values(
        "rank_score",
        ascending=False
    ).head(50)

    hit_rate = top50["actual"].mean()

    daily_scores.append(hit_rate)

print("\n===== Daily Top50 Mean =====")
print(np.mean(daily_scores))

# -----------------------------
# 전체 기준 Top-K
# -----------------------------
print("\n===== Top-K Performance =====")
for k in [3, 5, 10]:
    daily_actuals = []
    
    for date, group in result_df.groupby('date'):
        top_k = group.sort_values('rank_score', ascending=False).head(k)
        daily_actuals.extend(top_k['actual'].tolist())
    
    hit_rate = np.mean(daily_actuals)
    print(f"날짜별 Top{k} 평균 타율: {hit_rate:.4f}  "
          f"(베이스라인 대비 {hit_rate/result_df['actual'].mean():.2f}배)")

print(f"베이스라인: {result_df['actual'].mean():.4f}")



result_df.to_csv("prediction_result.csv", index=False)

print("\nprediction_result.csv 저장 완료")