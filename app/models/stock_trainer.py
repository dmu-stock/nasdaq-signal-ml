import pandas as pd
import os
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score
)

# -----------------------------
# 데이터 로드
# -----------------------------
df_tech = pd.read_csv("feature__indicator20260513.csv")
df_news = pd.read_csv("news_sentiment_20260513.csv")

final_df = pd.merge(df_tech, df_news[['ticker', 'date', 'sentiment_score']], 
                    on=['ticker', 'date'], how='left')

final_df['sentiment_score'] = final_df['sentiment_score'].fillna(0)

# 날짜 타입 변환
final_df['date'] = pd.to_datetime(final_df['date'])
final_df = final_df[abs(final_df['sentiment_score'] - 0.15) > 0.05]

# -----------------------------
# Feature 선택
# -----------------------------
feature_cols = [
    'change_rate',
    'return_5',

    'alpha',
    'alpha_5',
    'alpha_20',

    'ma_ratio',

    'rsi',
    'volatility_5',

    'volume_ratio',

    'nasdaq_change_rate',

    'sentiment_score'
]

target_col = 'label'

# -----------------------------
# Train / Test Split
# 시계열이라 날짜 기준 분리
# -----------------------------
train = final_df[final_df['date'] < '2025-07-01']
test = final_df[final_df['date'] >= '2025-07-01']

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
    objective='binary',
    boosting_type='gbdt',

    n_estimators=300,
    learning_rate=0.03,

    max_depth=5,
    num_leaves=31,

    subsample=0.8,
    colsample_bytree=0.8,

    random_state=42,
    class_weight='balanced'
)

# -----------------------------
# 학습
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# 예측
# -----------------------------
pred = model.predict(X_test)

# 상승 확률
pred_prob = model.predict_proba(X_test)[:, 1]

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

result_df.to_csv("prediction_result.csv", index=False)

print("\nprediction_result.csv 저장 완료")