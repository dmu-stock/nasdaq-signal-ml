from xgboost import XGBClassifier
import pandas as pd

from sklearn.metrics import (
classification_report,
accuracy_score,
roc_auc_score,
)

# -----------------------------

# 데이터 로드

# -----------------------------

df_tech = pd.read_csv("feature__indicator20260518.csv")

df_tech['date'] = pd.to_datetime(df_tech['date'])

# -----------------------------

# Feature 선택

# -----------------------------

feature_cols = [

'return_5',

'alpha',
'alpha_5',
'alpha_20',
'alpha_divergence',

'ma_ratio',

'rsi',
'volatility_5',

'volume_ratio',

'nasdaq_change_rate',

'macd_hist',

'drawdown_20',

'bb_percent',

'tr_5'

]

target_col = 'label'

# -----------------------------

# Train / Test Split

# -----------------------------

train = df_tech[df_tech['date'] < '2025-07-01']
test = df_tech[df_tech['date'] >= '2025-07-01']

X_train = train[feature_cols]
y_train = train[target_col]

X_test = test[feature_cols]
y_test = test[target_col]

print(f"Train Size: {len(train)}")
print(f"Test Size : {len(test)}")

# -----------------------------

# XGBoost 모델

# -----------------------------

model = XGBClassifier(
    objective='binary:logistic',

    n_estimators=300,
    learning_rate=0.03,

    max_depth=5,

    subsample=0.8,
    colsample_bytree=0.8,

    random_state=42,

    eval_metric='logloss',

    scale_pos_weight=(
        (y_train == 0).sum() / (y_train == 1).sum()
    )
)

# -----------------------------

# 학습

# -----------------------------

model.fit(X_train, y_train)

# -----------------------------

# 확률 예측

# -----------------------------

pred_prob = model.predict_proba(X_test)[:, 1]

# threshold tuning

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

result_df = result_df.sort_values(
by='pred_prob',
ascending=False
)

result_df.to_csv("xgboost_prediction_result.csv", index=False)

print("\nxgboost_prediction_result.csv 저장 완료")
