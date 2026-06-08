import pandas as pd
import os
import numpy as np
import shap
import joblib
from app.config.config import GBM_FEATURE_COLS
import lightgbm as lgb
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV


def walk_forward_eval(df: pd.DataFrame, feature_cols: list, target_col: str) -> pd.DataFrame:
    """
    확장 윈도우 워크-포워드 검증.
    각 폴드마다 독립적으로 모델을 학습하여 AUC를 측정한다.
    단순 날짜 분할 결과가 특정 시장 구간에 운 좋게 맞춰진 것인지 확인한다.
    """
    folds = [
        ('2024-01-01', '2024-07-01'),
        ('2024-04-01', '2024-10-01'),
        ('2024-07-01', '2025-01-01'),
        ('2024-10-01', '2025-04-01'),
    ]

    params = dict(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=6,
        num_leaves=20,
        min_data_in_leaf=80,
        feature_fraction=0.7,
        subsample=0.7,
        subsample_freq=1,
        lambda_l1=0.1,
        lambda_l2=0.1,
        objective='binary',
        boosting_type='gbdt',
        force_col_wise=True,
        random_state=42,
        class_weight='balanced',
        verbose=-1,
    )

    records = []
    for test_start, test_end in folds:
        fold_train = df[df['date'] < test_start]
        fold_test  = df[(df['date'] >= test_start) & (df['date'] < test_end)]

        if len(fold_train) < 100 or len(fold_test) < 10:
            continue
        if fold_test[target_col].nunique() < 2:
            continue

        m = LGBMClassifier(**params)
        m.fit(fold_train[feature_cols], fold_train[target_col])
        prob = m.predict_proba(fold_test[feature_cols])[:, 1]

        records.append({
            'test_period':  f'{test_start} ~ {test_end}',
            'train_rows':   len(fold_train),
            'test_rows':    len(fold_test),
            'pos_rate':     round(fold_test[target_col].mean(), 4),
            'auc':          round(roc_auc_score(fold_test[target_col], prob), 4),
        })

    return pd.DataFrame(records)

# -----------------------------
# 데이터 로드
# -----------------------------
df_tech = pd.read_csv("feature__indicator_20260608.csv")

# 날짜 타입 변환
df_tech['date'] = pd.to_datetime(df_tech['date'])

# -----------------------------
# Feature 선택
# -----------------------------
feature_cols = GBM_FEATURE_COLS

target_col = 'label'

# -----------------------------
# Train / Val / Test Split
# -----------------------------
train = df_tech[df_tech['date'] < '2024-07-01']
val   = df_tech[(df_tech['date'] >= '2024-07-01') & 
                (df_tech['date'] < '2025-07-01')]
test  = df_tech[df_tech['date'] >= '2025-07-01']

X_train, y_train = train[feature_cols], train[target_col]
X_val,   y_val   = val[feature_cols],   val[target_col]
X_test,  y_test  = test[feature_cols],  test[target_col]

print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
print(f"Train 양성비율: {y_train.mean():.4f}")
print(f"Val   양성비율: {y_val.mean():.4f}")
print(f"Test  양성비율: {y_test.mean():.4f}")

# -----------------------------
# LightGBM 모델 생성
# -----------------------------
model = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.005,
    max_depth=6,
    num_leaves=31,          # 20 → 31 (LightGBM 기본값, 표현력 증가)
    min_data_in_leaf=50,    # 80 → 50 (너무 보수적)
    feature_fraction=0.8,   # 0.7 → 0.8 (피처 더 활용)
    subsample=0.8,          # 0.7 → 0.8
    subsample_freq=1,
    colsample_bytree=0.8,   # 0.7 → 0.8
    lambda_l1=0.05,         # 0.1 → 0.05 (정규화 완화)
    lambda_l2=0.05,         # 0.1 → 0.05
    objective='binary',
    boosting_type='gbdt',
    force_col_wise=True,
    random_state=42,
    class_weight=None,
    scale_pos_weight=2.0,
    verbose=-1,
)

# -----------------------------
# 학습
# -----------------------------
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(100)
    ]
)
print(f"\n최적 트리 수: {model.best_iteration_}")
joblib.dump(model, 'best_lgbm_model.pkl')


# -----------------------------
# 보정
# -----------------------------
calibrated = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
calibrated.fit(X_val, y_val)
joblib.dump(calibrated, 'best_lgbm_model.pkl')  # 같은 파일명으로 덮어쓰기

# 보정 후 분포 확인
cal_prob = calibrated.predict_proba(X_test)[:, 1]
print("\n===== Calibrated Probability Distribution =====")
for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
    n = (cal_prob >= t).sum()
    print(f"  >= {t:.2f}: {n:>5}개  ({n/len(cal_prob)*100:.1f}%)")

# -----------------------------
# 예측
# -----------------------------
pred = model.predict(X_test)


# 상승 확률
# pred_prob = model.predict_proba(X_test)[:, 1]
pred_prob = calibrated.predict_proba(X_test)[:, 1]

# 확률 분포 확인 (threshold 튜닝용)
print("\n===== Probability Distribution =====")
for t in [0.40, 0.43, 0.45, 0.48, 0.50, 0.53, 0.55]:
    n = (pred_prob >= t).sum()
    print(f"  >= {t:.2f}: {n:>5}개  ({n/len(pred_prob)*100:.1f}%)")

threshold = 0.55
pred = (pred_prob >= threshold).astype(int)
print(f"\n[사용 threshold: {threshold}]")

# -----------------------------
# 평가
# -----------------------------
print("\n===== Classification Report =====")
print(classification_report(y_test, pred, zero_division=0))

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
# SHAP Feature Importance
# -----------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# LightGBM binary: shap_values가 list면 [1] (양성 클래스)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

shap_df = pd.DataFrame({
    'feature':    feature_cols,
    'mean_shap':  np.abs(sv).mean(axis=0)
}).sort_values('mean_shap', ascending=False)

print("\n===== SHAP Feature Importance =====")
print(shap_df.to_string(index=False))

weak = shap_df[shap_df['mean_shap'] < 0.001]['feature'].tolist()
if weak:
    print(f"\n[제거 후보] mean|SHAP| < 0.001: {weak}")
else:
    print("\n[제거 후보 없음] 모든 피처 기여 중")
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

# -----------------------------
# Walk-Forward 검증
# -----------------------------
print("\n===== Walk-Forward Validation =====")
wf_result = walk_forward_eval(df_tech, feature_cols, target_col)
print(wf_result.to_string(index=False))
print(f"\nWalk-Forward 평균 AUC: {wf_result['auc'].mean():.4f}  (편차: {wf_result['auc'].std():.4f})")
print("AUC 편차가 크면 특정 시장 구간에 과적합됐을 가능성이 높음")