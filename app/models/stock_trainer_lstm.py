import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    Activation
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#----------------------------------------------------
# 데이터 로드
# ---------------------------------------------------

# df = pd.read_csv("feature__indicator20260518.csv")
df = pd.read_csv("feature__indicator_lstm20260519.csv")

df['date'] = pd.to_datetime(df['date'])

# ---------------------------------------------------
# Feature 선택
# ---------------------------------------------------

feature_cols = [
        # ===== 방향 흐름 =====
        'log_return',
        'return_1',
        'return_3',

        # ===== 추세 변화 =====
        'momentum_3',
        'momentum_accel_3',
        # 'ma_slope_5',
        # 'volatility_compression',

        # ===== 변동성 흐름 =====
        'atr_change',
        'volatility_regime',
        # ===== 거래량 흐름 =====
        'volume_ratio',
        'volume_change',
        'volume_zscore',
         # ===== 캔들 흐름 =====
        'candle_body',
        'high_low_spread',

        # ===== 시장 동조 =====
        'relative_strength',
]


target_col = 'label'

# ---------------------------------------------------
# 종목별 독립 스케일링
# ---------------------------------------------------
# 종목 내부에서 각각 독립적으로 스케일링
def scale_by_ticker(dataframe, features):
    scaled_df = dataframe.sort_values(['ticker', 'date']).reset_index(drop=True)

    # 2025-07-01 기준으로 데이터 분할 기준점을 잡음 (Data Leakage 방지)
    train_mask = scaled_df['date'] < '2025-07-01'

    for ticker, group in scaled_df.groupby('ticker'):
        ticker_mask = scaled_df['ticker'] == ticker
        train_ticker_mask = ticker_mask & train_mask

        if train_ticker_mask.sum() > 0:
            scaler = StandardScaler()
            # 훈련 데이터로만 fit
            scaler.fit(scaled_df.loc[train_ticker_mask, features])
            # 해당 종목 전체(Train + Test)를 transform
            scaled_df.loc[ticker_mask, features] = scaler.transform(scaled_df.loc[ticker_mask, features])

    return scaled_df

df_scaled = scale_by_ticker(df, feature_cols)



# ---------------------------------------------------
# 시퀀스 생성 후 날짜 기준으로 Train/Test 분할
# ---------------------------------------------------

SEQ_LEN = 20

def create_sequences_all(dataframe, feature_cols, target_col):
    X, y, tickers, dates = [], [], [], []
    grouped = dataframe.groupby('ticker')

    for ticker, group in grouped:
        group = group.sort_values('date')
        if len(group) < SEQ_LEN:
            continue

        f_array = group[feature_cols].values
        t_array = group[target_col].values
        d_array = group['date'].values

        for i in range(len(group) - SEQ_LEN):
            X.append(f_array[i:i+SEQ_LEN])
            y.append(t_array[i+SEQ_LEN])
            tickers.append(ticker)
            dates.append(d_array[i+SEQ_LEN])

    return np.array(X), np.array(y), tickers, np.array(dates)


X_all, y_all, tickers_all, dates_all = create_sequences_all(
    df_scaled,
    feature_cols,
    target_col
)

train_idx = dates_all < pd.to_datetime('2025-07-01')
test_idx = dates_all >= pd.to_datetime('2025-07-01')

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_test, y_test = X_all[test_idx], y_all[test_idx]
test_tickers_split = [tickers_all[i] for i, val in enumerate(test_idx) if val]
test_dates_split = dates_all[test_idx]

# ---------------------------------------------------
# 시계열 순서를 보장하는 Val 분할
# ---------------------------------------------------
# 종목 셔플 분할 오류를 막기 위해, Train 데이터 내부에서 가장 최신 날짜 기준 20%를 Val
train_dates_only = dates_all[train_idx]
split_time = np.percentile(train_dates_only, 80) # 하위 80% 시점 날짜 추출

final_train_idx = train_dates_only <= split_time
val_idx = train_dates_only > split_time

X_train_final, y_train_final = X_train[final_train_idx], y_train[final_train_idx]
X_val, y_val = X_train[val_idx], y_train[val_idx]

print("X_train_final:", X_train_final.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

# ---------------------------------------------------
# LSTM 모델
# ---------------------------------------------------

model = Sequential([
    LSTM(
        64,
        return_sequences=True,
        input_shape=(SEQ_LEN, len(feature_cols)),
    ),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(
        32,
        return_sequences=False,
    ),
     BatchNormalization(),
    Dropout(0.3),

    # 완전 연결 레이어: he_normal 초기화 추가로 Relu 효율 극대화
    Dense(16, kernel_initializer='he_normal'),
    BatchNormalization(),
    Activation('relu'),
    
    Dense(1, activation='sigmoid')

])

# ---------------------------------------------------
# Compile
# ---------------------------------------------------
opt = Adam(learning_rate=0.0005)
model.compile(
    optimizer=opt,
    loss='binary_crossentropy',
    metrics=[
        'accuracy', 
        tf.keras.metrics.AUC(name='auc') # 'auc'라는 이름으로 메트릭 추가
    ]
)

# ---------------------------------------------------
# Early Stopping
# ---------------------------------------------------

early_stop = EarlyStopping(
    monitor='val_auc',
    patience=10,
    mode='max',
    restore_best_weights=True
)

# ---------------------------------------------------
# Class Weight
# ---------------------------------------------------

classes = np.unique(y_train_final)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train_final
)

class_weights = dict(enumerate(class_weights))

print(class_weights)

# ---------------------------------------------------
# 학습
# ---------------------------------------------------

history = model.fit(
    X_train_final,
    y_train_final,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=128,
    callbacks=[early_stop],
    class_weight=class_weights,
    verbose=1,
)

# ---------------------------------------------------
# 예측
# ---------------------------------------------------

pred_prob = model.predict(X_test).flatten()

# 전체 체점용
threshold = 0.53

pred = (pred_prob >= threshold).astype(int)

# ---------------------------------------------------
# 평가
# ---------------------------------------------------

# 평가 리포트 출력
print("\n===== Classification Report =====")
print(classification_report(y_test, pred))
print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, pred_prob):.4f}")

# ---------------------------------------------------
# 예측 결과 저장
# ---------------------------------------------------

result_df = pd.DataFrame({
    'ticker': test_tickers_split,
    'date': test_dates_split,
    'actual': y_test,
    'pred': pred,
    'pred_prob': pred_prob
})

# 날짜별 정렬
result_df = result_df.sort_values(
    ['date', 'pred_prob'],
    ascending=[True, False]
)

result_df.to_csv(
"lstm_prediction_result.csv",
index=False
)

print("\nlstm_prediction_result.csv 저장 완료")
print(len(X_all), len(y_all), len(tickers_all), len(dates_all))

# ---------------------------------------------------
# Top-K 평가
# ---------------------------------------------------
print("\n===== Daily Top-K Performance =====")

daily_top3_actuals = []
daily_top5_actuals = []
CONFIDENCE_THRESHOLD = 0.65

print(f"테스트 데이터 시작 날짜: {result_df['date'].min()}")
print(f"테스트 데이터 마지막 날짜: {result_df['date'].max()}")
print(f"총 테스트 일수 (영업일 기준): {result_df['date'].nunique()}일")

for date, group in result_df.groupby('date'):
    
    # 그날 뱉은 확률 중 가장 높은 순으로 정렬
    group_sorted = group.sort_values('pred_prob', ascending=False)

    top3 = group_sorted.head(3)
    top5 = group_sorted.head(5)

    valid_picks = top3[top3['pred_prob'] >= CONFIDENCE_THRESHOLD]
    valid_picks = top5[top5['pred_prob'] >= CONFIDENCE_THRESHOLD]

    if not valid_picks.empty:
        daily_top3_actuals.extend(valid_picks['actual'].tolist())
        daily_top5_actuals.extend(valid_picks['actual'].tolist())

    real_top3_accuracy = np.mean(daily_top3_actuals) if daily_top3_actuals else 0
    real_top5_accuracy = np.mean(daily_top5_actuals) if daily_top5_actuals else 0




print(f"가드레일 기준: 예측 확률 {CONFIDENCE_THRESHOLD} 이상인 종목만 진입")
print(f"매일 랭킹 Top 3 매수 시 진짜 타율 : {real_top3_accuracy:.4f}")
print(f"총 매수 진입 횟수 (종목 수)       : {len(daily_top3_actuals)}개")

print(f"매일 랭킹 Top 5 매수 시 진짜 타율 : {real_top5_accuracy:.4f}")
print(f"총 매수 진입 횟수 (종목 수)       : {len(daily_top3_actuals)}개")

print(f"시장 평균 상승 비율 (Baseline)     : {result_df['actual'].mean():.4f}")
