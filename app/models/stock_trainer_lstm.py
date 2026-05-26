import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
from app.config.config import LSTM_FEATURE_COLS

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
    Concatenate
    
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#----------------------------------------------------
# 데이터 로드
# ---------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# df = pd.read_csv("feature__indicator20260518.csv")
df = pd.read_csv("feature__indicator_lstm20260526.csv")

df['date'] = pd.to_datetime(df['date'])

# ---------------------------------------------------
# Feature 선택
# ---------------------------------------------------

feature_cols = LSTM_FEATURE_COLS
target_col = 'label'

# ---------------------------------------------------
# 종목별 독립 스케일링
# ---------------------------------------------------
# 종목 내부에서 각각 독립적으로 스케일링
def scale_by_ticker(dataframe, features):
    scalers = {}
    # 원본 데이터 보호를 위해 복사본 생성
    scaled_df = dataframe.sort_values(['ticker', 'date']).reset_index(drop=True).copy()

    scaled_df[features] = scaled_df[features].replace([np.inf, -np.inf], np.nan)
    scaled_df = scaled_df.dropna(subset=features).reset_index(drop=True)
    
    # 훈련/테스트 구분 마스크
    train_mask = scaled_df['date'] < '2025-07-01'

    for ticker, group in scaled_df.groupby('ticker'):
        ticker_mask = scaled_df['ticker'] == ticker
        
        # Train과 Test 기간 분리
        train_ticker_mask = ticker_mask & train_mask
        test_ticker_mask = ticker_mask & (~train_mask)

        if train_ticker_mask.sum() > 0:
            scaler = StandardScaler()
            
            # 훈련 데이터로만 fit (미래 정보 차단)
            scaler.fit(scaled_df.loc[train_ticker_mask, features])
            
            # 훈련 데이터와 테스트 데이터를 각각 transform
            scaled_df.loc[train_ticker_mask, features] = scaler.transform(scaled_df.loc[train_ticker_mask, features])
            
            # 테스트 데이터 변환 
            if test_ticker_mask.sum() > 0:
                scaled_df.loc[test_ticker_mask, features] = scaler.transform(scaled_df.loc[test_ticker_mask, features])
            
            scalers[ticker] = scaler
            
    joblib.dump(scalers, 'ticker_scalers.pkl')
    return scaled_df

df_scaled = scale_by_ticker(df, feature_cols)



# ---------------------------------------------------
# 시퀀스 생성 후 날짜 기준으로 Train/Test 분할
# ---------------------------------------------------

SEQ_LEN_20 = 20
SEQ_LEN_60 = 60

def create_sequences_all(dataframe, feature_cols, target_col):
    X_20, X_60, y, tickers, dates = [], [], [], [], []
    grouped = dataframe.groupby('ticker')

    for ticker, group in grouped:
        group = group.sort_values('date')

        if len(group) < SEQ_LEN_60:
            continue

        f_array = group[feature_cols].values
        t_array = group[target_col].values
        d_array = group['date'].values

        for i in range(SEQ_LEN_60 - 1, len(group)):
            # 1) 20일 시퀀스 (오늘 기준 과거 20일: i-19 부터 i 까지)
            X_20.append(f_array[i - (SEQ_LEN_20 - 1) : i + 1])
            
            # 2) 60일 시퀀스 (오늘 기준 과거 60일: i-59 부터 i 까지)
            X_60.append(f_array[i - (SEQ_LEN_60 - 1) : i + 1])
            
            y.append(t_array[i])
            tickers.append(ticker)
            dates.append(d_array[i])

    return np.array(X_20), np.array(X_60), np.array(y), tickers, np.array(dates)


X_20_all, X_60_all, y_all, tickers_all, dates_all = create_sequences_all(
    df_scaled,
    feature_cols,
    target_col
)

# ------------------------------------------------------------------
# 2. 시계열 기준 데이터 분할 (Train / Test / Val)
# ------------------------------------------------------------------
# 날짜형 정렬 유지 위해 Timestamp 변환 후 마스킹 처리
dates_all_dt = pd.to_datetime(dates_all)

train_idx = dates_all < pd.to_datetime('2025-07-01')
test_idx = dates_all >= pd.to_datetime('2025-07-01')

X_20_train, X_20_test = X_20_all[train_idx], X_20_all[test_idx]
X_60_train, X_60_test = X_60_all[train_idx], X_60_all[test_idx]

y_train, y_test = y_all[train_idx], y_all[test_idx]

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

X_20_train_final, X_20_val = X_20_train[final_train_idx], X_20_train[val_idx]
X_60_train_final, X_60_val = X_60_train[final_train_idx], X_60_train[val_idx]
y_train_final, y_val = y_train[final_train_idx], y_train[val_idx]

print("--- 데이터셋 분할 ---")
print("X_20_train_final:", X_20_train_final.shape, " | X_60_train_final:", X_60_train_final.shape)
print("X_20_val:", X_20_val.shape, "         | X_60_val:", X_60_val.shape)
print("X_20_test:", X_20_test.shape, "       | X_60_test:", X_60_test.shape)

# ---------------------------------------------------
# LSTM 모델
# ---------------------------------------------------
num_features = len(feature_cols)

# 다중 입력 레이어 정의
input_20 = Input(shape=(SEQ_LEN_20, num_features), name='Input_20d')
input_60 = Input(shape=(SEQ_LEN_60, num_features), name='Input_60d')

# 채널 1: 20일선 추세 추출 레이어
lstm_20_1 = LSTM(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(input_20)
lstm_20_2 = LSTM(32, return_sequences=False, dropout=0.3, recurrent_dropout=0.2)(lstm_20_1)

# 채널 2: 60일선 중기 추세 추출 레이어
lstm_60_1 = LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(input_60)
lstm_60_2 = LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(lstm_60_1)
lstm_60_3 = LSTM(32, return_sequences=False, dropout=0.3, recurrent_dropout=0.2)(lstm_60_2)

merged = Concatenate()([lstm_20_2, lstm_60_3])

#단계적 dense
dense1 = Dense(48, kernel_initializer='he_normal')(merged)
act1 = Activation('relu')(BatchNormalization()(dense1))
drop1 = Dropout(0.3)(act1)

dense2 = Dense(16, kernel_initializer='he_normal')(drop1)
act2 = Activation('relu')(BatchNormalization()(dense2))
drop2 = Dropout(0.2)(act2)

# 최종 출력
output = Dense(1, activation='sigmoid', name='Output_Probability')(drop2)

model = Model(inputs=[input_20, input_60], outputs=output)
# ---------------------------------------------------
# Compile
# ---------------------------------------------------
opt = Adam(learning_rate=0.0005)
model.compile(
    optimizer=opt,
    loss='binary_crossentropy',
    metrics=[
        'accuracy', 
        tf.keras.metrics.AUC(name='auc')
    ]
)
model.summary()

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
    x=[X_20_train_final, X_60_train_final],
    y=y_train_final,
    validation_data=([X_20_val, X_60_val], y_val),
    epochs=50,
    batch_size=128,
    callbacks=[early_stop],
    class_weight=class_weights,
    verbose=1,
)
model.save('best_multi_input_lstm.keras')

# ---------------------------------------------------
# 예측
# ---------------------------------------------------

pred_prob = model.predict([X_20_test, X_60_test]).flatten()

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
print(len(X_20_all), len(y_all), len(tickers_all), len(dates_all))

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

    valid_top3 = top3[top3['pred_prob'] >= CONFIDENCE_THRESHOLD]
    valid_top5 = top5[top5['pred_prob'] >= CONFIDENCE_THRESHOLD]

    if not valid_top3.empty:
        daily_top3_actuals.extend(valid_top3['actual'].tolist())
    if not valid_top5.empty:
        daily_top5_actuals.extend(valid_top5['actual'].tolist())

    real_top3_accuracy = np.mean(daily_top3_actuals) if daily_top3_actuals else 0
    real_top5_accuracy = np.mean(daily_top5_actuals) if daily_top5_actuals else 0

print(f"가드레일 기준: 예측 확률 {CONFIDENCE_THRESHOLD} 이상인 종목만 진입")
print(f"매일 랭킹 Top 3 매수 시 진짜 타율 : {real_top3_accuracy:.4f}")
print(f"총 매수 진입 횟수 (종목 수)       : {len(daily_top3_actuals)}개")

print(f"매일 랭킹 Top 5 매수 시 진짜 타율 : {real_top5_accuracy:.4f}")
print(f"총 매수 진입 횟수 (종목 수)       : {len(daily_top3_actuals)}개")

print(f"시장 평균 상승 비율 (Baseline)     : {result_df['actual'].mean():.4f}")
