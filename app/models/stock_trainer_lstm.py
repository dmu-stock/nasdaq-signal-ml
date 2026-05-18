import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------

# 데이터 로드

# ---------------------------------------------------

# df = pd.read_csv("feature__indicator20260518.csv")
df = pd.read_csv("feature__indicator_lstm20260518.csv")

df['date'] = pd.to_datetime(df['date'])

# ---------------------------------------------------

# Feature 선택

# ---------------------------------------------------

# feature_cols = [
#     'alpha',
#     'alpha_5',
#     'alpha_20',
#     'alpha_divergence',

#     'ma_ratio',

#     'rsi',

#     'volatility_5',

#     'volume_ratio',

#     'nasdaq_change_rate',

#     'macd_hist',

#     'drawdown_20',

#     'bb_percent',

#     'tr_5'

# ]
# feature_cols = [

#         # raw
#         'open',
#         'high',
#         'low',
#         'adj_close',
#         'volume',

#         # returns
#         'change_rate',
#         'log_return',

#         # candle
#         'candle_body',
#         'high_low_spread',

#         # indicator
#         'rsi',
#         'macd_hist',
#         'bb_percent',
#         'volatility_5',

#         # market
#         'nasdaq_change_rate'
#         ]
feature_cols = [
            # 1. 가격의 변화율 및 캔들 모양 (이미 0을 기준으로 정규화된 형태)
            'change_rate',
            'log_return',
            'candle_body',
            'high_low_spread',
            
            # 2. 거래량 지표 (Raw volume은 절대 금지, 비율로 치환된 것만 사용)
            'volume_ratio',
            'volume_change',
            
            # 3. 보조지표 (0~1 사이 혹은 스케일이 안정적인 녀석들)
            'bb_percent',     # 0~1 사이 값이라 LSTM이 환장하고 좋아함
            'volatility_5',   # 변동성 크기 표준편차
            'drawdown_20',     # 최고점 대비 낙폭 비율 (-0.1, -0.2 등)
            'macd_hist',      # 단기 에너지
            
            # 4. 시장 리스크 및 타겟
            'nasdaq_change_rate'
        ]

target_col = 'label'

# ---------------------------------------------------
# 종목별 독립 스케일링
# ---------------------------------------------------
# 전체 통짜 정규화 대신, 종목 내부에서 각각 독립적으로 스케일링을 먹입니다.
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
        input_shape=(SEQ_LEN, len(feature_cols)),
        return_sequences=False
    ),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')

])

# ---------------------------------------------------

# Compile

# ---------------------------------------------------

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ---------------------------------------------------

# Early Stopping

# ---------------------------------------------------

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
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

    epochs=30,

    batch_size=64,

    callbacks=[early_stop],

    verbose=1,

    class_weight=class_weights

)

# ---------------------------------------------------

# 예측

# ---------------------------------------------------

pred_prob = model.predict(X_test).flatten()

threshold = 0.6

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
}).sort_values(by='pred_prob', ascending=False)

result_df.to_csv(
"lstm_prediction_result.csv",
index=False
)

print("\nlstm_prediction_result.csv 저장 완료")
print(len(X_all), len(y_all), len(tickers_all), len(dates_all))

# ---------------------------------------------------
# Top-K 평가
# ---------------------------------------------------

top_50 = result_df.drop_duplicates(subset=['ticker']).head(50)
top_100 = result_df.head(100)
top_200 = result_df.head(200)

print("\n===== Top-K Performance =====")

print(f"Top 50 상승 비율  : {top_50['actual'].mean():.4f}")
print(top_50)
print(f"Top 100 상승 비율 : {top_100['actual'].mean():.4f}")
print(f"Top 200 상승 비율 : {top_200['actual'].mean():.4f}")

print(f"\n전체 상승 비율 : {result_df['actual'].mean():.4f}")
