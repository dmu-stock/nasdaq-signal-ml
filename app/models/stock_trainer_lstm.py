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
feature_cols = [

        # raw
        'open',
        'high',
        'low',
        'adj_close',
        'volume',

        # returns
        'change_rate',
        'log_return',

        # candle
        'candle_body',
        'high_low_spread',

        # indicator
        'rsi',
        'macd_hist',
        'bb_percent',
        'volatility_5',

        # market
        'nasdaq_change_rate'
        ]

target_col = 'label'

# ---------------------------------------------------

# Train / Test Split

# ---------------------------------------------------

train_df = df[df['date'] < '2025-07-01'].copy()
test_df = df[df['date'] >= '2025-07-01'].copy()

# ---------------------------------------------------

# 정규화

# ---------------------------------------------------

scaler = StandardScaler()

train_df[feature_cols] = scaler.fit_transform(
train_df[feature_cols]
)

test_df[feature_cols] = scaler.transform(
test_df[feature_cols]
)

# ---------------------------------------------------

# Sequence 생성 함수

# ---------------------------------------------------

SEQ_LEN = 20

def create_sequences(dataframe, feature_cols, target_col):

    X = []
    y = []
    tickers = []
    dates = []

    grouped = dataframe.groupby('ticker')

    for ticker, group in grouped:

        group = group.sort_values('date')

        feature_array = group[feature_cols].values
        target_array = group[target_col].values
        date_array = group['date'].values

        for i in range(len(group) - SEQ_LEN):

            X.append(
                feature_array[i:i+SEQ_LEN]
            )

            y.append(
                target_array[i+SEQ_LEN]
            )

            tickers.append(ticker)

            dates.append(
                date_array[i+SEQ_LEN]
            )

    return (
        np.array(X),
        np.array(y),
        tickers,
        dates
    )

# ---------------------------------------------------

# Sequence 생성

# ---------------------------------------------------

X_train, y_train, train_tickers, train_dates = create_sequences(
    train_df,
    feature_cols,
    target_col
)

X_test, y_test, test_tickers, test_dates = create_sequences(
    test_df,
    feature_cols,
    target_col
)
# ---------------------------------------------------
# Validation Split
# ---------------------------------------------------
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    shuffle=False
)

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

    verbose=1

)

# ---------------------------------------------------

# 예측

# ---------------------------------------------------

pred_prob = model.predict(X_test).flatten()

threshold = 0.35

pred = (pred_prob >= threshold).astype(int)

# ---------------------------------------------------

# 평가

# ---------------------------------------------------

print("\n===== Classification Report =====")

print(
    classification_report(
    y_test,
    pred
    )
)

print(
    f"Accuracy : {accuracy_score(y_test, pred):.4f}"
)

print(
    f"ROC-AUC  : {roc_auc_score(y_test, pred_prob):.4f}"
)

# ---------------------------------------------------
# 예측 결과 저장
# ---------------------------------------------------

result_df = pd.DataFrame({
    'ticker': test_tickers,
    'date': test_dates,
    'actual': y_test,
    'pred': pred,
    'pred_prob': pred_prob
})

result_df = result_df.sort_values(
    by='pred_prob',
    ascending=False
)

result_df.to_csv(
"lstm_prediction_result.csv",
index=False
)

print("\nlstm_prediction_result.csv 저장 완료")

# ---------------------------------------------------
# Top-K 평가
# ---------------------------------------------------

top_50 = result_df.head(50)
top_100 = result_df.head(100)
top_200 = result_df.head(200)

print("\n===== Top-K Performance =====")

print(f"Top 50 상승 비율  : {top_50['actual'].mean():.4f}")
print(top_50)
print(f"Top 100 상승 비율 : {top_100['actual'].mean():.4f}")
print(f"Top 200 상승 비율 : {top_200['actual'].mean():.4f}")

print(f"\n전체 상승 비율 : {result_df['actual'].mean():.4f}")
