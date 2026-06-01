import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import joblib
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from app.config.config import LSTM_FEATURE_COLS,LSTM_TEST_FEATURE_COLS
from app.models.lstm_model import DualLSTMModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------------------
# GPU 확인
# ---------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Device] {device}")
if device.type == 'cuda':
    torch.cuda.manual_seed_all(42)
    print(f"  GPU  : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

use_amp = device.type == 'cuda'
scaler_amp = torch.amp.GradScaler('cuda', enabled=use_amp)

# ---------------------------------------------------
# Data load
# ---------------------------------------------------
df = pd.read_csv("feature__indicator_lstm20260601.csv")
df['date'] = pd.to_datetime(df['date'])

feature_cols = LSTM_FEATURE_COLS
target_col   = 'label'

# ---------------------------------------------------
# Ticker-level scaling (train-only fit, leakage-free)
# ---------------------------------------------------
def scale_by_ticker(dataframe, features):
    scalers = {}
    scaled_df = dataframe.sort_values(['ticker', 'date']).reset_index(drop=True).copy()
    scaled_df[features] = scaled_df[features].replace([np.inf, -np.inf], np.nan)
    scaled_df = scaled_df.dropna(subset=features).reset_index(drop=True)

    train_mask = scaled_df['date'] < '2025-07-01'

    for ticker, _ in scaled_df.groupby('ticker'):
        ticker_mask       = scaled_df['ticker'] == ticker
        train_ticker_mask = ticker_mask & train_mask
        test_ticker_mask  = ticker_mask & (~train_mask)

        if train_ticker_mask.sum() == 0:
            continue

        sc = StandardScaler()
        sc.fit(scaled_df.loc[train_ticker_mask, features])
        scaled_df.loc[train_ticker_mask, features] = sc.transform(
            scaled_df.loc[train_ticker_mask, features]
        )
        if test_ticker_mask.sum() > 0:
            scaled_df.loc[test_ticker_mask, features] = sc.transform(
                scaled_df.loc[test_ticker_mask, features]
            )
        scalers[ticker] = sc

    # 추론시 재사용
    joblib.dump(scalers, 'ticker_scalers.pkl')
    return scaled_df

df_scaled = scale_by_ticker(df, feature_cols)

# ---------------------------------------------------
# Sequence generation
# ---------------------------------------------------
SEQ_LEN_20 = 20
SEQ_LEN_60 = 60

def create_sequences_all(dataframe, feature_cols, target_col):
    X_20, X_60, y, tickers, dates = [], [], [], [], []
    for ticker, group in dataframe.groupby('ticker'):
        group = group.sort_values('date')
        if len(group) < SEQ_LEN_60:
            continue
        f_arr = group[feature_cols].values
        t_arr = group[target_col].values
        d_arr = group['date'].values
        for i in range(SEQ_LEN_60 - 1, len(group)):
            X_20.append(f_arr[i - (SEQ_LEN_20 - 1): i + 1])
            X_60.append(f_arr[i - (SEQ_LEN_60 - 1): i + 1])
            y.append(t_arr[i])
            tickers.append(ticker)
            dates.append(d_arr[i])
    return (
        np.array(X_20, dtype=np.float32),
        np.array(X_60, dtype=np.float32),
        np.array(y,    dtype=np.float32),
        tickers,
        np.array(dates),
    )

X_20_all, X_60_all, y_all, tickers_all, dates_all = create_sequences_all(
    df_scaled, feature_cols, target_col
)

# ---------------------------------------------------
# Train / Val / Test split (time-based, no shuffle)
# ---------------------------------------------------
dates_all_dt = pd.to_datetime(dates_all)

train_mask = dates_all_dt < pd.Timestamp('2025-07-01')
test_mask  = dates_all_dt >= pd.Timestamp('2025-07-01')

X_20_train, X_20_test = X_20_all[train_mask], X_20_all[test_mask]
X_60_train, X_60_test = X_60_all[train_mask], X_60_all[test_mask]
y_train, y_test       = y_all[train_mask],    y_all[test_mask]

test_tickers = [tickers_all[i] for i, v in enumerate(test_mask) if v]
test_dates   = dates_all[test_mask]

# Val: latest 20% of train set by date (time-safe)
train_dates_ns  = dates_all_dt[train_mask].astype(np.int64).values
split_time_ns   = np.percentile(train_dates_ns, 80)

final_train_mask = train_dates_ns <= split_time_ns
val_mask         = train_dates_ns >  split_time_ns

X_20_tr, X_20_val = X_20_train[final_train_mask], X_20_train[val_mask]
X_60_tr, X_60_val = X_60_train[final_train_mask], X_60_train[val_mask]
y_tr, y_val       = y_train[final_train_mask],    y_train[val_mask]

print("--- 데이터셋 ---")
print(f"Train : {len(y_tr):>7,}  |  Val : {len(y_val):>6,}  |  Test : {len(y_test):>6,}")

# ---------------------------------------------------
# DataLoader
# ---------------------------------------------------
BATCH_SIZE  = 128
PIN_MEMORY  = device.type == 'cuda'

def make_loader(x20, x60, y, shuffle=True):
    ds = TensorDataset(
        torch.from_numpy(x20),
        torch.from_numpy(x60),
        torch.from_numpy(y),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, pin_memory=PIN_MEMORY)

train_loader = make_loader(X_20_tr,   X_60_tr,   y_tr)
val_loader   = make_loader(X_20_val,  X_60_val,  y_val,   shuffle=False)
test_loader  = make_loader(X_20_test, X_60_test, y_test,  shuffle=False)

# ---------------------------------------------------
# Model
# ---------------------------------------------------
num_features = len(feature_cols)
model = DualLSTMModel(num_features).to(device)
print(model)
print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

# ---------------------------------------------------
# Loss: BCEWithLogitsLoss + pos_weight for class imbalance
# ---------------------------------------------------
classes = np.unique(y_tr)
cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
cw_dict = dict(zip(classes.astype(int), cw))
pos_weight = torch.tensor([cw_dict[1] / cw_dict[0]], dtype=torch.float32).to(device)
print(f"[클래스 가중치] neg={cw_dict[0]:.4f}  pos={cw_dict[1]:.4f}  pos_weight={pos_weight.item():.4f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# weight_decay: L2 정규화 → val→test 갭(과적합) 완화
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
# LR 스케줄러: val_auc 3 epoch 정체 시 LR 절반 → 수렴 품질 향상
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-5
)

# ---------------------------------------------------
# Training loop with early stopping on val AUC
# ---------------------------------------------------
EPOCHS   = 100
PATIENCE = 15

best_auc       = 0.0
patience_count = 0
best_weights   = None

for epoch in range(1, EPOCHS + 1):
    # ----- Train -----
    model.train()
    total_loss = 0.0
    for x20_b, x60_b, y_b in train_loader:
        x20_b = x20_b.to(device, non_blocking=True)
        x60_b = x60_b.to(device, non_blocking=True)
        y_b   = y_b.to(device,   non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x20_b, x60_b)
            loss   = criterion(logits, y_b)
        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()
        total_loss += loss.item() * len(y_b)

    avg_loss = total_loss / len(y_tr)

    # ----- Validate -----
    model.eval()
    val_probs, val_labels = [], []
    with torch.no_grad():
        for x20_b, x60_b, y_b in val_loader:
            x20_b = x20_b.to(device, non_blocking=True)
            x60_b = x60_b.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x20_b, x60_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            val_probs.extend(probs.tolist())
            val_labels.extend(y_b.numpy().tolist())

    val_auc = roc_auc_score(val_labels, val_probs)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:>3}/{EPOCHS}  loss={avg_loss:.4f}  val_auc={val_auc:.4f}  lr={current_lr:.2e}")

    scheduler.step(val_auc)   # val_auc 기준 LR 조정

    if val_auc > best_auc:
        best_auc     = val_auc
        best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"Early stopping at epoch {epoch}  (best val_auc={best_auc:.4f})")
            break

# Restore best weights and save
model.load_state_dict(best_weights)
torch.save(
    {'model_state_dict': best_weights, 'num_features': num_features},
    'best_multi_input_lstm.pt',
)
print(f"\n[저장] best_multi_input_lstm.pt  (best val_auc={best_auc:.4f})")

# ---------------------------------------------------
# Test evaluation
# ---------------------------------------------------
model.eval()
pred_prob = []
with torch.no_grad():
    for x20_b, x60_b, _ in test_loader:
        x20_b = x20_b.to(device, non_blocking=True)
        x60_b = x60_b.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x20_b, x60_b)
        pred_prob.extend(torch.sigmoid(logits).cpu().numpy().tolist())

pred_prob = np.array(pred_prob)
threshold = 0.53
pred      = (pred_prob >= threshold).astype(int)

print("\n===== Classification Report =====")
print(classification_report(y_test.astype(int), pred))
print(f"Accuracy : {accuracy_score(y_test.astype(int), pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, pred_prob):.4f}")

# ---------------------------------------------------
# Prediction result CSV
# ---------------------------------------------------
result_df = pd.DataFrame({
    'ticker':    test_tickers,
    'date':      test_dates,
    'actual':    y_test.astype(int),
    'pred':      pred,
    'pred_prob': pred_prob,
})
result_df = result_df.sort_values(['date', 'pred_prob'], ascending=[True, False])
result_df.to_csv("lstm_prediction_result.csv", index=False)
print("\nlstm_prediction_result.csv 저장 완료")

# ---------------------------------------------------
# Daily Top-K evaluation
# ---------------------------------------------------
print("\n===== Daily Top-K Performance =====")
CONFIDENCE_THRESHOLD = 0.65
daily_top3_actuals   = []
daily_top5_actuals   = []

print(f"테스트 시작: {result_df['date'].min()}")
print(f"테스트 마감: {result_df['date'].max()}")
print(f"총 테스트 영업일: {result_df['date'].nunique()}일")

for date, group in result_df.groupby('date'):
    group_sorted = group.sort_values('pred_prob', ascending=False)
    valid_top3 = group_sorted.head(3)
    valid_top5 = group_sorted.head(5)
    valid_top3 = valid_top3[valid_top3['pred_prob'] >= CONFIDENCE_THRESHOLD]
    valid_top5 = valid_top5[valid_top5['pred_prob'] >= CONFIDENCE_THRESHOLD]
    if not valid_top3.empty:
        daily_top3_actuals.extend(valid_top3['actual'].tolist())
    if not valid_top5.empty:
        daily_top5_actuals.extend(valid_top5['actual'].tolist())

real_top3 = np.mean(daily_top3_actuals) if daily_top3_actuals else 0.0
real_top5 = np.mean(daily_top5_actuals) if daily_top5_actuals else 0.0

print(f"가드레일: 예측 확률 {CONFIDENCE_THRESHOLD} 이상만 진입")
print(f"Top3 타율: {real_top3:.4f}  ({len(daily_top3_actuals)}회 진입)")
print(f"Top5 타율: {real_top5:.4f}  ({len(daily_top5_actuals)}회 진입)")
print(f"베이스라인 (시장 평균): {result_df['actual'].mean():.4f}")
