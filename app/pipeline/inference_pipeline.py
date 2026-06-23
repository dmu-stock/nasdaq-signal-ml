import pandas as pd
import numpy as np
import joblib
import torch
from datetime import datetime
from app.config.config import GBM_FEATURE_COLS, LSTM_FEATURE_COLS, TICKERS
from app.models.lstm_model import DualLSTMModel
from app.collector.price_yfinance import fetch_all_stocks_price_data
from app.features.processor import FeatureProcessorGBM
from app.features.processor_lstm import FeatureProcessorLSTM

# ---------------------------------------------------
# 가드레일 & 타임프레임 스펙
# ---------------------------------------------------
GBM_MIN     = 0.50   # GBM 최소 기준 (이 아래는 후보 제외 → 현금 보유 여지)
TOP_N_GBM   = 8      # GBM 상위 후보 풀 크기 (시장 추천 3종목의 2~3배)
SEQ_LEN_20  = 20
SEQ_LEN_60  = 60

print("듀얼 파이프라인 실전 추론")

# ===================================================
# 1. 모델 & 스케일러 로드
# ===================================================
lgb_model = joblib.load("best_lgbm_model.pkl")

_device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_ckpt     = torch.load('best_multi_input_lstm.pt', map_location=_device)
lstm_model = DualLSTMModel(_ckpt['num_features']).to(_device)
lstm_model.load_state_dict(_ckpt['model_state_dict'])
lstm_model.eval()

scalers = joblib.load('ticker_scalers.pkl')
print(f"모델 로드 완료  (device={_device})")

# ===================================================
# 2. 주가 데이터 수집
# ===================================================
print("야후 파이낸스 데이터 수집 중...")
df_raw = fetch_all_stocks_price_data(tickers=TICKERS, period="2y")

if df_raw.empty:
    print("[에러] 마켓 데이터 수집 실패")
    exit()

vix_now = float(df_raw['vix'].iloc[-1])
print(f"현재 VIX: {vix_now:.2f}")
if vix_now >= 30:
    print("VIX 30 초과 → 극공포 구간, 매수 중단")
    exit()

# ===================================================
# 3. 피처 엔지니어링
# ===================================================
gbm_proc  = FeatureProcessorGBM()
lstm_proc = FeatureProcessorLSTM()

df_gbm  = gbm_proc.calc_technical_indicators(df_raw.copy(), is_inference=True)
df_lstm = lstm_proc.calc_technical_indicators(df_raw.copy(), is_inference=True)

df_gbm  = df_gbm.replace([np.inf, -np.inf], np.nan)
df_lstm = df_lstm.replace([np.inf, -np.inf], np.nan)

# ===================================================
# 4. 종목별 추론 루프
# ===================================================
_DRIFT_THRESHOLD = 4.0
results = []

for ticker in TICKERS:
    # -----------------------------------------------
    # LightGBM
    # -----------------------------------------------
    tg = df_gbm[df_gbm['ticker'] == ticker].sort_values('date')
    if tg.empty:
        continue

    prob_lgb = lgb_model.predict_proba(tg[GBM_FEATURE_COLS].iloc[[-1]])[0][1]

    # -----------------------------------------------
    # LSTM
    # -----------------------------------------------
    tl = df_lstm[df_lstm['ticker'] == ticker].sort_values('date')
    if len(tl) < SEQ_LEN_60:
        continue

    if ticker not in scalers:
        print(f"[경고] {ticker} 스케일러 없음, 스킵")
        continue

    sc     = scalers[ticker]
    seq_20 = tl[LSTM_FEATURE_COLS].iloc[-SEQ_LEN_20:].values
    seq_60 = tl[LSTM_FEATURE_COLS].iloc[-SEQ_LEN_60:].values

    seq_20_scaled = sc.transform(pd.DataFrame(seq_20, columns=LSTM_FEATURE_COLS))
    seq_60_scaled = sc.transform(pd.DataFrame(seq_60, columns=LSTM_FEATURE_COLS))

    # 학습 분포 이탈 감지
    drift_cols = [
        LSTM_FEATURE_COLS[j]
        for j in range(len(LSTM_FEATURE_COLS))
        if np.abs(seq_60_scaled[:, j]).max() > _DRIFT_THRESHOLD
    ]
    if drift_cols:
        print(f"[분포 경고] {ticker} 이탈 피처 (|z|>{_DRIFT_THRESHOLD:.0f}): {drift_cols}")

    t20 = torch.tensor(seq_20_scaled, dtype=torch.float32).unsqueeze(0).to(_device)
    t60 = torch.tensor(seq_60_scaled, dtype=torch.float32).unsqueeze(0).to(_device)

    with torch.no_grad():
        prob_lstm = float(torch.sigmoid(lstm_model(t20, t60)).cpu().item())

    # -----------------------------------------------
    # 조화평균 앙상블
    # -----------------------------------------------
    final_prob = 2 * (prob_lgb * prob_lstm) / (prob_lgb + prob_lstm + 1e-9)

    results.append({
        'ticker':     ticker,
        'prob_lgb':   round(prob_lgb,   4),
        'prob_lstm':  round(prob_lstm,  4),
        'final_prob': round(final_prob, 4),
    })

# ===================================================
# 5. 후보 선정 (GBM 후보 → LSTM 재정렬)
#    1단계: GBM_MIN 이상인 종목 중 prob_lgb 상위 TOP_N_GBM 후보
#    2단계: 그 후보 중 prob_lstm 상위 3종목 최종 선택
# ===================================================
inference_df = pd.DataFrame(results)

candidates = (
    inference_df[inference_df['prob_lgb'] >= GBM_MIN]
    .sort_values('prob_lgb', ascending=False)
    .head(TOP_N_GBM)
)
valid_picks = candidates.sort_values('prob_lstm', ascending=False).head(3)

# ===================================================
# 6. 최종 출력
# ===================================================
print("\n=============================================")
print(f"[주먹봇] {datetime.now().strftime('%Y-%m-%d')} 매수 시그널")
print("=============================================")
if not valid_picks.empty:
    for _, row in valid_picks.iterrows():
        print(
            f"진입: {row['ticker']}  score={row['final_prob']:.4f} "
            f"[LGBM={row['prob_lgb']:.4f} / LSTM={row['prob_lstm']:.4f}]"
        )
else:
    print(f"GBM {GBM_MIN} 이상 종목 없음 — 현금 보유 권장")
print("=============================================\n")

# 후보 풀 (GBM 통과) 표시
print(f"GBM 후보 풀 (prob_lgb >= {GBM_MIN}, 상위 {TOP_N_GBM}):")
if candidates.empty:
    print("  없음")
else:
    print(candidates.sort_values('prob_lstm', ascending=False).to_string(index=False))

print(f"\n추론 종목 수: {len(inference_df)}")
print(inference_df.sort_values('prob_lgb', ascending=False).to_string(index=False))
