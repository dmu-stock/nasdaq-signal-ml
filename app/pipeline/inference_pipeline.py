import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime
from app.config.config import GBM_FEATURE_COLS,LSTM_FEATURE_COLS
from tensorflow.keras.models import load_model

# ===================================================
#  함수/클래스 호출
# ===================================================
# 데이터 수집 모듈에서 일괄 수집 함수 가져오기
from app.collector.price_yfinance import fetch_all_stocks_price_data  

# GBM, LSTM 전처리 파일에서 클래스/함수 가져오기
from app.features.processor import FeatureProcessorGBM
from app.features.processor_lstm import FeatureProcessorLSTM   

# ---------------------------------------------------
# 가드레일 & 타임프레임 스펙 고정
# ---------------------------------------------------
LGBM_THRESHOLD = 0.65
LSTM_THRESHOLD = 0.45
SEQ_LEN_20 = 20
SEQ_LEN_60 = 60

# 주먹 봇 핵심 관리 대장주 리스트
TICKERS = [
    "NVDA",
    "AMD",
    "AVGO",
    "MU",
    "QCOM",

    "MSFT",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",

    "NFLX",
    "PLTR",
    "SNOW",
    "CRWD",
    "PANW",
    "^SOX"
]

print("독립 듀얼 파이프라인 실전 추론 테스트")

# ===================================================
# 2. 저장되어 있는 최종 가중치 파일 로드
# ===================================================
lgb_model = joblib.load("best_lgbm_model.pkl")
lstm_model = load_model('best_multi_input_lstm.keras')
scalers = joblib.load('ticker_scalers.pkl')
print("✅ LightGBM & Multi-Input LSTM 가중치 로드 완료.")

# ===================================================
# 데이터 수집 (최근 120일치)
# ===================================================
print("야후 파이낸스 최신 마켓 데이터 및 나스닥 등락율 결합")

df_raw = fetch_all_stocks_price_data(tickers=TICKERS, period="1y")

if df_raw.empty:
    print("[에러] 최신 마켓 데이터 수집에 실패")
    exit()

# ===================================================
# 듀얼 프로세서 가동 (is_inference=True 필수 지정)
# ===================================================

# 클래스 인스턴스 생성
gbm_proc = FeatureProcessorGBM()
lstm_proc = FeatureProcessorLSTM()

# 원본 데이터 오염 방지를 위해 .copy() 전달 및 실전 추론 플래그 True 작동
df_features_gbm = gbm_proc.calc_technical_indicators(df_raw.copy(), is_inference=True)
df_features_lstm = lstm_proc.calc_technical_indicators(df_raw.copy(), is_inference=True)

# 무한대값 정리
df_features_gbm = df_features_gbm.replace([np.inf, -np.inf], np.nan)
df_features_lstm = df_features_lstm.replace([np.inf, -np.inf], np.nan)

# ===================================================
# 종목별 독립 추론 루프 가동
# ===================================================
results = []

for ticker in TICKERS:
    # -----------------------------------------------
    # LightGBM 실전 추론
    # -----------------------------------------------
    ticker_gbm = df_features_gbm[df_features_gbm['ticker'] == ticker].sort_values('date')
    if ticker_gbm.empty:
        continue    
    
    gbm_cols = GBM_FEATURE_COLS
    
    # 오늘 밤 마감된 가장 마지막 행 추출 (데이터프레임 형태 유지를 위해 .iloc[[-1]])
    lgb_input = ticker_gbm[gbm_cols].iloc[[-1]]
    
    # LightGBM 상승 확률 스코어 추출 [0][1]
    prob_lgb = lgb_model.predict_proba(lgb_input)[0][1]

    # -----------------------------------------------
    # Multi-Input LSTM 실전 추론
    # -----------------------------------------------
    ticker_lstm = df_features_lstm[df_features_lstm['ticker'] == ticker].sort_values('date')
    # 최소 60영업일치 데이터가 확보되었는지 체크
    if len(ticker_lstm) < SEQ_LEN_60:
        continue
        
    lstm_cols = LSTM_FEATURE_COLS
    ticker_lstm_ordered = ticker_lstm[lstm_cols]
    # 최신 20영업일, 60영업일 피처 배열 분리 (.values)
    seq_20 = ticker_lstm_ordered.iloc[-SEQ_LEN_20:].values
    seq_60 = ticker_lstm_ordered.iloc[-SEQ_LEN_60:].values
    
    ticker_scaler = scalers[ticker] 
    
    seq_20_scaled = ticker_scaler.transform(pd.DataFrame(seq_20, columns=lstm_cols))
    seq_60_scaled = ticker_scaler.transform(pd.DataFrame(seq_60, columns=lstm_cols))
    
    # 텐서 차원 맞춤 (1, time_steps, features)
    lstm_input_20 = np.expand_dims(seq_20_scaled, axis=0)
    lstm_input_60 = np.expand_dims(seq_60_scaled, axis=0)

    # 텐서플로 데이터 타입으로 강제 캐스팅 (케라스 연산 보장)
    tensor_input_20 = tf.convert_to_tensor(lstm_input_20, dtype=tf.float32)
    tensor_input_60 = tf.convert_to_tensor(lstm_input_60, dtype=tf.float32)
    
    # Multi-Input LSTM 모델 추론연산 기동
    prob_lstm = lstm_model.predict(
        {
            'Input_20d': tensor_input_20, 
            'Input_60d': tensor_input_60  
        }, 
        verbose=0
    ).flatten()[0]

    # -----------------------------------------------
    # 앙상블 가중치 결합 및 데이터 적재 (7:3)
    # -----------------------------------------------
    final_prob = (prob_lgb * 0.7) + (prob_lstm * 0.3)
    
    results.append({
        'ticker': ticker,
        'prob_lgb': prob_lgb,
        'prob_lstm': prob_lstm,
        'final_prob': final_prob
    })

# 결과 데이터프레임 빌드
inference_df = pd.DataFrame(results, columns=['ticker', 'prob_lgb', 'prob_lstm', 'final_prob'])

# ===================================================
# 교집합 AND 가드레일 필터링 알고리즘 가동
# ===================================================

inference_df['signal'] = (inference_df['prob_lgb'] >= LGBM_THRESHOLD) & (inference_df['prob_lstm'] >= LSTM_THRESHOLD)

# 조건을 통과한 우량주 안에서만 합산 점수(final_prob) 기준으로 내림차순 정렬
valid_picks = inference_df[inference_df['signal'] == True].sort_values('final_prob', ascending=False)

# ===================================================
# 7. 최종 출력 결과창 
# ===================================================
print("\n=============================================")
print(f"[주먹 봇] {datetime.now().strftime('%Y-%m-%d')} 최종 실전 매수 시그널")
print("=============================================")
if not valid_picks.empty:
    # 정렬된 상위 최대 3개 종목만 압축 추출
    final_top3 = valid_picks.head(3)
    for idx, row in final_top3.iterrows():
        print(f"진입 확정: {row['ticker']} | 결합스코어: {row['final_prob']:.4f} [LGBM: {row['prob_lgb']:.4f} / LSTM: {row['prob_lstm']:.4f}]")
else:
    print("오늘 밤 가드레일 조건을 패스한 대장주가 없습니다. 현금보유 권장.")
print("=============================================\n")
print("DEBUG - 추론 완료된 종목 개수:", len(inference_df))
print(inference_df) 
print("[GBM 전처리 후 데이터 개수]:", len(df_features_gbm))
print("[LSTM 전처리 후 데이터 개수]:", len(df_features_lstm))

for _, row in inference_df.iterrows():
    lgb_ok = row['prob_lgb'] >= LGBM_THRESHOLD
    lstm_ok = row['prob_lstm'] >= LSTM_THRESHOLD
    if not (lgb_ok and lstm_ok):
        print(f"DEBUG: {row['ticker']} 탈락 이유 -> LGBM 통과: {lgb_ok}, LSTM 통과: {lstm_ok}")