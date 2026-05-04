# TODO:
# 1. 데이터 정제 (Data Cleaning)
#    - yfinance에서 가져온 데이터 중 비어있는 값(NaN) 처리
#    - 주식 분할이나 배당 등이 반영된 수정 종가(Adj Close) 사용 여부 결정
#
# 2. 기술적 지표 생성 (Technical Indicators)
#    - 이동평균선(SMA/EMA): 5일, 20일, 60일선 등 계산
#    - 변동성 지표: 볼린저 밴드(Bollinger Bands), ATR 등
#    - 모멘텀 지표: RSI, MACD, Stochastic 등
#
# 3. 모델 입력용 데이터셋 구성 (Feature Matrix)
#    - XGBoost 모델이 학습할 때 사용했던 컬럼 순서와 동일하게 정렬
#    - 예측 시점(t)을 기준으로 과거 n일간의 데이터를 한 줄로 펼치기(Lag features)
#
# 4. 정규화 및 스케일링 (Optional)
#    - 데이터의 범위를 0~1 사이로 맞추는 등의 스케일링 작업 (필요 시)
#
# 5. 최종 데이터 유효성 검사
#    - 모델에 넣기 직전 데이터에 이상치나 무한대(Inf) 값이 없는지 확인

import yfinance as yf
import pandas as pd
from typing import List, Optional

#RSI 계산 함수
def calcurate_rsi(df, period=14):
    delta = df['Adj Close'].diff()

    # 2. 상승분(U)과 하락분(D) 분리
    up = delta.copy()
    down = delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    avg_gain = up.rolling(window=period).mean()
    avg_loss = down.abs().rolling(window=period).mean()

    # 4. RS(상대강도) 및 RSI 계산
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

# 등락률 컬럼 추가
        

        # df['ma5'] = df['Adj Close'].rolling(window=5).mean()
        # df['ma20'] = df['Adj Close'].rolling(window=20).mean()

        # df['rsi'] = calcurate_rsi(df, period=14)
       