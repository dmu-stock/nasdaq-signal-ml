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