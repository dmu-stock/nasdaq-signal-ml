# TODO: 
# 1. yfinance 라이브러리를 활용하여 미국 우량주 10종목 리스트 정의
#    (AAPL, MSFT, AMZN, GOOGL, META, TSLA, NVDA, BRK-B, V, UNH 등)
#
# 2. 지정된 Ticker들에 대해 과거 주가 데이터(OHLCV) 수집 함수 구현
#    - 기간: 최근 1년 ~ 2년 (학습용 및 추론용)
#    - 간격: 1일 (Daily)
#
# 3. 데이터 수집 시 발생할 수 있는 에러 처리 (네트워크 오류, 잘못된 티커 등)
#
# 4. 수집된 데이터를 Pandas DataFrame으로 변환 및 기본 전처리
#    - 결측치 처리
#    - 날짜 형식 통일
#
# 5. (선택사항) 수집된 데이터를 CSV 또는 SQLite에 임시 저장하는 로직