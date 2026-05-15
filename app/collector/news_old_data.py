import pandas as pd
from datetime import timedelta
from datetime import time, datetime, timedelta

# 1. 파일 읽기 (기존 코드)
df = pd.read_parquet('train-00000-of-00001.parquet', columns=['date', 'title', 'content'])

def get_predictive_session(pub_time):
    """
    뉴스가 발생한 시간을 기준으로, 
    어떤 본장 예측에 기여할지 결정합니다.
    """
    # 1. KST -> UTC로 변환 (미국 주식 시장은 UTC 기준이 편함)
    # 2. 뉴스가 '전날 05:00 ~ 오늘 05:00' 사이인지 판별 등 로직 적용
    
    # 더 직관적인 방법:
    # 한국 시간 05:00 ~ 22:30 사이에 발생한 뉴스는
    # '오늘 밤 22:30에 시작되는 본장'의 예측에 반영합니다.
    
    # 05:00 ~ 22:30 사이인가?
    if time(5, 0) <= pub_time.time() < time(22, 30):
        return "PREDICT_TONIGHT" # 오늘 밤 본장 예측용
    else:
        return "PREDICT_TOMORROW" # 내일 밤 본장 예측용

# 2. 시간대 변환 및 보정 로직 추가
def align_news_time(df):
    # 날짜 컬럼을 datetime 객체로 변환 (UTC 기준이라고 가정)
    df['pub_time'] = pd.to_datetime(df['date']).dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
    
    # target_date 계산을 위한 함수 정의
    def calculate_target(row):
        pub_time = row['pub_time']
        
        # 지호님의 기존 세션 분류 로직 호출 (함수가 정의되어 있어야 함)
        # 만약 함수가 다른 파일에 있다면 import 하거나 여기에 정의하세요.
        session = get_predictive_session(pub_time) 
        
        target_date = pub_time.date()
        
        if session == "PREDICT_TOMORROW" and pub_time.hour >= 22:
            target_date = target_date + timedelta(days=1)
            
        return pd.Series([target_date, session])

    # 새로운 컬럼 생성
    df[['target_date', 'market_session']] = df.apply(calculate_target, axis=1)
    return df

# 로직 적용
df = align_news_time(df)
df['date'].unique()
# 결과 확인
print("\n--- 보정된 데이터 확인 ---")
print(df[['pub_time', 'target_date', 'market_session']].head())
print(df.info())
df.to_csv("historical_news_data.csv", index=False, encoding="utf-8-sig")
print("완료: historical_news_data.csv 저장됨")