"""
news_crawler.py

[무엇을 하는 파일인가?]
- Yahoo Finance에서 뉴스 '헤드라인'만 수집한다.
- 뉴스 수집 범위를
  1) 개별 종목
  2) 섹터(ETF)
  3) 시장/지수(ETF)
  로 확장하여 데이터 부족 문제를 보완한다.
- 수집한 헤드라인을 LLM으로 전처리하여
  FinBERT 감정 분석에 바로 사용할 수 있는 형태로 반환한다.

[왜 이렇게 설계했는가?]
- 뉴스 본문은 의견·배경 설명이 많아 감정 분석 노이즈가 크다.
- 시장 반응이 가장 압축된 정보는 '헤드라인'이므로 헤드라인만 사용한다.
- 개별 종목 뉴스만으로는 표본이 적을 수 있어,
  섹터/시장 뉴스로 맥락 정보를 함께 제공한다.
"""

import yfinance as yf
import finnhub
import pandas as pd
from datetime import time, datetime, timedelta
import time as time_module
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import os

# LLM 기반 헤드라인 전처리 함수
# (단건 호출이 아닌 배치 호출로 API 비용/시간 절감)
load_dotenv()
from app.collector.headline_preprocessor import preprocess_headlines_batch

# Finnhub 클라이언트 초기화
API_KEY = os.getenv("FINNHUB_API_KEY")
finnhub_client = finnhub.Client(api_key=API_KEY)
# -------------------------------------------------
# 반환 DataFrame 컬럼 정의
# (항상 동일한 구조를 유지하기 위함)
# -------------------------------------------------
NEWS_COLUMNS = [
    "date",           # 뉴스 발행 날짜
    "pub_datetime",     
    "session",         
    "ticker",         # 티커 (종목 / 섹터 ETF / 시장 ETF)
    "company_name",
    "news_type",      # stock / sector / market
    "headline",       # 원본 뉴스 헤드라인
    "clean_headline", # LLM으로 정제된 헤드라인 (FinBERT 입력용)
    "source",         # 뉴스 출처 (Reuters, Bloomberg 등)
]


# -------------------------------------------------
# 뉴스 수집 대상 정의
# -------------------------------------------------

# 1) 개별 종목 뉴스
STOCK_TICKERS = {

    # Big Tech
    "AAPL": "apple",
    "MSFT": "microsoft",
    "AMZN": "amazon",
    "GOOGL": "google",
    "META": "meta",
    "NVDA": "nvidia",
    "TSLA": "tesla",

    # Semiconductor
    "AMD": "amd",
    "AVGO": "broadcom",
    "QCOM": "qualcomm",
    "INTC": "intel",
    "MU": "micron",

    # Finance
    "JPM": "jpmorgan",
    "BAC": "bank of america",
    "GS": "goldman sachs",
    "MS": "morgan stanley",
    "V": "visa",
    "MA": "mastercard",

    # Healthcare
    "UNH": "unitedhealth",
    "JNJ": "johnson and johnson",
    "PFE": "pfizer",
    "LLY": "eli lilly",
    "MRK": "merck",

    # Consumer
    "WMT": "walmart",
    "COST": "costco",
    "KO": "coca cola",
    "PEP": "pepsi",
    "MCD": "mcdonalds",
    "NKE": "nike",

    # Industrial
    "CAT": "caterpillar",
    "GE": "general electric",
    "HON": "honeywell",
    "BA": "boeing",

    # Energy
    "XOM": "exxon mobil",
    "CVX": "chevron",

    # Communication / Entertainment
    "NFLX": "netflix",
    "DIS": "disney",

}

# 2) 섹터 뉴스 (ETF 기준)
# → 산업 전반 분위기 파악 목적
SECTOR_TICKERS = {
    "XLK": "technology",
    "SOXX": "semiconductor",
    "XLF": "financial"
}

# 3) 시장 / 증시 뉴스 (지수 ETF)
# → 거시적인 시장 분위기 반영 목적
MARKET_TICKERS = {
    "SPY": "us_market",
    "QQQ": "tech_market"
}
def fetch_finnhub_news(ticker, start_date, end_date):
    """
    Finnhub을 이용해 과거 뉴스 데이터를 수집
    start_date, end_date: 'YYYY-MM-DD' 형식
    """
    # 1. 날짜 변환
    s_date = datetime.strptime(start_date, '%Y-%m-%d')
    e_date = datetime.strptime(end_date, '%Y-%m-%d')

    # 3개월씩 쪼개기
    date_ranges = pd.date_range(start=s_date, end=e_date, freq='3MS')
    all_news = []
    
    for i in range(len(date_ranges)-1):
        s = date_ranges[i].strftime('%Y-%m-%d')
        e = date_ranges[i+1].strftime('%Y-%m-%d')
        
        try:
            # Finnhub 호출
            news = finnhub_client.company_news(ticker, _from=s, to=e)
            if not news: continue
            
            for item in news:
                all_news.append({
                    "date": pd.to_datetime(item['datetime'], unit='s').date(),
                    "ticker": ticker,
                    "headline": item['headline'],
                    "source": item['source']
                })
            # API 호출 제한 방지 (무료 플랜 기준 1초당 호출 제한)
            time_module.sleep(1) 
        except Exception as e:
            print(f"[에러] {ticker} 수집 실패: {e}")
            
    rows = []
    for item in news:
        # 1. Unix Timestamp -> KST 변환
        pub_time = pd.to_datetime(item['datetime'], unit='s', utc=True).tz_convert('Asia/Seoul')
        
        # 2. 기존에 잘 짜두었던 세션 분류 로직 적용
        session = get_predictive_session(pub_time) # 지호님 기존 함수 호출
        
        # 3. 매칭용 target_date 계산 (22시 이후면 다음 날로 보정)
        target_date = pub_time.date()
        if session == "PREDICT_TOMORROW" and pub_time.hour >= 22:
            target_date = pub_time.date() + timedelta(days=1)
            
        rows.append({
            "date": target_date,          # 매칭을 위한 기준 날짜
            "pub_datetime": pub_time,     # 실제 발행 시각
            "session": session,           # PREDICT_TONIGHT or TOMORROW
            "ticker": ticker,
            "headline": item['headline'],
            "source": item['source']
        })
    return pd.DataFrame(rows)

def run_collection(tickers):
    """
    모든 티커에 대해 3년 치 뉴스 수집 및 저장
    """
    start_date = "2023-05-13" # 3년 전
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    total_data = []
    for ticker in tickers:
        print(f"[진행] {ticker} 데이터 수집 중...")
        df = fetch_finnhub_news(ticker, start_date, end_date)
        total_data.append(df)
        
    final_df = pd.concat(total_data, ignore_index=True)
    final_df.to_csv("historical_news_data.csv", index=False, encoding="utf-8-sig")
    print("완료: historical_news_data.csv 저장됨")

############################################################################################
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

def fetch_news_by_ticker(ticker: str, news_type: str) -> pd.DataFrame:
    """
    단일 티커(종목 / 섹터 / 시장)에 대한 뉴스 헤드라인을 수집한다.

    처리 흐름:
    1. Yahoo Finance에서 뉴스 목록 수집
    2. 헤드라인, 날짜, 출처만 추출
    3. 헤드라인을 LLM으로 전처리
    4. DataFrame 형태로 반환

    Parameters
    ----------
    ticker : str
        Yahoo Finance 티커 (예: AAPL, XLK, SPY)
    news_type : str
        뉴스 구분값 (stock / sector / market)

    Returns
    -------
    pd.DataFrame
        NEWS_COLUMNS 구조를 갖는 DataFrame
    """

    # -----------------------------
    # Step 1. Yahoo Finance 뉴스 수집
    # -----------------------------
    try:
        stock = yf.Ticker(ticker)
        # 뉴스가 없을 경우 None을 반환할 수 있으므로 방어적 처리
        news_items = stock.news or []
    except Exception as e:
        print(f"[에러] {ticker} 뉴스 수집 실패: {e}")
        return pd.DataFrame(columns=NEWS_COLUMNS)

    if not news_items:
        return pd.DataFrame(columns=NEWS_COLUMNS)

    rows = []

    # 종목 이름 저장
    if news_type == "stock":
        company_name = STOCK_TICKERS.get(ticker, "")
    elif news_type == "sector":
        company_name = SECTOR_TICKERS.get(ticker, "")
    else:
        company_name = MARKET_TICKERS.get(ticker, "")

    # -----------------------------
    # Step 2. 뉴스 메타데이터 파싱
    # -----------------------------
    for item in news_items:
        try:
            # yfinance 최신 뉴스 API 구조 기준
            content = item.get("content", {})

            headline = str(content.get("title", "")).replace("\n", " ").replace("\r", " ").strip()

            ## None 방지
            headline = str(headline)

            # 양쪽 공백 제거
            headline = headline.strip()

            # 너무 짧으면 제외
            if len(headline) < 10:
                continue

            #소문자 비교
            headline_lower = headline.lower()

            #company_name과 ticker가 없으면 continue
            # if (company_name.lower() not in headline_lower) and (ticker.lower() not in headline_lower):
            #     continue
            

            # 발행일 (ISO 문자열 → datetime)
            pub_date = content.get("pubDate", "")
            pub_time = pd.to_datetime(pub_date, utc=True).tz_convert('Asia/Seoul')

            if pd.isna(pub_time):
                continue

            session = get_predictive_session(pub_time)
            target_date = pub_time.date()

            if session == "PREDICT_TOMORROW" and pub_time.hour >= 22:
                # 22시 이후 뉴스면 다음 날 수익률 예측 데이터로 매칭
                target_date = pub_time.date() + timedelta(days=1)

            source = content.get("provider", {}).get("displayName", "")

            rows.append({
                "date": target_date,         # 매칭용 날짜
                "pub_datetime": pub_time,    # 정확한 발생 시각 (디버깅/필터링용)
                "session": session,          # 세션 정보
                "ticker": ticker,
                "company_name": company_name,
                "news_type": news_type,
                "headline": headline,
                "clean_headline": "",
                "source": source,
            })

        except Exception:
            # 개별 뉴스 파싱 실패 시 해당 건만 스킵
            continue

    if not rows:
        return pd.DataFrame(columns=NEWS_COLUMNS)

    # -----------------------------
    # Step 3. 헤드라인 배치 전처리 (LLM)
    # -----------------------------
    # 감정 방향은 유지하고, 분석에 방해되는 표현만 제거
    raw_headlines = [r["headline"] for r in rows]
    cleaned_headlines = preprocess_headlines_batch(raw_headlines)

    for r, c in zip(rows, cleaned_headlines):

        c = str(c)
        # 짧으면 원본 사용
        if len(c)<10:
            c=r['headline']

        r["clean_headline"] = c

    if news_type == "stock":
        filtered = []
        for r in rows:
            text = r["clean_headline"].lower()
            ticker_lower = ticker.lower()
            name_lower = company_name.lower()

            if ticker_lower in text or name_lower in text:
                filtered.append(r)

        rows = filtered

    return pd.DataFrame(rows, columns=NEWS_COLUMNS)




def fetch_all_news(save_debug_csv: bool = False) -> pd.DataFrame:
    """
    종목 / 섹터 / 시장 뉴스를 모두 수집하여 하나의 DataFrame으로 결합한다.

    Parameters
    ----------
    save_debug_csv : bool
        True일 경우 원본/전처리 비교용 CSV 저장
        (전처리 검증 및 팀 공유용)
    """

    all_news = []

    # -----------------------------
    # 종목 뉴스 수집
    # -----------------------------
    print("\n[종목 뉴스 수집]")
    for ticker,name in STOCK_TICKERS.items():
        df = fetch_news_by_ticker(ticker, "stock")

        if not df.empty:
            df["company_name"] = name
            all_news.append(df)

    # -----------------------------
    # 섹터 뉴스 수집
    # -----------------------------
    print("\n[섹터 뉴스 수집]")
    for ticker in SECTOR_TICKERS:
        df = fetch_news_by_ticker(ticker, "sector")
        if not df.empty:
            all_news.append(df)

    # -----------------------------
    # 시장 / 증시 뉴스 수집
    # -----------------------------
    print("\n[시장 뉴스 수집]")
    for ticker in MARKET_TICKERS:
        df = fetch_news_by_ticker(ticker, "market")
        if not df.empty:
            all_news.append(df)

    if not all_news:
        return pd.DataFrame(columns=NEWS_COLUMNS)

    # 모든 뉴스 결합 후 최신순 정렬
    combined = pd.concat(all_news, ignore_index=True)
    # 중복 제거
    combined = combined.drop_duplicates(subset=['ticker', 'clean_headline'])
    combined = combined.sort_values("date", ascending=False).reset_index(drop=True)

    print(
        f"\n[완료] 총 뉴스 {len(combined)}건 "
        f"(종목 {combined[combined.news_type == 'stock'].shape[0]}, "
        f"섹터 {combined[combined.news_type == 'sector'].shape[0]}, "
        f"시장 {combined[combined.news_type == 'market'].shape[0]})"
    )

    # -----------------------------
    # 전처리 검증 및 리뷰용 CSV 저장
    # -----------------------------
    if save_debug_csv:
        debug_df = combined[
            ["date","ticker","pub_datetime","session", "news_type", "headline", "clean_headline", "source"]
        ]
        debug_df.to_csv(
            "news_headline_preprocess_debug.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print("[저장] news_headline_preprocess_debug.csv")

    return combined


# -------------------------------------------------
# 단독 실행 테스트
# -------------------------------------------------
if __name__ == "__main__":
    # df_news = fetch_all_news(save_debug_csv=True)

    # print("\n[미리보기]")
    # print(df_news.head(50))
    target_tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "AMD"] # 테스트용 일부
    run_collection(target_tickers)