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
import pandas as pd
from typing import List

# LLM 기반 헤드라인 전처리 함수
# (단건 호출이 아닌 배치 호출로 API 비용/시간 절감)
from headline_preprocessor import preprocess_headlines_batch


# -------------------------------------------------
# 반환 DataFrame 컬럼 정의
# (항상 동일한 구조를 유지하기 위함)
# -------------------------------------------------
NEWS_COLUMNS = [
    "date",           # 뉴스 발행 날짜
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
    "AAPL": "apple",
    "MSFT": "microsoft",
    "AMZN": "amazon",
    "GOOGL": "google",
    "META": "meta",
    "TSLA": "tesla",
    "NVDA": "nvidia",
    "V": "visa",
    "UNH": "unitedhealth",
    "PLTR" : "palantir",
    "IREN" : "iren"
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
    "SPY": "market",
    "QQQ": "nasdaq"
}


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

            headline = content.get("title", "")
            headline_lower = headline.lower()

            if (company_name.lower() not in headline_lower) and (ticker.lower() not in headline_lower):
                continue
            

            # 발행일 (ISO 문자열 → date)
            pub_date = content.get("pubDate", "")
            date = pd.to_datetime(pub_date, utc=True).date() if pub_date else None

            if date is None:
                continue

            source = content.get("provider", {}).get("displayName", "")

            rows.append({
                "date": date,
                "ticker": ticker,
                "company_name": company_name,
                "news_type": news_type,
                "headline": headline,
                # clean_headline은 전처리 단계에서 채운다
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
    combined = combined.drop_duplicates(subset=['clean_headline'])
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
            ["date", "ticker", "news_type", "headline", "clean_headline", "source"]
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
    df_news = fetch_all_news(save_debug_csv=True)

    print("\n[미리보기]")
    print(df_news.head(50))