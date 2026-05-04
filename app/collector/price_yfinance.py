"""
price_yfinance.py

[무엇을 하는 파일인가?]
- Yahoo Finance를 이용해 미국 우량주 10종목의 주가 데이터를 수집한다.
- 최근 1~2년 수준의 일봉(OHLCV) 데이터를 원천 데이터 형태로 제공한다.
- 기술지표(RSI, MA 등)는 여기서 계산하지 않고,
  이후 feature 설계 단계에서 계산할 수 있도록
  '깨끗한 가격 시계열 데이터'만 반환한다.

[설계 의도]
- 본 파일의 역할은 '가격 데이터 수집'에 한정한다.
- 지표 계산, 점수화, 모델 입력 가공은 다른 단계에서 수행한다.
- 예외 처리와 전처리를 통해 이후 파이프라인이 안정적으로 동작하도록 한다.
"""

import yfinance as yf
import pandas as pd
from typing import List, Optional
from app.database.sqlite_db import save_price_to_db


# -------------------------------------------------
# 수집 대상: 미국 우량주 10종목
# (Yahoo Finance 티커 기준)
# -------------------------------------------------
BLUE_CHIP_STOCKS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "AMZN",   # Amazon
    "GOOGL",  # Alphabet (Google)
    "META",   # Meta Platforms
    "TSLA",   # Tesla
    "NVDA",   # NVIDIA
    "BRK-B",  # Berkshire Hathaway (B주)
    "V",      # Visa
    "UNH",    # UnitedHealth Group
]

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


def fetch_price_data(
    ticker: str,
    period: str = "2y",
    interval: str = "1d"
) -> Optional[pd.DataFrame]:
    """
    단일 종목의 과거 주가 데이터를 수집하고 전처리하여 반환한다.

    처리 흐름:
    1. Yahoo Finance에서 주가 이력 데이터 요청
    2. Date 인덱스를 컬럼으로 변환
    3. 분석에 필요한 컬럼만 선택
    4. 결측치 제거 및 정렬

    Parameters
    ----------
    ticker : str
        종목 티커 심볼 (예: "AAPL")
    period : str
        조회 기간 (기본값: "2y")
    interval : str
        데이터 간격 (기본값: "1d", 일봉)

    Returns
    -------
    pd.DataFrame or None
        전처리된 OHLCV DataFrame.
        수집 실패 시 None 반환.

        컬럼:
        - Date
        - ticker
        - Open
        - High
        - Low
        - Close
        - Adj Close
        - Volume
    """

    try:
        # Yahoo Finance Ticker 객체 생성
        # stock = yf.Ticker(ticker)

        # 주가 이력 데이터 요청
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False, 
            progress=False
        )

        # 데이터가 없는 경우 조기 반환
        if df.empty:
            print(f"[경고] {ticker}: 수집된 데이터가 없습니다.")
            return None

        # -----------------------------
        # 데이터 전처리
        # -----------------------------

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Date 인덱스를 컬럼으로 변환
        df = df.reset_index()

        # 종목 식별용 ticker 컬럼 추가
        df["ticker"] = ticker

        # 타임존 제거 및 date 타입으로 변환
        # (CSV 저장, 병합 시 오류 방지 목적)
        df["Date"] = (
            pd.to_datetime(df["Date"])
            .dt.tz_localize(None)
            .dt.date
        )

        # 등락률 컬럼 추가
        df['change_rate'] = df.groupby('ticker')['Adj Close'].pct_change()

        df['ma5'] = df['Adj Close'].rolling(window=5).mean()
        df['ma20'] = df['Adj Close'].rolling(window=20).mean()

        df['rsi'] = calcurate_rsi(df, period=14)
        df = df.fillna(0)


        df = df.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume"
        })

        # 분석에 필요한 컬럼만 선택
        # (Dividends, Stock Splits 등은 제거)
        df = df[
            ["date", "ticker", "open", "high", "low", "close", "adj_close","volume","change_rate", "ma5", "ma20", "rsi"]
        ]

        # 결측치 제거
        df = df.dropna()

        # 날짜 기준 오름차순 정렬
        df = df.sort_values("date").reset_index(drop=True)
        
        return df

    except Exception as e:
        # 네트워크 오류, 잘못된 티커 등 모든 예외 처리
        print(f"[에러] {ticker} 데이터 수집 실패: {e}")
        return None


def fetch_all_stocks_price_data(
    tickers: List[str] = BLUE_CHIP_STOCKS,
    period: str = "2y"
) -> pd.DataFrame:
    """
    여러 종목의 주가 데이터를 일괄 수집하여 하나의 DataFrame으로 결합한다.

    Parameters
    ----------
    tickers : List[str]
        수집할 종목 티커 리스트
    period : str
        조회 기간 (기본값: "2y")

    Returns
    -------
    pd.DataFrame
        모든 종목 데이터를 수직 결합한 DataFrame.
        수집 실패한 종목은 자동으로 제외된다.
    """

    data_frames = []

    for ticker in tickers:
        print(f"[수집 중] {ticker} ...")
        df = fetch_price_data(ticker, period=period)

        # 수집 성공한 경우에만 추가
        if df is not None:
            data_frames.append(df)

    # 모든 종목 수집에 실패한 경우
    if not data_frames:
        print("[경고] 수집된 데이터가 없습니다. 티커 목록 또는 네트워크를 확인하세요.")
        return pd.DataFrame()

    # 모든 종목 데이터를 하나로 결합
    combined_df = pd.concat(data_frames, ignore_index=True)

    print(f"\n[완료] 총 {len(data_frames)}개 종목, {len(combined_df)}개 행 수집")
    return combined_df


def save_to_csv(df: pd.DataFrame, filename: str) -> None:
    """
    수집된 가격 데이터를 CSV 파일로 저장한다.

    Parameters
    ----------
    df : pd.DataFrame
        저장할 데이터
    filename : str
        저장할 파일 경로
    """

    if df.empty:
        print("[경고] 저장할 데이터가 없습니다.")
        return

    # index=False: DataFrame 인덱스는 저장하지 않음
    # utf-8-sig: Excel에서 한글 깨짐 방지
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"[CSV 저장 완료] {filename} ({len(df)}행)")


# -------------------------------------------------
# 단독 실행 테스트
# -------------------------------------------------
if __name__ == "__main__":
    # 전체 종목 가격 데이터 수집
    df_prices = fetch_all_stocks_price_data()
    if df_prices is not None and not df_prices.empty:
        # db 저장(학습 데이터용)
        save_price_to_db(df_prices)

        print("\n[미리보기]")
        print(df_prices.head(10))
        print(f"\n수집 기간: {df_prices['date'].min()} ~ {df_prices['date'].max()}")
        print(f"종목 수: {df_prices['ticker'].nunique()}")
        print(f"총 데이터 수: {len(df_prices)}")

        # CSV 저장 (검증/공유용)
        save_to_csv(df_prices, "bluechip_price_data.csv")
    else:
        print("수집된 데이터가 없습니다.")
        