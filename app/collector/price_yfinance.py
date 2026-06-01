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
from app.config.config import TICKERS


# -------------------------------------------------
# 수집 대상: nasdaq 지수
# -------------------------------------------------
tickers = TICKERS

def get_nasdaq_data(period: str = "4y") ->  Optional[pd.DataFrame]:
    nasdaq = yf.download(
        "^IXIC",
        period=period, 
        interval="1d",
        auto_adjust=False, 
        progress=False 
    ) 
    # 데이터가 없는 경우 조기 반환
    if nasdaq.empty:
        print("[경고] ^IXIC: 수집된 데이터가 없습니다.")
        return None

    if isinstance(nasdaq.columns, pd.MultiIndex):
        nasdaq.columns = nasdaq.columns.get_level_values(0)
    
    nasdaq = nasdaq.reset_index()

    nasdaq["Date"] = (
        pd.to_datetime(nasdaq["Date"])
        .dt.tz_localize(None)
        .dt.date
    )

    nasdaq['nasdaq_change_rate'] = nasdaq['Adj Close'].pct_change()

    nasdaq=nasdaq.rename(columns={
            'Date': 'date',
            'Adj Close': 'nasdaq_close'
            })
    
    # 날짜 기준 오름차순 정렬
    nasdaq = nasdaq.sort_values(["date"]).reset_index(drop=True)
    

    return nasdaq[['date', 'nasdaq_close', 'nasdaq_change_rate']]

# -------------------------------------------------
# 수집 대상: vix 지수, 10년물 미국 국채금리 수집
# -------------------------------------------------
def get_market_regime_data(period: str = "4y") ->  Optional[pd.DataFrame]:
    vix_raw = yf.download(
        "^VIX",
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    tnx_raw = yf.download(
        "^TNX",
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if vix_raw.empty or tnx_raw.empty:
        print("[경고] VIX 또는 TNX 데이터 없음")
        return None

    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = vix_raw.columns.get_level_values(0)
    if isinstance(tnx_raw.columns, pd.MultiIndex):
        tnx_raw.columns = tnx_raw.columns.get_level_values(0)

    # Close 컬럼만 추출
    vix = vix_raw[['Close']].copy()
    tnx = tnx_raw[['Close']].copy()

    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    tnx.index = pd.to_datetime(tnx.index).tz_localize(None)

    vix.columns = ['vix']
    tnx.columns = ['tnx']

    # 날짜 기준으로 합치기
    regime_df = pd.merge(
        vix.reset_index(),
        tnx.reset_index(),
        on='Date',
        how='outer'
    )

    regime_df = regime_df.rename(columns={'Date': 'date'})
    regime_df['date'] = pd.to_datetime(regime_df['date'])
    regime_df = regime_df.sort_values('date').reset_index(drop=True)

    # 빈 날짜 채우기
    regime_df['vix'] = regime_df['vix'].ffill()
    regime_df['tnx'] = regime_df['tnx'].ffill()

    print(f"[완료] VIX/TNX 수집: {len(regime_df)}행 "
          f"| {regime_df['date'].min()} ~ {regime_df['date'].max()}")

    return regime_df    



# -------------------------------------------------
# 수집 대상: 대상 tickers 4y stock row data
# 동작 방식: 주가 데이터 수집 후 나스닥 데이터와 결합, alpha 계산 후 추가
# -------------------------------------------------
def fetch_price_data(
    ticker: str,
    nasdaq_df,
    period: str = "4y",
    interval: str = "1d"
) -> Optional[pd.DataFrame]:
    
    try:
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
        df["date"] = (
            pd.to_datetime(df["Date"])
            .dt.tz_localize(None)
            .dt.date
        )

        # 등락률 컬럼 추가
        df['change_rate'] = df.groupby('ticker')['Adj Close'].pct_change()

        
        # 나스닥 데이터와 날짜 기준으로 병합 (Left Join)
        df = pd.merge(df, nasdaq_df, on='date', how='left')
        
        # 시장 변화율 - 종목 등락률
        # 값 > 0 → 시장보다 강함
        df['alpha'] = df['change_rate'] - df['nasdaq_change_rate']

        df = df.dropna()

        df = df.rename(columns={
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
            ["date", "ticker", "open", "high", "low", "close", "adj_close","volume","change_rate", "nasdaq_close", "nasdaq_change_rate","alpha"]
        ]

        # 결측치 제거
        df = df.dropna()

        # 날짜 기준 오름차순 정렬
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        
        return df

    except Exception as e:
        # 네트워크 오류, 잘못된 티커 등 모든 예외 처리
        print(f"[에러] {ticker} 데이터 수집 실패: {e}")
        return None


def fetch_all_stocks_price_data(
    tickers: List[str] = tickers,
    period: str = "4y"
) -> pd.DataFrame:

    data_frames = []
    nasdaq_df=get_nasdaq_data(period=period)
    regime_df = get_market_regime_data(period=period)

    for ticker in tickers:
        print(f"[수집 중] {ticker} ...")
        df = fetch_price_data(ticker,nasdaq_df,period=period)

        # 수집 성공한 경우에만 추가
        if df is not None:
            data_frames.append(df)

    # 모든 종목 수집에 실패한 경우
    if not data_frames:
        print("[경고] 수집된 데이터가 없습니다. 티커 목록 또는 네트워크를 확인하세요.")
        return pd.DataFrame()

    # 모든 종목 데이터를 하나로 결합
    combined_df = pd.concat(data_frames, ignore_index=True)

    # 날짜형식 통일
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    regime_df['date']   = pd.to_datetime(regime_df['date'])

    # regime 1번만 병합
    if regime_df is not None:
        combined_df = pd.merge(combined_df, regime_df, on='date', how='left')
        combined_df['vix'] = combined_df['vix'].ffill()
        combined_df['tnx'] = combined_df['tnx'].ffill()
    else:
        combined_df['vix'] = 0.0
        combined_df['tnx'] = 0.0

    print(f"\n[완료] 총 {len(data_frames)}개 종목, {len(combined_df)}개 행 수집")
    return combined_df


# -----------------------------
# 수집된 데이터 csv 저장
# -----------------------------
def save_to_csv(df: pd.DataFrame, filename: str) -> None:
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
    nasdaq_df=get_nasdaq_data()
    df_prices = fetch_all_stocks_price_data()
    if df_prices is not None and not df_prices.empty:
        # db 저장(학습 데이터용)
        save_price_to_db(df_prices)

        print("\n[미리보기]")
        print(df_prices.head(10))
        print(f"\n수집 기간: {df_prices['date'].min()} ~ {df_prices['date'].max()}")
        print(f"종목 수: {df_prices['ticker'].nunique()}")
        print(f"총 데이터 수: {len(df_prices)}")

        print(
            df_prices.loc[495:505,
            ['date', 'ticker', 'adj_close']]
        )
        print(
            df_prices.groupby(['ticker', 'date']).size()
            .sort_values(ascending=False)
            .head(20)
        )   

        # CSV 저장 (검증/공유용)
        save_to_csv(df_prices, "bluechip_price_data.csv")
    else:
        print("수집된 데이터가 없습니다.")
        