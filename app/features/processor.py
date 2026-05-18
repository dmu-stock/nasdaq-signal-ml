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

import pandas as pd
import numpy as np
from app.database.sqlite_db import get_connection
from datetime import datetime

class FeatureProcessor:
    def __init__(self, db_path: str = "stock_data.db"):
        self.db_path = db_path

    def get_raw_data(self)->pd.DataFrame:
        conn = get_connection()
        query = f"SELECT * FROM stock_prices ORDER BY date ASC"
        df = pd.read_sql(query,conn)
        conn.close()

        return df
        
    def calc_technical_indicators(self, df, rsi_period=14):
        df = df.sort_values(['ticker','date']).reset_index(drop=True)

        # ---------------------------
        # 이동평균 (Trend)
        # ---------------------------
        df['ma5'] = (
        df.groupby('ticker')['adj_close']
        .transform(lambda x: x.rolling(5).mean())
        )

        df['ma20'] = (
            df.groupby('ticker')['adj_close']
            .transform(lambda x: x.rolling(20).mean())
        )

        df['ma_ratio'] = df['ma5'] / df['ma20']
        df['price_ma20'] = df['adj_close'] / df['ma20']

        df['disparity_20'] = (df['adj_close'] / df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(20).mean())) * 100
        df['disparity_20'] = df['disparity_20'].fillna(100)

        # MACD (이동평균 수렴 확산 지수) : 단기 이평선과 장기 이평선이 얼마나 빨리 멀어지는지(에너지)를 측정
        short_ema = df.groupby('ticker')['adj_close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
        long_ema = df.groupby('ticker')['adj_close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
        df['macd'] = short_ema - long_ema
        df['macd_signal'] = df.groupby('ticker')['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
        df['macd_hist'] = df['macd'] - df['macd_signal'] # 이 히스토그램이 중요!

        # 5일간의 고가 - 저가 평균 (종목의 활동성)
        df['price_range'] = (df['high'] - df['low']) / df['adj_close']
        df['tr_5'] = df.groupby('ticker')['price_range'].transform(lambda x: x.rolling(5).mean())


        # ---------------------------
        # RSI (Momentum)
        # ---------------------------
        delta = df.groupby('ticker')['adj_close'].diff()
        # 상승, 하락분 분리
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = (
            gain.groupby(df['ticker'])
            .transform(lambda x : x.rolling(rsi_period).mean())
        )
        
        avg_loss = (
            loss.groupby(df['ticker'])
            .transform(lambda x : x.rolling(rsi_period).mean())
        )

        # RS(상대강도) 및 RSI 계산
        rs = avg_gain / (avg_loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))

        #이격도
        # df['disparity_20'] = (df['adj_close'] - df['ma20']) / df['ma20']

        # 볼린저 밴드 %B
        std = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(20).std())
        ma20 = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(20).mean())
        df['upper_band'] = ma20 + (std * 2)
        df['lower_band'] = ma20 - (std * 2)
        # 현재가가 밴드 내 어디 위치하는지 (0~1 사이 값)
        df['bb_percent'] = (df['adj_close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])

        # ---------------------------
        # 3. 수익률 (Return)
        # ---------------------------
        if 'change_rate' in df.columns:
            df['return_1'] = df['change_rate']
            #5일 누적 수익률
            df['return_5'] = (
                df.groupby('ticker')['change_rate']
                .transform(lambda x: (1 + x).rolling(5).apply(np.prod, raw=True) - 1)
            )

            df['volatility_5'] = (
                df.groupby('ticker')['return_1']
                .transform(lambda x: x.rolling(5).std())
            )

        # ---------------------------
        # 거래량
        # ---------------------------
        if 'volume' in df.columns:
            df['volume_ma5'] = (
                df.groupby('ticker')['volume']
                .transform(lambda x: x.rolling(5).mean())
            )
            df['volume_ratio'] = df['volume'] / (df['volume_ma5'] + 1e-9)

        # ---------------------------
        # 시장 대비 (Alpha)
        # ---------------------------
        if 'alpha' in df.columns:
            df['alpha_5'] = (
                df.groupby('ticker')['alpha']
                .transform(lambda x: x.rolling(5).mean())
            )
            df['alpha_20'] = (
                df.groupby('ticker')['alpha']
                .transform(lambda x: x.rolling(20).mean())
            )
            df['alpha_divergence'] = df['alpha'] - df['alpha_5']
            df['alpha_5'] = df['alpha_5'].fillna(0)
            df['alpha_20'] = df['alpha_20'].fillna(0)
            df['alpha_divergence'] = df['alpha_divergence'].fillna(0)

        #심리도
        df['is_up'] = (df['change_rate'] > 0).astype(int)
        df['psychological'] = df.groupby('ticker')['is_up'].transform(lambda x: x.rolling(10).mean()) * 100

        # 미래 수익률
        df['target_1'] = (
            df.groupby('ticker')['adj_close']
            .shift(-1) / df['adj_close'] - 1
        )

        df['target_5'] = (
            df.groupby('ticker')['adj_close']
            .shift(-5) / df['adj_close'] - 1
        )
        # 분류 (노이즈 제거)
        # df['label'] = (df['target_5'] > 0.03).astype(int)
        # print(df['label'].value_counts(normalize=True))

        # 내일부터 3일 이내에 '종가 기준'으로 한 번이라도 2.5% 이상 상승하면 1, 아니면 0
        df['future_max_close_3d'] = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(3, min_periods=1).max().shift(-3))
        df['label'] = np.where((df['future_max_close_3d'] - df['adj_close']) / df['adj_close'] >= 0.025, 1, 0)

        # 최고가 대비 하락률 (High Drawdown)
        df['max_20'] = df.groupby('ticker')['high'].transform(lambda x: x.rolling(20).max())
        df['drawdown_20'] = (df['adj_close'] - df['max_20']) / df['max_20']
        
        check_cols = ['disparity_20', 'alpha_20', 'drawdown_20', 'future_max_close_3d']
        df = df.dropna(subset=check_cols).reset_index(drop=True)

        meta_cols = ['ticker', 'date']
        feature_cols = [
            # 모멘텀
            'change_rate',
            'return_1',
            'return_5',
            #이격도
            'disparity_20',

            # 시장 상대 강도
            'alpha',
            'alpha_5',
            'alpha_20',
            'alpha_divergence',

            # 이동평균
            'ma_ratio',
            'price_ma20',

            # RSI / 변동성
            'rsi',
            'volatility_5',

            #볼린저
            'bb_percent',

            #심리도
            'psychological',

            #macd
            'macd_hist',

            # 거래량
            'volume_ratio',

            #최고가 대비 하락률
            'drawdown_20',

            # 시장
            'nasdaq_change_rate',

            # 타겟
            'label',
            # 5일간의 고가 - 저가 평균 (종목의 활동성)
            'tr_5'
        ]
        df = df[meta_cols + feature_cols]

        print(df.columns)

        return df
    
if __name__ == "__main__":
    processor = FeatureProcessor()
    df = processor.get_raw_data()
    df = processor.calc_technical_indicators(df)

    today = datetime.now().strftime("%Y%m%d")
    df.to_csv(
        f"feature__indicator{today}.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print(df.shape)
    print(
    df[['ticker', 'date']].duplicated().sum()
    )
    print(
        df[['ticker', 'date']]
        .value_counts()
        .head(20)
    )
    print(df.head())
