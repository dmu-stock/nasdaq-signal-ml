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
from app.config.config import GBM_FEATURE_COLS

class FeatureProcessorGBM:
    def __init__(self, db_path: str = "stock_data.db"):
        self.db_path = db_path

    def get_raw_data(self)->pd.DataFrame:
        conn = get_connection()
        query = f"SELECT * FROM stock_prices ORDER BY date ASC"
        df = pd.read_sql(query,conn)
        conn.close()

        return df

    def calc_technical_indicators(self, df, rsi_period=14,is_inference=False):
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

        # 최고가 대비 하락률 (High Drawdown)
        df['max_20'] = df.groupby('ticker')['high'].transform(lambda x: x.rolling(20).max())
        df['drawdown_20'] = (df['adj_close'] - df['max_20']) / df['max_20']

        # 52주 고가 대비 현재가 위치 (0=52주 저점, 1=52주 고점)
        df['price_position_52w'] = (
            (df['adj_close'] - df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(252).min())) /
            (df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(252).max()) - 
            df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(252).min()) + 1e-9)
        )

        # 현재가가 20일 밴드 어디쯤인지 (이미 bb_percent 있으므로 60일 버전 추가)
        std_60 = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(60).std())
        ma_60  = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(60).mean())
        df['bb_percent_60'] = (df['adj_close'] - (ma_60 - 2*std_60)) / (4*std_60 + 1e-9)

        # 나스닥 5일 누적 수익률
        df['nasdaq_5d'] = (
            df.groupby('date')['nasdaq_change_rate']
            .transform(lambda x: (1 + x).rolling(5).apply(np.prod, raw=True) - 1)
        )


        if not is_inference:
            # 미래 수익률
            # df['target_1'] = (
            #     df.groupby('ticker')['adj_close']
            #     .shift(-1) / df['adj_close'] - 1
            # )

            # df['target_5'] = (
            #     df.groupby('ticker')['adj_close']
            #     .shift(-5) / df['adj_close'] - 1
            # )

            # # 1. 내일 종가(t1), 모레 종가(t2), 글피 종가(t3)를 미래에서 정확히 당겨오기
            # df['close_t1'] = df.groupby('ticker')['adj_close'].shift(-1)
            # df['close_t2'] = df.groupby('ticker')['adj_close'].shift(-2)
            # df['close_t3'] = df.groupby('ticker')['adj_close'].shift(-3)

            # # 2. 오늘 종가(adj_close) 대비 미래 각 영업일 종가의 수익률 계산
            # df['return_t1'] = (df['close_t1'] - df['adj_close']) / df['adj_close']
            # df['return_t2'] = (df['close_t2'] - df['adj_close']) / df['adj_close']
            # df['return_t3'] = (df['close_t3'] - df['adj_close']) / df['adj_close']

            # # 3. 미래 3일의 '종가 기준 최고 수익률'만 쏙 뽑아내기
            # df['max_return_3d'] = df[['return_t1', 'return_t2', 'return_t3']].max(axis=1)

            # # 4. 라벨링: 3일 중 한 번이라도 종가 기준으로 +2.5% 이상 올랐으면 1, 아니면 0
            # df['label'] = np.where(df['max_return_3d'] >= 0.025, 1, 0)

            df['high_20'] = df.groupby('ticker')['high'].transform(lambda x: x.rolling(20).max())

            # 5일 후 수익률
            df['forward_5d'] = (
                df.groupby('ticker')['adj_close'].shift(-5) / df['adj_close'] - 1
            )
            
            # 나스닥 5일 누적 수익률 (시장 베타 제거)
            df['nasdaq_forward_5d'] = df.groupby('ticker')['nasdaq_change_rate'].transform(
                lambda x: (1 + x).rolling(5).apply(np.prod, raw=True) - 1
            ).shift(-5)  # 미래 5일
            
            # 초과수익 = 종목 수익률 - 나스닥 수익률
            df['excess_5d'] = df['forward_5d'] - df['nasdaq_forward_5d']
            
            # 눌림목 z-score (종목 자체 기준)
            df['pullback'] = (df['adj_close'] / df['high_20']) - 1
            df['pullback_zscore'] = (
                df['pullback'] - df.groupby('ticker')['pullback']
                .transform(lambda x: x.rolling(60).mean())
            ) / (df.groupby('ticker')['pullback']
                .transform(lambda x: x.rolling(60).std()) + 1e-9)
            
            # 라벨: 평소보다 더 눌렸고 + 나스닥보다 더 올랐으면 1
            df['label'] = np.where(
                (df['pullback_zscore'] <= -0.5) &   # 평소 대비 더 눌린 상태
                (df['excess_5d'] >= 0.01),           # 나스닥 대비 +2% 초과수익
                1, 0
            )
            
            print("양성 비율:", df['label'].mean())
        else:
            # 라벨이 없으므로 -1로 초기화
            df['label'] = -1
            
            # 필요한 컬럼만 체크
            check_cols_inf = ['disparity_20', 'alpha_20', 'drawdown_20', 'rsi', 'macd_hist']
            # 결측치 제거
            df = df.dropna(subset=check_cols_inf).reset_index(drop=True)

        
        

        meta_cols = ['ticker', 'date']
        feature_cols = GBM_FEATURE_COLS
        if not is_inference:
            df = df[meta_cols + feature_cols + ['label']]
        else:
            df = df[meta_cols + feature_cols]

        print(df.columns)

        return df

if __name__ == "__main__":
    processor = FeatureProcessorGBM()
    df = processor.get_raw_data()
    df = processor.calc_technical_indicators(df)

    today = datetime.now().strftime("%Y%m%d")
    df.to_csv(
        f"feature__indicator_{today}.csv",
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
