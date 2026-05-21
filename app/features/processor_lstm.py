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
from app.config.config import LSTM_FEATURE_COLS

class FeatureProcessorLSTM:
    def __init__(self, db_path: str = "stock_data.db"):
        self.db_path = db_path

    def get_raw_data(self)->pd.DataFrame:
        conn = get_connection()
        query = f"SELECT * FROM stock_prices ORDER BY date ASC"
        df = pd.read_sql(query,conn)
        conn.close()

        return df

    def calc_technical_indicators(self, df, rsi_period=14, is_inference=False):
        # 종목과 날짜 순으로 철저하게 정렬
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

        # ---------------------------
        # 1. 이동평균 (Trend)
        # ---------------------------
        df['ma10'] = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(10).mean())
        df['ma20'] = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(20).mean())

        df['close_ratio_10'] = df['adj_close'] / (df['ma10'] + 1e-9)
        df['close_ratio_20'] = df['adj_close'] / (df['ma20'] + 1e-9)

        # 변동성 활동성 지표
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                (df['high'] - df.groupby('ticker')['adj_close'].shift(1)).abs(),
                (df['low'] - df.groupby('ticker')['adj_close'].shift(1)).abs()
            )
        )

        df['atr_5'] = df.groupby('ticker')['tr'].transform(lambda x: x.rolling(5).mean())
        df['atr_change'] = df.groupby('ticker')['atr_5'].pct_change(fill_method=None)
        df['atr_change'] = df['atr_change'].replace([np.inf, -np.inf], np.nan)

        # ---------------------------
        # 2. 수익률 및 변동성 (Return & Vol)
        # ---------------------------
        if 'change_rate' in df.columns:
            df['return_1'] = df['change_rate']
            df['return_3'] = df.groupby('ticker')['change_rate'].transform(
                lambda x: (1 + x).rolling(3).apply(np.prod, raw=True) - 1
            )
            df['return_5'] = df.groupby('ticker')['change_rate'].transform(
                lambda x: (1 + x).rolling(5).apply(np.prod, raw=True) - 1
            )
            # 20일선 중기 수익률 추가
            df['return_20'] = df.groupby('ticker')['change_rate'].transform(
                lambda x: (1 + x).rolling(20).apply(np.prod, raw=True) - 1
            )

            df['volatility_5'] = df.groupby('ticker')['return_1'].transform(
                lambda x: x.rolling(5).std()
            )
            # 단/중기 변동성 상태 (20일선)
            df['volatility_regime_20'] = (
                df['volatility_5'] / (
                    df.groupby('ticker')['volatility_5'].transform(lambda x: x.rolling(20).mean()) + 1e-9)
            )
            # 장기 변동성 상태 (60일선 대비 현재 변동성 압축률)
            df['volatility_regime_60'] = (
                df['volatility_5'] / (
                    df.groupby('ticker')['volatility_5'].transform(lambda x: x.rolling(60).mean()) + 1e-9)
            )

            df['volatility_compression'] = df['volatility_5'] / (df.groupby('ticker')['volatility_5'].transform(lambda x: x.rolling(20).mean()) + 1e-9)
            
            df['volatility_change'] = np.log((df['volatility_5'] + 1e-9) / (df.groupby('ticker')['volatility_5'].shift(3) + 1e-9))


        # ---------------------------
        # 3. 거래량 (Volume)
        # ---------------------------
        if 'volume' in df.columns:
            df['volume_ma5'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(5).mean())
            df['volume_ratio'] = df['volume'] / (df['volume_ma5'] + 1e-9)
            df['volume_change'] = df.groupby('ticker')['volume'].pct_change(fill_method=None)
            # 기존 10일선 수급 지표
            df['volume_z'] = (
                df['volume'] - df.groupby('ticker')['volume'].transform(lambda x: x.rolling(10).mean())
            ) / (df.groupby('ticker')['volume'].transform(lambda x: x.rolling(10).std()) + 1e-9)

            df['volume_shock'] = df.groupby('ticker')['volume_z'].transform(lambda x: x.rolling(3).max())
            
            # 20일선 수급 지표
            df['volume_zscore_20'] = (df['volume'] - df.groupby('ticker')['volume'].transform(lambda x: x.rolling(20).mean())) / (df.groupby('ticker')['volume'].transform(lambda x: x.rolling(20).std()) + 1e-9)
            
            # 60일 장기 수급 지표 (3달 평균 대비 찐 고래 진입 확인용)
            df['volume_zscore_60'] = (df['volume'] - df.groupby('ticker')['volume'].transform(lambda x: x.rolling(60).mean())) / (df.groupby('ticker')['volume'].transform(lambda x: x.rolling(60).std()) + 1e-9)
        # ---------------------------
        # 4. 로그수익률 및 모멘텀
        # ---------------------------
        df['log_return'] = np.log(df['adj_close']) - np.log(df.groupby('ticker')['adj_close'].shift(1))

        df['momentum_3'] = df.groupby('ticker')['log_return'].transform(lambda x: x.rolling(3).sum())
        df['momentum_5'] = df.groupby('ticker')['log_return'].transform(lambda x: x.rolling(5).sum())

        df['momentum_20'] = df.groupby('ticker')['log_return'].transform(lambda x: x.rolling(20).sum())
        df['momentum_60'] = df.groupby('ticker')['log_return'].transform(lambda x: x.rolling(60).sum())

        df['momentum_accel_3'] = (
            df.groupby('ticker')['momentum_3']
            .transform(lambda x: x.diff().rolling(3).mean())
        )

        df['momentum_accel_5'] = (
            df.groupby('ticker')['momentum_5']
            .transform(lambda x: x.diff().rolling(5).mean())
        )
        df['momentum_accel_20'] = (
            df.groupby('ticker')['momentum_20']
            .transform(lambda x: x.diff().rolling(20).mean())
        )
        
        # 캔들 구조 변수
        df['candle_body'] = (df['adj_close'] - df['open']) / (df['open'] + 1e-9)
        df['high_low_spread'] = (df['high'] - df['low']) / (df['adj_close'] + 1e-9)

        df['high_10'] = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(10).max())
        df['breakout_pressure'] = df['adj_close'] / (df['high_10'] + 1e-9)

        # ---------------------------
        # 시장 상대 강도
        # ---------------------------
        df['relative_strength'] = df['change_rate'] - df['nasdaq_change_rate']

        # 고점 돌파 거리(돌파 직전 패턴 잡기)
        df['high_breakout_20'] =df.groupby('ticker')['high'].transform(lambda x: x.rolling(20).max())
        df['high_breakout_20'] = df['adj_close'] / (df['high_breakout_20'] + 1e-9)

        df['high_breakout_60'] = df.groupby('ticker')['high'].transform(lambda x: x.rolling(60).max())
        df['high_breakout_60'] = df['adj_close'] / (df['high_breakout_60'] + 1e-9)

        # ---------------------------------------------------
        # 미래 3영업일 종가 기준 라벨링
        # ---------------------------------------------------

        if not is_inference:
            # 1. 내일 종가(t1), 모레 종가(t2), 글피 종가(t3)를 미래에서 정확히 당겨오기
            df['close_t1'] = df.groupby('ticker')['adj_close'].shift(-1)
            df['close_t2'] = df.groupby('ticker')['adj_close'].shift(-2)
            df['close_t3'] = df.groupby('ticker')['adj_close'].shift(-3)

            # 2. 오늘 종가(adj_close) 대비 미래 각 영업일 종가의 수익률 계산
            df['return_t1'] = (df['close_t1'] - df['adj_close']) / df['adj_close']
            df['return_t2'] = (df['close_t2'] - df['adj_close']) / df['adj_close']
            df['return_t3'] = (df['close_t3'] - df['adj_close']) / df['adj_close']

            # 3. 미래 3일의 '종가 기준 최고 수익률'만 쏙 뽑아내기
            df['max_return_3d'] = df[['return_t1', 'return_t2', 'return_t3']].max(axis=1)

            # 4. 라벨링: 3일 중 한 번이라도 종가 기준으로 +2.5% 이상 올랐으면 1, 아니면 0
            df['label'] = np.where(df['max_return_3d'] >= 0.025, 1, 0)
        
        # ---------------------------------------------------
        # 6. 컬럼 필터링 및 데이터 동적 정리
        # ---------------------------------------------------
        meta_cols = ['ticker', 'date']
        feature_cols = LSTM_FEATURE_COLS

        # 학습 모드일 때만 'label'을 피처 명단에 결합
        if not is_inference:
            feature_cols.append('label')

        # 필요한 컬럼만 슬라이싱 (존재하지 않는 컬럼 참조 원천 차단)
        df = df[meta_cols + feature_cols]

        # 결측치 제거 (label을 제외한 순수 인풋 지표 기준)
        feature_only = [c for c in feature_cols if c != 'label']
        df = df.dropna(subset=feature_only)

        # 시퀀스 길이(최대 60일) 채울 수 있는 데이터만 남기기 (실전 추론 시 안전하게 60일 확보)
        window_size = 60 if is_inference else 20
        df = df.groupby('ticker').filter(lambda x: len(x) >= window_size).reset_index(drop=True)

        return df

if __name__ == "__main__":
    processor = FeatureProcessorLSTM()
    df = processor.get_raw_data()
    df = processor.calc_technical_indicators(df)

    today = datetime.now().strftime("%Y%m%d")
    df.to_csv(
        f"feature__indicator_lstm{today}.csv",
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
