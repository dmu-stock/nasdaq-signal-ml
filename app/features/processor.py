import pandas as pd
import numpy as np
from datetime import datetime
from app.config.config import GBM_FEATURE_COLS
from app.features.base_processor import BaseFeatureProcessor


class FeatureProcessorGBM(BaseFeatureProcessor):
    pass

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

        # df['disparity_20'] = (df['adj_close'] / df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(20).mean())) * 100
        # df['disparity_20'] = df['disparity_20'].fillna(100)

        # MACD (이동평균 수렴 확산 지수) : 단기 이평선과 장기 이평선이 얼마나 빨리 멀어지는지(에너지)를 측정
        short_ema = df.groupby('ticker')['adj_close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
        long_ema = df.groupby('ticker')['adj_close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
        df['macd'] = short_ema - long_ema
        df['macd_signal'] = df.groupby('ticker')['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
        df['macd_hist'] = df['macd'] - df['macd_signal'] # 이 히스토그램이 중요!

        # 5일간의 고가 - 저가 평균 (종목의 활동성)
        df['price_range'] = (df['high'] - df['low']) / df['adj_close']
        df['tr_5'] = df.groupby('ticker')['price_range'].transform(lambda x: x.rolling(5).mean())
        df['tr_20'] = df.groupby('ticker')['price_range'].transform(lambda x: x.rolling(20).mean())
        df['tr_60'] = df.groupby('ticker')['price_range'].transform(lambda x: x.rolling(60).mean())

        # ---------------------------
        # RSI (Momentum)
        # ---------------------------
        delta = df.groupby('ticker')['adj_close'].diff()
        
        # 상승, 하락분 분리
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # 지수이동평균(.ewm)을 사용하여 와일더 평활화 구현
        avg_gain = (
            gain.groupby(df['ticker'])
            .transform(lambda x: x.ewm(com=rsi_period - 1, adjust=False).mean())
        )

        avg_loss = (
            loss.groupby(df['ticker'])
            .transform(lambda x: x.ewm(com=rsi_period - 1, adjust=False).mean())
        )

        # RS 및 RSI 계산
        rs = avg_gain / (avg_loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))

        #이격도
        df['disparity_20'] = (df['adj_close'] - df['ma20']) / df['ma20']

        # 볼린저 밴드 %B
        std = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(20).std())
        ma20 = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(20).mean())
        df['upper_band'] = ma20 + (std * 2)
        df['lower_band'] = ma20 - (std * 2)
        # 현재가가 밴드 내 어디 위치하는지 (0~1 사이 값)
        df['bb_percent'] = (df['adj_close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])

        df['disparity_zscore'] = (df['adj_close'] - ma20) / (std + 1e-9)

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
            # MFI (Money Flow Index) 14일
            # ---------------------------
            df['typical_price'] = (df['high'] + df['low'] + df['adj_close']) / 3
            df['raw_mf']        = df['typical_price'] * df['volume']

            tp_diff = df.groupby('ticker')['typical_price'].diff()
            df['positive_mf'] = df['raw_mf'].where(tp_diff > 0, 0)
            df['negative_mf'] = df['raw_mf'].where(tp_diff < 0, 0)

            pos_flow = df.groupby('ticker')['positive_mf'].transform(
                lambda x: x.rolling(14).sum()
            )
            neg_flow = df.groupby('ticker')['negative_mf'].transform(
                lambda x: x.rolling(14).sum()
            )
            df['mfi'] = 100 - (100 / (1 + pos_flow / (neg_flow + 1e-9)))

            # ---------------------------
            # OBV (On-Balance Volume) slope
            # ---------------------------
             # 가격 방향에 따라 +거래량 / -거래량 누적
            price_diff = df.groupby('ticker')['adj_close'].diff()
            obv_dir = price_diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            df['obv'] = (obv_dir * df['volume']).groupby(df['ticker']).cumsum()

            # 5일 변화량을 20일 표준편차로 나눠 정규화 → 스케일 무관
            df['obv_slope_5'] = df.groupby('ticker')['obv'].transform(
                lambda x: x.diff(5) / (x.rolling(20).std() + 1e-9)
            )
            

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

        # 현재가가 20일 밴드 어디쯤인지 (이미 bb_percent 있으므로 60일 버전 추가)
        std_60 = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(60).std())
        ma_60  = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(60).mean())
        df['bb_percent_60'] = (df['adj_close'] - (ma_60 - 2*std_60)) / (4*std_60 + 1e-9)

        
        df['high_20'] = df.groupby('ticker')['high'].transform(lambda x: x.rolling(20).max())

        # 5일 후 수익률
        df['forward_5d'] = (
            df.groupby('ticker')['adj_close'].shift(-5) / df['adj_close'] - 1
        )
        
        # 나스닥 5일 누적 수익률
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
        
        df['price_position_52w'] = (
        (df['adj_close'] - df.groupby('ticker')['adj_close'].transform(
            lambda x: x.rolling(252).min())) /
        (df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(252).max()) -
        df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(252).min()) + 1e-9)
        )
        # vix/tnx 파생 피처
        df['vix_regime'] = pd.cut(
            df['vix'],
            bins=[0, 15, 20, 25, 30, 999],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)

        df['tnx_change_5']  = df['tnx'].pct_change(5)   # 5일 변화율
        df['tnx_change_20'] = df['tnx'].pct_change(20)

        if not is_inference:
            # 장중 저가 기준 SL + 종가 기준 TP (base_processor.make_label 사용)
            df = self._apply_labels(df)
        else:
            # 라벨이 없으므로 -1로 초기화
            df['label'] = -1
            # 필요한 컬럼만 체크
            check_cols_inf = ['disparity_20', 'alpha_20', 'drawdown_20', 'rsi', 'macd_hist','pullback_zscore']
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
    print("양성 비율:", df['label'].mean())
    print("양성 개수:", df['label'].sum())
    print("음성 개수:", (df['label'] == 0).sum())
    # 학습에 쓴 피처들의 분포
    print(df[GBM_FEATURE_COLS].describe())

    # 추론 시점 데이터 분포
    print(df[GBM_FEATURE_COLS].tail(20).describe())
    print(df[['pullback_zscore']].tail(5))
    print(df['pullback_zscore'].isna().sum())
    print([c for c in GBM_FEATURE_COLS if 'excess' in c or 'forward' in c])
    print("시작일:", df['date'].min())
    print("종료일:", df['date'].max())
    print("총 행수:", len(df))

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