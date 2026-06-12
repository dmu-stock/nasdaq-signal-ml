import pandas as pd
import numpy as np
from datetime import datetime
from app.config.config import LSTM_FEATURE_COLS
from app.features.base_processor import BaseFeatureProcessor


class FeatureProcessorLSTM(BaseFeatureProcessor):
    pass

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
            # 60일 장기 수급 지표
            df['volume_zscore_60'] = (df['volume'] - df.groupby('ticker')['volume'].transform(lambda x: x.rolling(60).mean())) / (df.groupby('ticker')['volume'].transform(lambda x: x.rolling(60).std()) + 1e-9)
        # ---------------------------
        # 4. 로그수익률 및 모멘텀
        # ---------------------------
        df['log_return'] = np.log(df['adj_close']) - np.log(df.groupby('ticker')['adj_close'].shift(1))

        #로그리턴 차분값 평균
        df['momentum_3'] = df.groupby('ticker')['log_return'].transform(lambda x: x.rolling(3).sum())
        df['momentum_5'] = df.groupby('ticker')['log_return'].transform(lambda x: x.rolling(5).sum())
        df['momentum_20'] = df.groupby('ticker')['log_return'].transform(lambda x: x.rolling(20).sum())
        df['momentum_60'] = df.groupby('ticker')['log_return'].transform(lambda x: x.rolling(60).sum())

        # 가속도
        df['momentum_accel_3'] = df.groupby('ticker')['momentum_3'].transform(lambda x: x.diff().rolling(3).mean())
        df['momentum_accel_5'] = df.groupby('ticker')['momentum_5'].transform(lambda x: x.diff().rolling(5).mean())
        df['momentum_accel_20'] = df.groupby('ticker')['momentum_20'].transform(lambda x: x.diff().rolling(20).mean())
        
        
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

        # ---------------------------
        # vix/tnx 파생 피처
        # ---------------------------
        # VIX가 개별 종목의 변동성에 비해 얼마나 과도한지 적은지
        df['vix_vs_stock_vol'] = df['vix'] / (df['volatility_5'] * 100 + 1e-9)
        # ---------------------------
        # test 피처 BMG 핵심 피처의 시계열 추가
        # ---------------------------
        
        # rsi
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

        # MACD (이동평균 수렴 확산 지수) : 단기 이평선과 장기 이평선이 얼마나 빨리 멀어지는지(에너지)를 측정
        short_ema = df.groupby('ticker')['adj_close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
        long_ema = df.groupby('ticker')['adj_close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
        df['macd'] = short_ema - long_ema
        df['macd_signal'] = df.groupby('ticker')['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # 거래량 흐름
        df['volume_ma5'] = (
                df.groupby('ticker')['volume']
                .transform(lambda x: x.rolling(5).mean())
            )
        df['volume_ratio'] = df['volume'] / (df['volume_ma5'] + 1e-9)

        # 최고가 대비 하락률 (High Drawdown)
        df['max_20'] = df.groupby('ticker')['high'].transform(lambda x: x.rolling(20).max())
        df['drawdown_20'] = (df['adj_close'] - df['max_20']) / df['max_20']

        # 볼린저 밴드 %B
        std = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(20).std())
        ma20 = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(20).mean())
        df['upper_band'] = ma20 + (std * 2)
        df['lower_band'] = ma20 - (std * 2)
        # 현재가가 밴드 내 어디 위치하는지 (0~1 사이 값)
        df['bb_percent'] = (df['adj_close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])

        df['price_position_52w'] = (
        (df['adj_close'] - df.groupby('ticker')['adj_close'].transform(
            lambda x: x.rolling(252).min())) /
        (df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(252).max()) -
        df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(252).min()) + 1e-9)
        )
        
        # ---------------------------
        # 시장 모멘텀 팩터 (레짐 인지)
        # "모멘텀 전략이 현재 작동 중인가" = 추세장 vs 반전장 신호
        # ---------------------------
        # 어제까지의 20일 모멘텀 (오늘 정보 사용 안 함 → 인과적)
        df['_mom_lag'] = df.groupby('ticker')['momentum_20'].shift(1)

        def _mom_factor(g):
            # g: 같은 날짜의 전체 종목
            valid = g['_mom_lag'].notna()
            if valid.sum() < 5:
                return np.nan
            hi_th = g['_mom_lag'].quantile(0.7)   # 모멘텀 상위 30%
            lo_th = g['_mom_lag'].quantile(0.3)   # 모멘텀 하위 30%
            hi = g.loc[g['_mom_lag'] >= hi_th, 'return_1'].mean()
            lo = g.loc[g['_mom_lag'] <= lo_th, 'return_1'].mean()
            return hi - lo   # 양수=모멘텀 작동(추세장), 음수=반전장

        # 날짜별 팩터 수익률 (전 종목 공통 신호)
        factor = df.groupby('date', group_keys=False).apply(_mom_factor).sort_index()
        factor_20 = factor.rolling(20).sum()    # 20일 누적 → 레짐 신호 증폭

        df['mom_factor']   = df['date'].map(factor)
        df['mom_factor_20'] = df['date'].map(factor_20)

        df['momentum_regime_adj'] = df['momentum_20'] * np.sign(df['mom_factor_20'])

        df = df.drop(columns=['_mom_lag'])

        # ---------------------------
        # ADX (추세 강도) — 레짐 인지
        # ---------------------------
        high, low, close = df['high'], df['low'], df['adj_close']
        prev_close = df.groupby('ticker')['adj_close'].shift(1)

        # True Range (이미 tr 있으면 재사용 가능)
        tr_adx = np.maximum(high - low,
                 np.maximum((high - prev_close).abs(), (low - prev_close).abs()))

        # +DM, -DM
        up_move = high - df.groupby('ticker')['high'].shift(1)
        dn_move = df.groupby('ticker')['low'].shift(1) - low
        plus_dm  = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)

        df['_plus_dm']  = plus_dm
        df['_minus_dm'] = minus_dm
        df['_tr_adx']   = tr_adx

        # 14일 평활 (Wilder)
        atr14 = df.groupby('ticker')['_tr_adx'].transform(lambda x: x.ewm(alpha=1/14, adjust=False).mean())
        pdi = 100 * df.groupby('ticker')['_plus_dm'].transform(lambda x: x.ewm(alpha=1/14, adjust=False).mean()) / (atr14 + 1e-9)
        mdi = 100 * df.groupby('ticker')['_minus_dm'].transform(lambda x: x.ewm(alpha=1/14, adjust=False).mean()) / (atr14 + 1e-9)

        dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
        df['dx'] = dx
        df['adx'] = df.groupby('ticker')['dx'].transform(lambda x: x.ewm(alpha=1/14, adjust=False).mean())
        df = df.drop(columns=['_plus_dm', '_minus_dm', '_tr_adx', 'dx'])

        
        # ---------------------------------------------------
        # 미래 3영업일 종가 기준 라벨링
        # ---------------------------------------------------

        if not is_inference:
            # 장중 저가 기준 SL + 종가 기준 TP (base_processor.make_label 사용)
            df = self._apply_lstm_labels(df)
        
        # ---------------------------------------------------
        # 6. 컬럼 필터링 및 데이터 동적 정리
        # ---------------------------------------------------
        meta_cols = ['ticker', 'date']
        # feature_cols = LSTM_FEATURE_COLS.copy()
        feature_cols = LSTM_FEATURE_COLS.copy()

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
