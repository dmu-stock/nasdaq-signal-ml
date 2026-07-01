import pandas as pd
import numpy as np
from app.database.sqlite_db import get_connection


class BaseFeatureProcessor:
    def __init__(self, db_path: str = "stock_data.db"):
        self.db_path = db_path

    def get_raw_data(self) -> pd.DataFrame:
        conn = get_connection()
        df = pd.read_sql("SELECT * FROM stock_prices ORDER BY date ASC", conn)
        conn.close()
        return df

    def _compute_label_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """미래 t+1~t+3 종가 수익률 컬럼 + (변동성비례 라벨용) 과거 20일 일간변동성."""
        for shift in [1, 2, 3]:
            df[f'close_t{shift}'] = df.groupby('ticker')['adj_close'].shift(-shift)
            df[f'return_t{shift}'] = (df[f'close_t{shift}'] - df['adj_close']) / df['adj_close']
        # 과거 20일 일간수익률 표준편차 (인과적 — 미래 미참조). 변동성비례 라벨 기준.
        _dret = df.groupby('ticker')['adj_close'].pct_change()
        df['_vol20'] = _dret.groupby(df['ticker']).transform(lambda x: x.rolling(20).std())
        return df

    @staticmethod
    def make_label(row, tp: float = 0.025) -> int:
        """
        GBM 라벨. 환경변수 GBM_LABEL로 방식 전환:
          - 'direction' : 다음날 종가 상승 여부 (up=1) — 시장 무관, 신호 약함
          - 'vol'       : 3일 내 종목별 변동성 기준(√3·σ20) 초과 상승 — 시장 무관 + 신호 유지
          - 그 외(기본)  : 3일 안에 +2.5% 달성 (절대 임계) — 고변동성 특화
        """
        import os
        mode = os.environ.get('GBM_LABEL')
        if mode == 'direction':
            r1 = row.get('return_t1')
            if pd.isna(r1):
                return 0
            return int(r1 > 0)
        if mode == 'vol':
            # 그 종목 3일 기대변동(√3·σ20)을 임계로 — 종목별 공정
            vol = row.get('_vol20')
            if pd.isna(vol) or vol == 0:
                return 0
            thr = (3 ** 0.5) * vol   # 3일 1σ 상승
            for i in [1, 2, 3]:
                r = row.get(f'return_t{i}')
                if pd.isna(r):
                    continue
                if r >= thr:
                    return 1
            return 0
        # 기본: 3일 내 +2.5%
        for i in [1, 2, 3]:
            close_r = row.get(f'return_t{i}')
            if pd.isna(close_r):
                continue
            if close_r >= tp:
                return 1
        return 0

    def _apply_labels(self, df: pd.DataFrame, tp: float = 0.025) -> pd.DataFrame:
        df = self._compute_label_columns(df)
        df['label'] = df.apply(lambda row: self.make_label(row, tp), axis=1)
        print("양성 비율:", df['label'].mean())
        return df
    
    def _apply_lstm_labels(self, df, forward_days=5, top_pct=0.30):
        df = df.copy()
        
        # 5일 후 수익률
        df['_fwd'] = (
            df.groupby('ticker')['adj_close'].shift(-forward_days)
            / df['adj_close'] - 1
        )
        
        # 날짜별 상위 30% → 1
        df['label'] = (
            df.groupby('date')['_fwd']
            .transform(lambda x: x.rank(pct=True, method='average'))
            >= (1 - top_pct)
        ).astype(int)
        
        df = df.drop(columns=['_fwd'])
        print(f"LSTM 라벨 양성 비율: {df['label'].mean():.4f}  (목표 ~{top_pct:.0%})")
        return df
