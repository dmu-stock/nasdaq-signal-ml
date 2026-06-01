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
        """미래 t+1~t+3 종가 수익률 컬럼 추가."""
        for shift in [1, 2, 3]:
            df[f'close_t{shift}'] = df.groupby('ticker')['adj_close'].shift(-shift)
            df[f'return_t{shift}'] = (df[f'close_t{shift}'] - df['adj_close']) / df['adj_close']
        return df

    @staticmethod
    def make_label(row, tp: float = 0.025) -> int:
        """
        3일 안에 종가 기준 +2.5% 달성하면 1, 아니면 0.
        SL은 모델 라벨이 아닌 실전 리스크 관리 레이어에서 처리한다.
        """
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
    
    def _apply_lstm_labels(self, df, forward_days=20, top_pct=0.30):
        df = df.copy()
        
        # 20일 후 수익률
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
