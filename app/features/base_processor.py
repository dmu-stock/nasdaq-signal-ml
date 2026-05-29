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

    def _compute_label_columns(self, df: pd.DataFrame,vol_window: int = 20) -> pd.DataFrame:
        """미래 t+1~t+3 종가 수익률 컬럼 추가."""
        for shift in [1, 2, 3]:
            df[f'close_t{shift}'] = df.groupby('ticker')['adj_close'].shift(-shift)
            df[f'return_t{shift}'] = (df[f'close_t{shift}'] - df['adj_close']) / df['adj_close']

        # --- 종목별 3일 변동성 (과거 vol_window일 일간수익률 표준편차 → 3일로 스케일) ---
        # rolling 윈도우는 t 시점까지의 과거만 사용하므로 룩어헤드 없음.
        ret_1d = df.groupby('ticker')['adj_close'].pct_change()
        sigma_1d = ret_1d.groupby(df['ticker']).transform(
            lambda s: s.rolling(vol_window, min_periods=vol_window // 2).std()
        )
        df['vol_target_unit'] = sigma_1d * np.sqrt(3)   # 라벨이 '3일 수익률'이므로 √3 스케일
        return df

    @staticmethod
    def make_label(row, k: float = 1.0, min_tp: float = 0.01) -> int:
        """
        3일 안에 종가 기준 +2.5% 달성하면 1, 아니면 0.
        SL은 모델 라벨이 아닌 실전 리스크 관리 레이어에서 처리한다.
        """
        unit = row.get('vol_target_unit')
        if pd.isna(unit):
            return 0

        target = max(k * unit, min_tp)   # 종목별 동적 임계치

        for i in [1, 2, 3]:
            close_r = row.get(f'return_t{i}')
            if pd.isna(close_r):
                continue
            if close_r >= target:
                return 1
        return 0

    def _apply_labels(self, df: pd.DataFrame, k: float = 1.0) -> pd.DataFrame:
        df = self._compute_label_columns(df)
        df['label'] = df.apply(lambda row: self.make_label(row, k=k), axis=1)
        pos_rate = df['label'].mean()
        print(f"양성 비율:, {pos_rate:.4f}")
        
        # 참고: 종목별 동적 목표 분포 (튜닝 감각용)
        tgt = (k * df['vol_target_unit']).clip(lower=0.01)
        print(f"[label] 종목 목표(%) 분포  중앙값 {tgt.median()*100:.2f}% / "
              f"10%~90%: {tgt.quantile(0.1)*100:.2f}% ~ {tgt.quantile(0.9)*100:.2f}%")
        return df
