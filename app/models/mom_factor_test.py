# 진단 스크립트 — mom_factor가 2024 하반기에 레짐을 제대로 읽었나
import pandas as pd
import numpy as np

df = pd.read_csv("feature__indicator_lstm20260608.csv")
df['date'] = pd.to_datetime(df['date'])

# 날짜별 mom_factor (종목 공통이라 첫 값만)
daily = df.groupby('date')['mom_factor_20'].first()

for label, s, e in [
    ("2024 하반기", "2024-07-01", "2025-01-01"),
    ("2025 상반기", "2025-01-01", "2025-07-01"),
    ("2025 하반기", "2025-07-01", "2026-01-01"),
    ("2026 상반기", "2026-01-01", "2026-07-01"),
]:
    seg = daily[(daily.index >= s) & (daily.index < e)]
    pos = (seg > 0).mean()
    print(f"{label}: mom_factor 평균 {seg.mean():+.4f}  양수비율 {pos:.0%}  "
          f"(양수=모멘텀장, 음수=반전장)")