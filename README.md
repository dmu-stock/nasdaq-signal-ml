# 주먹봇 (Joomuk Bot)
**나스닥 대장주 매수 시그널 생성 시스템**

LightGBM(기술지표) + Dual-Input LSTM(추세) 앙상블로 미국 나스닥 종목의 단기 상승 가능성을 예측하고, 매일 장 마감 후 상위 3개 종목을 추천한다.

---

## 설계 철학

| 모델 | 역할 | 입력 |
|---|---|---|
| **LightGBM** | 기술지표 기반 진입 타이밍 탐지 | 당일 스냅샷 (RSI, MACD, MA 등) |
| **Dual LSTM** | 추세 흐름 감지 | 20일/60일 시퀀스 |
| **AND 게이트** | 두 모델 모두 통과한 종목만 선별 | `prob_lgb >= 0.48 AND prob_lstm >= 0.60` |

추세도 좋고 기술적 타이밍도 맞는 종목만 시그널을 발생시키는 구조.

---

## 라벨 정의

두 모델은 역할이 달라 **라벨도 다르게** 정의한다. GBM은 "지금 사면 곧 오르나"(절대 수익), LSTM은 "오늘 종목 중 어느 게 더 갈까"(상대 순위)를 학습한다.

### LightGBM — 절대 수익 임계 (단기 진입 타이밍)

```
Label = 1  →  3영업일(t+1~t+3) 안에 종가 기준 +2.5% 이상 달성
Label = 0  →  그 외
```

- 종목별 미래 종가(`shift(-1~-3)`)로 수익률을 구해, 셋 중 하나라도 +2.5%를 넘으면 1 (`make_label`, `base_processor.py`).
- 임계값(`tp=0.025`) 고정 → 종목·날짜와 무관한 **절대 기준**.

### Dual LSTM — 횡단면 상위 분위 (상대 추세 강도)

```
Label = 1  →  20영업일 후 수익률이 "그날 전체 종목 중 상위 30%"
Label = 0  →  그 외
```

- 종목별 20일 후 수익률(`shift(-20)`)을 구한 뒤, **날짜별로 순위(percentile)** 를 매겨 상위 30%만 1 (`_apply_lstm_labels`, `base_processor.py`).
- 절대 수익이 아니라 **같은 날 종목 간 상대 비교** → 시장 전체가 오르거나 내려도 양성 비율이 ~30%로 유지된다(레짐 편향 완화).

> **공통:** SL(손절)은 라벨에 포함하지 않는다. 일봉 피처로는 장중 저가 예측이 불가능하므로
> 손절은 실전 리스크 관리 레이어에서 별도 처리 예정.

---

## 성능 (테스트 기간: 2025-07-01 ~ 2026-05-27, 228 영업일)

| 방식 | Top3 일평균 타율 | 베이스라인 대비 |
|---|---|---|
| 시장 평균 (베이스라인) | 0.3680 | 1.00배 |
| LightGBM 단독 | 0.5526 | 1.50배 |
| 앙상블 AND 게이트 | 0.5288 | 1.44배 |

---

## 평가 방법론

금융 시계열 모델의 성능 숫자는 **데이터 누수(look-ahead bias)** 한 줄로 무너진다.
본 프로젝트는 "테스트 구간 정보가 학습에 새어 들어가지 않는다"를 다음 네 가지로 보장한다.

### 1. 시간순 분할 (시간 누수 차단)

랜덤 셔플 분할은 미래 데이터로 과거를 예측하는 누수를 만든다.
모든 분할은 **날짜 기준 단방향(과거 → 미래)** 으로만 한다.

| 구간 | 기간 | 비고 |
|---|---|---|
| Train | ~ 2025-06 | 학습 |
| Validation | GBM: 2024-07 ~ 2025-07 / LSTM: train 내 최신 20% | early stopping·하이퍼파라미터 기준 |
| Test | 2025-07 ~ | 학습·검증에서 한 번도 보지 않은 미래 구간 |

- LSTM의 validation은 train 날짜의 **80 percentile 시점**을 잘라 그 이후를 val로 둔다 (`stock_trainer_lstm.py`). 미래 구간이 val로 들어가지 않는다.
- 테스트 `DataLoader`는 `shuffle=False` — 시퀀스의 시간 순서를 유지한다.

### 2. 스케일러 누수 차단 (fit은 train에서만)

표준화(StandardScaler)를 **전체 데이터에 fit하면 테스트 구간의 평균·분산이 학습에 새어 들어간다.**
LSTM은 종목별 스케일러를 **train 구간(2025-07-01 이전)에만 fit**하고, 그 통계로 test를 transform한다 (`scale_by_ticker`, `stock_trainer_lstm.py`).

```python
train_mask = scaled_df['date'] < '2025-07-01'
sc.fit(scaled_df.loc[train_ticker_mask, features])      # train만 fit
sc.transform(scaled_df.loc[test_ticker_mask, features]) # test는 transform만
```

학습 시 fit한 스케일러는 `ticker_scalers.pkl`로 저장해 **실전 추론에서 동일하게 재사용**한다 (학습/추론 분포 일치).
GBM은 트리 기반이라 스케일링이 불필요하며, 확률 보정(`CalibratedClassifierCV`)도 `cv='prefit'`으로 **validation에서만** 적합한다.

### 3. 미래 피처 차단 (피처 누수 차단)

- 모든 입력 피처는 **인과적(causal)** — `rolling`, `ewm`, `shift(+n)`(과거 방향)만 사용한다. 미래 값을 참조하는 피처는 학습 입력에 없다.
- 라벨 계산용으로 미래 수익률(`forward_5d`, `excess_5d`, `nasdaq_forward_5d`)을 일부 산출하지만, **`GBM_FEATURE_COLS` 명단에서 제외**되며 마지막 컬럼 슬라이싱(`df[meta_cols + feature_cols + ['label']]`)에서 걸러진다. → 학습 입력으로 들어가지 않는다.
- 미래 정보는 **타깃(label)에만** 사용한다: GBM은 `shift(-1~-3)` 종가로 "3영업일 내 +2.5%", LSTM은 `shift(-20)` 수익률의 당일 횡단면 상위 30%. 예측 대상이므로 정상이다.

### 4. Walk-Forward 검증

단일 시간 분할의 좋은 성능이 **특정 시장 국면에 운 좋게 맞은 것인지** 확인하기 위해,
확장 윈도우(expanding window)로 4개 폴드를 잘라 **폴드마다 독립적으로 재학습**하고 AUC를 측정한다 (`walk_forward_eval`, `stock_trainer.py`).

```
Fold 1: train(~2024-01) → test(2024-01 ~ 2024-07)
Fold 2: train(~2024-04) → test(2024-04 ~ 2024-10)
Fold 3: train(~2024-07) → test(2024-07 ~ 2025-01)
Fold 4: train(~2024-10) → test(2024-10 ~ 2025-04)
```

폴드 간 **AUC 표준편차가 크면 특정 구간 과적합**으로 판단한다 (평균과 편차를 함께 리포트).

### 평가 지표

| 지표 | 의미 |
|---|---|
| Top-K 일평균 타율 | 매일 확률 상위 K종목의 라벨 적중률 (실전 추천 방식과 동일) |
| ROC-AUC | 임계값 무관 분류 성능 |
| Walk-Forward AUC 평균 ± 편차 | 시간 구간 안정성 |
| 베이스라인 대비 배수 | 시장 평균(무작위 진입) 대비 개선폭 |

### 한계 (정직한 고지)

- 평가는 **분류 정밀도(타율) 중심**이며, 누적 수익률·샤프지수·MDD 같은 **금융 백테스트 지표와 거래비용·슬리피지는 아직 반영하지 않았다.** 타율 개선이 곧 실현 수익을 보장하지는 않는다.
- 종목 유니버스가 고정(생존 편향 가능)이며, 단일 시장 레짐(2022~2026)에 한정된다.

---

## 프로젝트 구조

```
app/
├── config/
│   └── config.py               # GBM_FEATURE_COLS, LSTM_FEATURE_COLS, TICKERS
│
├── collector/
│   ├── price_yfinance.py       # yfinance 주가/지수 수집 (VIX, TNX, NASDAQ 포함)
│   └── news_crawler.py         # yfinance 뉴스 헤드라인 수집
│
├── features/
│   ├── base_processor.py       # 라벨 생성 공통 로직 (GBM/LSTM 공유)
│   ├── processor.py            # GBM용 기술지표 계산 (FeatureProcessorGBM)
│   └── processor_lstm.py       # LSTM용 시퀀스 피처 계산 (FeatureProcessorLSTM)
│
├── models/
│   ├── lstm_model.py           # PyTorch DualLSTMModel 아키텍처
│   ├── stock_trainer.py        # LightGBM 학습 스크립트
│   ├── stock_trainer_lstm.py   # LSTM 학습 스크립트 (PyTorch)
│   └── stock_trainer_merge.py  # 앙상블 평가 스크립트
│
└── pipeline/
    └── inference_pipeline.py   # 실전 추론 파이프라인 (매일 실행)
```

---

## 모델 아키텍처

### LightGBM
- `n_estimators=2000`, `learning_rate=0.005`
- `class_weight=None`, `scale_pos_weight=2.0`
- Early stopping on validation AUC
- 주요 피처: `nasdaq_change_rate`, `tr_5`, `return_5`, `drawdown_20`, `macd_hist`

### Dual-Input LSTM (PyTorch)

```
Input_20d (20일 시퀀스)          Input_60d (60일 시퀀스)
    │                                   │
InputDropout(0.3)               InputDropout(0.3)
    │                                   │
LSTM(→32) → LSTM(→32)     LSTM(→64) → LSTM(→64) → LSTM(→32)
    │                                   │
 last hidden                        last hidden
    └──────────── Concat (64) ──────────┘
                      │
              Linear(64→48) → LayerNorm → ReLU → Dropout(0.3)
              Linear(48→16) → LayerNorm → ReLU → Dropout(0.2)
              Linear(16→1)  → Sigmoid (추론 시)
```

- Loss: `BCEWithLogitsLoss(pos_weight=...)` — 클래스 불균형 보정
- Optimizer: `Adam(lr=5e-4, weight_decay=1e-4)`
- LR Schedule: `ReduceLROnPlateau(mode='max', factor=0.5, patience=3)`
- Early stopping: val AUC 기준 patience=15

---

## 학습 데이터

| 항목 | 내용 |
|---|---|
| 종목 수 | 44개 (나스닥 대형주, 반도체, AI, 핀테크 등) |
| 학습 기간 | 2022 ~ 2025-06 |
| 테스트 기간 | 2025-07 ~ 현재 |
| GBM val | 2024-07 ~ 2025-07 (early stopping 기준) |
| LSTM val | train 내 최신 20% (시계열 순서 보장) |

---

## 실전 추론 실행

```bash
python -m app.pipeline.inference_pipeline
```

**실행 순서:**
1. `yfinance`로 전체 종목 2년치 주가 수집 (VIX 포함)
2. VIX >= 30 이면 매수 중단 (극공포 구간 가드레일)
3. GBM/LSTM 피처 엔지니어링
4. 종목별 LightGBM + LSTM 추론
5. AND 게이트 필터 → 상위 3개 출력

**출력 예시:**
```
=============================================
[주먹봇] 2026-05-28 매수 시그널
=============================================
진입: NVDA  score=0.5123 [LGBM=0.4921 / LSTM=0.6234]
진입: META  score=0.4987 [LGBM=0.4856 / LSTM=0.6102]
=============================================
```

---

## 학습 스크립트 실행 순서

```bash
# 1. GBM 학습
python -m app.models.stock_trainer

# 2. LSTM 학습
python -m app.models.stock_trainer_lstm

# 3. 앙상블 평가
python -m app.models.stock_trainer_merge
```

---

## 산출물 파일

| 파일 | 설명 |
|---|---|
| `best_lgbm_model.pkl` | LightGBM 모델 가중치 |
| `best_multi_input_lstm.pt` | LSTM 모델 가중치 (PyTorch) |
| `ticker_scalers.pkl` | 종목별 StandardScaler (LSTM 추론용) |
| `prediction_result.csv` | GBM 테스트 예측 결과 |
| `lstm_prediction_result.csv` | LSTM 테스트 예측 결과 |
| `ensemble_prediction_result.csv` | 앙상블 최종 결과 |

---

## 환경

```
Python 3.10+
torch == 2.6.0+cu124   (CUDA 12.4, GPU 학습)
lightgbm
scikit-learn
yfinance
pandas / numpy
joblib
```
