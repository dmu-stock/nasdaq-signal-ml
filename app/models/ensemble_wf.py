"""
앙상블 Walk-Forward 검증
폴드마다 GBM + LSTM 둘 다 독립 재학습 → AND 게이트 → 구간별 Top3 타율.
배포 파이프라인과 동일: GBM calibration(sigmoid), LSTM 종목별 스케일러 재적합.

실행: python -m app.models.ensemble_wf
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random

random.seed(42); np.random.seed(42); torch.manual_seed(42)

import os
from app.config.config import GBM_FEATURE_COLS, LSTM_FEATURE_COLS
from app.models.lstm_model import DualLSTMModel, SingleLSTMModel
# 환경변수 LSTM_ARCH=single 이면 Single-LSTM으로 비교 (기본: Dual)
LSTMClass = SingleLSTMModel if os.environ.get('LSTM_ARCH') == 'single' else DualLSTMModel
print(f"[LSTM 아키텍처] {LSTMClass.__name__}")
from lightgbm import LGBMClassifier, early_stopping
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = device.type == 'cuda'

# 환경변수 DATA_TAG로 데이터셋 전환 (기본 20260623=나스닥, 20260701=S&P 검증)
_TAG = os.environ.get('DATA_TAG', '20260623')
GBM_CSV  = f"feature__indicator_{_TAG}.csv"
LSTM_CSV = f"feature__indicator_lstm{_TAG}.csv"
print(f"[데이터셋] {_TAG}")

LGBM_TH = 0.54
LSTM_TH = 0.49
SEQ20, SEQ60 = 20, 60

# 폴드: (학습 끝 = 테스트 시작, 테스트 끝)
FOLDS = [
    ('2024-07-01', '2025-01-01'),
    ('2025-01-01', '2025-07-01'),
    ('2025-07-01', '2026-01-01'),
    ('2026-01-01', '2026-07-01'),
]

gbm_df  = pd.read_csv(GBM_CSV);  gbm_df['date']  = pd.to_datetime(gbm_df['date'])
lstm_df = pd.read_csv(LSTM_CSV); lstm_df['date'] = pd.to_datetime(lstm_df['date'])

gfeat, lfeat = GBM_FEATURE_COLS, LSTM_FEATURE_COLS

# ---------------------------------------------------
# 고전 전략 베이스라인용 신호 (Jegadeesh-Titman 모멘텀, RSI)
# bluechip_price_data.csv에서 종목별 20일 모멘텀 계산 후 (date,ticker)로 머지
# ---------------------------------------------------
px = pd.read_csv("bluechip_price_data.csv")
px['date'] = pd.to_datetime(px['date'])
px = px.sort_values(['ticker', 'date'])
# 과거 20일 수익률 (어제까지 → 인과적). t-1 기준 모멘텀으로 t일 진입
px['mom20'] = px.groupby('ticker')['adj_close'].transform(lambda s: s.pct_change(20)).groupby(px['ticker']).shift(1)
signal_df = px[['date', 'ticker', 'mom20']]


# ---------------------------------------------------
# GBM: 폴드별 학습 + sigmoid 보정 → 테스트 예측
# ---------------------------------------------------
def gbm_fold(tr_end, te_end):
    d = gbm_df[gbm_df['date'] < te_end].copy()
    tr = d[d['date'] < tr_end]
    te = d[d['date'] >= tr_end]
    if len(te) < 50:
        return None
    cut = tr['date'].quantile(0.8)
    core = tr[tr['date'] <= cut]
    val  = tr[tr['date'] >  cut]

    m = LGBMClassifier(
        n_estimators=2000, learning_rate=0.005, max_depth=6, num_leaves=31,
        min_data_in_leaf=50, feature_fraction=0.8, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, lambda_l1=0.05, lambda_l2=0.05, objective='binary',
        boosting_type='gbdt', force_col_wise=True, random_state=42,
        scale_pos_weight=2.0, verbose=-1,
    )
    m.fit(core[gfeat], core['label'],
          eval_set=[(val[gfeat], val['label'])],
          callbacks=[early_stopping(50, verbose=False)])
    cal = CalibratedClassifierCV(m, method='sigmoid', cv='prefit')
    cal.fit(val[gfeat], val['label'])

    out = te[['date', 'ticker', 'label', 'rsi']].copy()   # rsi: RSI 역추세 베이스라인용
    out['prob_lgb'] = cal.predict_proba(te[gfeat])[:, 1]
    return out  # actual = label(3일 +2.5%) = 매매 타깃


# ---------------------------------------------------
# LSTM: 폴드별 스케일러 재적합 + 학습 → 테스트 예측
# ---------------------------------------------------
def scale_fold(d, tr_end):
    d = d.sort_values(['ticker', 'date']).reset_index(drop=True).copy()
    d[lfeat] = d[lfeat].replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=lfeat).reset_index(drop=True)
    tr = d['date'] < tr_end
    for tk, _ in d.groupby('ticker'):
        mk = d['ticker'] == tk
        if (mk & tr).sum() == 0:
            continue
        sc = StandardScaler().fit(d.loc[mk & tr, lfeat])
        d.loc[mk, lfeat] = sc.transform(d.loc[mk, lfeat])
    return d

def make_seq(d):
    X20, X60, y, tks, dts = [], [], [], [], []
    for tk, g in d.groupby('ticker'):
        g = g.sort_values('date')
        if len(g) < SEQ60:
            continue
        f, t, dd = g[lfeat].values, g['label'].values, g['date'].values
        for i in range(SEQ60 - 1, len(g)):
            X20.append(f[i-SEQ20+1:i+1]); X60.append(f[i-SEQ60+1:i+1])
            y.append(t[i]); tks.append(tk); dts.append(dd[i])
    return (np.array(X20, np.float32), np.array(X60, np.float32),
            np.array(y, np.float32), np.array(tks), np.array(dts))

def dl(x20, x60, y, sh):
    ds = TensorDataset(torch.from_numpy(x20), torch.from_numpy(x60), torch.from_numpy(y))
    return DataLoader(ds, batch_size=128, shuffle=sh, pin_memory=use_amp)

def lstm_fold(tr_end, te_end):
    d = scale_fold(lstm_df[lstm_df['date'] < te_end].copy(), tr_end)
    X20, X60, y, tks, dts = make_seq(d)
    dts = pd.to_datetime(dts)
    trm = dts < pd.Timestamp(tr_end)
    tem = dts >= pd.Timestamp(tr_end)
    if tem.sum() < 50 or len(np.unique(y[trm])) < 2:
        return None
    tr_ns = dts[trm].astype(np.int64).values
    cut = np.percentile(tr_ns, 80)
    fin = tr_ns <= cut; vm = tr_ns > cut

    model = LSTMClass(len(lfeat)).to(device)
    cw = compute_class_weight('balanced', classes=np.unique(y[trm][fin]), y=y[trm][fin])
    pw = torch.tensor([cw[1]/cw[0]], dtype=torch.float32).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', factor=0.5, patience=3, min_lr=1e-5)
    sc_amp = torch.amp.GradScaler('cuda', enabled=use_amp)

    tl = dl(X20[trm][fin], X60[trm][fin], y[trm][fin], True)
    vl = dl(X20[trm][vm],  X60[trm][vm],  y[trm][vm],  False)
    best, best_w, pat = 0, None, 0
    for ep in range(100):
        model.train()
        for a, b, c in tl:
            a, b, c = a.to(device), b.to(device), c.to(device)
            opt.zero_grad()
            with torch.autocast(device.type, enabled=use_amp):
                loss = crit(model(a, b), c)
            sc_amp.scale(loss).backward(); sc_amp.step(opt); sc_amp.update()
        model.eval(); vp, vy = [], []
        with torch.no_grad():
            for a, b, c in vl:
                a, b = a.to(device), b.to(device)
                with torch.autocast(device.type, enabled=use_amp):
                    vp += torch.sigmoid(model(a, b)).cpu().tolist()
                vy += c.tolist()
        va = roc_auc_score(vy, vp); sch.step(va)
        if va > best:
            best, best_w, pat = va, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 15:
                break
    model.load_state_dict(best_w)

    model.eval(); tp = []
    teL = dl(X20[tem], X60[tem], y[tem], False)
    with torch.no_grad():
        for a, b, c in teL:
            a, b = a.to(device), b.to(device)
            with torch.autocast(device.type, enabled=use_amp):
                tp += torch.sigmoid(model(a, b)).cpu().tolist()
    return pd.DataFrame({'date': dts[tem], 'ticker': tks[tem],
                         'prob_lstm': tp, 'label_lstm': y[tem]})


# ---------------------------------------------------
# 폴드 루프
# ---------------------------------------------------
print("===== 앙상블 Walk-Forward (GBM 후보 → LSTM 재정렬) =====\n")

# 폴드별 예측을 한 번만 계산해서 재사용 (TOP_N 여러 개 비교용)
fold_data = []
for tr_end, te_end in FOLDS:
    g = gbm_fold(tr_end, te_end)
    l = lstm_fold(tr_end, te_end)
    if g is None or l is None:
        print(f"{tr_end} ~ {te_end}  스킵"); continue
    e = pd.merge(g, l, on=['date', 'ticker'], how='inner')
    e = pd.merge(e, signal_df, on=['date', 'ticker'], how='left')   # mom20 추가
    fold_data.append((tr_end, te_end, e))

GBM_MIN = 0.50   # 실전 파이프라인과 동일: GBM 최소 기준 (현금 보유 여지)

# ---------------------------------------------------
# 임계값 비교: GBM_MIN 여러 값 (TOP_N=8 고정) — 논문 표용
# ---------------------------------------------------
print("===== 임계값(GBM_MIN) 비교 (TOP_N=8) =====")
print(f"{'GBM_MIN':<10}{'재정렬 적중률':<14}{'배수':<8}{'진입수':<8}{'폴드별'}")
for gmin in [0.40, 0.45, 0.50, 0.55, 0.60]:
    rows_g = []
    for tr_end, te_end, e in fold_data:
        base = e['label'].mean()
        rer = []
        for date, grp in e.groupby('date'):
            pool = grp[grp['prob_lgb'] >= gmin]
            cand = pool.sort_values('prob_lgb', ascending=False).head(8)
            rer += cand.sort_values('prob_lstm', ascending=False).head(3)['label'].tolist()
        hit = np.mean(rer) if rer else 0.0
        rows_g.append((hit, base, len(rer)))
    hits_g = [r[0] for r in rows_g]
    mults_g = [r[0]/r[1] for r in rows_g if r[1] > 0]
    n_g = sum(r[2] for r in rows_g)
    folds_g = ' / '.join(f'{h:.3f}' for h in hits_g)
    print(f"{gmin:<10}{np.mean(hits_g):<14.4f}{np.mean(mults_g):<8.2f}{n_g:<8}{folds_g}")
print()

for TOP_N_GBM in [5, 6, 8, 10]:
    rows = []
    for tr_end, te_end, e in fold_data:
        base = e['label'].mean()
        # GBM_MIN 이상 중 상위 N개 후보 → 그 안에서 LSTM 순위 상위 3개
        rer, gbm_only, lstm_only = [], [], []
        for date, grp in e.groupby('date'):
            pool = grp[grp['prob_lgb'] >= GBM_MIN]
            cand = pool.sort_values('prob_lgb', ascending=False).head(TOP_N_GBM)
            r3 = cand.sort_values('prob_lstm', ascending=False).head(3)
            rer += r3['label'].tolist()
            gbm_only += grp.sort_values('prob_lgb', ascending=False).head(3)['label'].tolist()
            lstm_only += grp.sort_values('prob_lstm', ascending=False).head(3)['label'].tolist()

        hit   = np.mean(rer) if rer else 0.0
        g_hit = np.mean(gbm_only) if gbm_only else 0.0
        l_hit = np.mean(lstm_only) if lstm_only else 0.0
        mult  = hit/base if base > 0 else 0
        rows.append((hit, base, mult, g_hit, l_hit, len(rer)))

    hits  = [r[0] for r in rows]
    mults = [r[2] for r in rows]
    g_avg = np.mean([r[3] for r in rows])
    l_avg = np.mean([r[4] for r in rows])
    n_tot = sum(r[5] for r in rows)
    print(f"[TOP_N={TOP_N_GBM}]  재정렬 {np.mean(hits):.4f} ({np.mean(mults):.2f}배, "
          f"편차 {np.std(mults):.3f}, 진입 {n_tot}회)  | GBM단독 {g_avg:.4f}  LSTM단독 {l_avg:.4f}")

# ---------------------------------------------------
# 최종 채택 설정(TOP_N=8) 폴드별 상세 — 논문 표용
# ---------------------------------------------------
# ---------------------------------------------------
# 같은 4폴드에서 모델별 AUC (각 모델은 자기 라벨 기준) — 논문 표 1용
# ---------------------------------------------------
print("\n===== 모델별 Walk-Forward AUC (동일 폴드) =====")
for tr_end, te_end, e in fold_data:
    gbm_auc  = roc_auc_score(e['label'], e['prob_lgb'])
    lstm_auc = roc_auc_score(e['label_lstm'], e['prob_lstm'])
    print(f"{tr_end}~{te_end}  GBM_AUC {gbm_auc:.4f}  LSTM_AUC {lstm_auc:.4f}")

print("\n===== 최종 설정 폴드별 상세 (TOP_N=8, GBM_MIN=0.50) =====")
TOP_N_GBM = 8
for tr_end, te_end, e in fold_data:
    base = e['label'].mean()
    rer, gbm_only, lstm_only = [], [], []
    for date, grp in e.groupby('date'):
        pool = grp[grp['prob_lgb'] >= GBM_MIN]
        cand = pool.sort_values('prob_lgb', ascending=False).head(TOP_N_GBM)
        rer += cand.sort_values('prob_lstm', ascending=False).head(3)['label'].tolist()
        gbm_only += grp.sort_values('prob_lgb', ascending=False).head(3)['label'].tolist()
        lstm_only += grp.sort_values('prob_lstm', ascending=False).head(3)['label'].tolist()
    print(f"{tr_end}~{te_end}  재정렬 {np.mean(rer):.4f}  GBM단독 {np.mean(gbm_only):.4f}  "
          f"LSTM단독 {np.mean(lstm_only):.4f}  베이스 {base:.4f}  (진입 {len(rer)}회)")

# ---------------------------------------------------
# 고전 전략 베이스라인 비교 (동일 폴드·동일 평가) — 논문 표용
#   · 단순 모멘텀(Jegadeesh-Titman): 과거 20일 수익률 상위 3
#   · RSI 역추세: RSI 낮은(과매도) 3
#   · 시장평균: 라벨 자연 발생 비율
# ---------------------------------------------------
print("\n===== 고전 전략 베이스라인 비교 (Top3 적중률) =====")
TOP_N_GBM = 8
strat = {'시장평균': [], '단순모멘텀': [], 'RSI역추세': [], 'GBM단독': [], '재정렬앙상블': []}
for tr_end, te_end, e in fold_data:
    s = {k: [] for k in strat}
    for date, grp in e.groupby('date'):
        s['시장평균'].append(grp['label'].mean())
        mom = grp.dropna(subset=['mom20'])
        if len(mom) >= 3:
            s['단순모멘텀'] += mom.sort_values('mom20', ascending=False).head(3)['label'].tolist()
        s['RSI역추세'] += grp.sort_values('rsi', ascending=True).head(3)['label'].tolist()
        s['GBM단독'] += grp.sort_values('prob_lgb', ascending=False).head(3)['label'].tolist()
        pool = grp[grp['prob_lgb'] >= GBM_MIN]
        cand = pool.sort_values('prob_lgb', ascending=False).head(TOP_N_GBM)
        s['재정렬앙상블'] += cand.sort_values('prob_lstm', ascending=False).head(3)['label'].tolist()
    for k in strat:
        strat[k].append(np.mean(s[k]))

print(f"{'전략':<12}{'평균 Top3 적중률':<16}{'폴드별'}")
for k in ['시장평균', '단순모멘텀', 'RSI역추세', 'GBM단독', '재정렬앙상블']:
    folds_str = ' / '.join(f'{v:.3f}' for v in strat[k])
    print(f"{k:<12}{np.mean(strat[k]):<16.4f}{folds_str}")

# ---------------------------------------------------
# 통계적 유의성 (TOP_N=8) — Bootstrap CI + 짝지은 검정
# ---------------------------------------------------
from scipy import stats

TOP_N_GBM = 8
# 날짜를 블록으로 보존: 각 날의 픽 묶음(list)을 통째로 저장
ens_days, gbm_days = [], []           # 각 원소 = 그날 픽들의 라벨 리스트
daily_ens, daily_gbm = [], []         # 일별 적중률 쌍 — 짝지은 검정용

for tr_end, te_end, e in fold_data:
    for date, grp in e.groupby('date'):
        pool = grp[grp['prob_lgb'] >= GBM_MIN]
        cand = pool.sort_values('prob_lgb', ascending=False).head(TOP_N_GBM)
        r3 = cand.sort_values('prob_lstm', ascending=False).head(3)['label'].tolist()
        g3 = grp.sort_values('prob_lgb', ascending=False).head(3)['label'].tolist()
        if not r3 or not g3:
            continue
        ens_days.append(r3)
        gbm_days.append(g3)
        daily_ens.append(np.mean(r3))
        daily_gbm.append(np.mean(g3))

daily_ens = np.array(daily_ens)
daily_gbm = np.array(daily_gbm)

def block_boot_ci(day_lists, n=2000, seed=42):
    """일(day) 단위 블록 부트스트랩: 날짜를 복원추출하고, 뽑힌 날들의
    픽을 모두 모아 전체 평균을 계산. 같은 날 종목 상관(블록)을 보존한다."""
    rng = np.random.default_rng(seed)
    D = len(day_lists)
    idx = np.arange(D)
    means = []
    for _ in range(n):
        pick = rng.choice(idx, size=D, replace=True)
        pooled = [v for d in pick for v in day_lists[d]]
        means.append(np.mean(pooled))
    return np.percentile(means, 2.5), np.percentile(means, 97.5)

ens_mean = np.mean([v for d in ens_days for v in d])
gbm_mean = np.mean([v for d in gbm_days for v in d])

print("\n===== 통계적 유의성 (TOP_N=8, 일 단위 블록 부트스트랩) =====")
e_lo, e_hi = block_boot_ci(ens_days)
g_lo, g_hi = block_boot_ci(gbm_days)
print(f"앙상블  적중률 {ens_mean:.4f}  95% CI [{e_lo:.4f}, {e_hi:.4f}]  (영업일 {len(ens_days)}일)")
print(f"GBM단독 적중률 {gbm_mean:.4f}  95% CI [{g_lo:.4f}, {g_hi:.4f}]  (영업일 {len(gbm_days)}일)")

# 일별 적중률 짝지은 검정 (앙상블 vs GBM단독)
diff = daily_ens - daily_gbm
#t-검정
t_stat, t_p = stats.ttest_rel(daily_ens, daily_gbm)
try: # wilcoxon 검증
    w_stat, w_p = stats.wilcoxon(daily_ens, daily_gbm)
except ValueError:
    w_p = float('nan')
print(f"\n일별 적중률 차이 (앙상블 - GBM단독): 평균 {diff.mean():+.4f}  (영업일 {len(diff)}일)")
print(f"  짝지은 t-test : t={t_stat:.3f}, p={t_p:.4f}")
print(f"  Wilcoxon      : p={w_p:.4f}")

# 차이값(앙상블-GBM)의 CI — 같은 날 쌍을 블록으로 보존 (CI 겹침 오해 차단)
def diff_block_boot_ci(ens_days, gbm_days, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    D = len(ens_days); idx = np.arange(D); diffs = []
    for _ in range(n):
        pick = rng.choice(idx, size=D, replace=True)
        e = np.mean([v for d in pick for v in ens_days[d]])
        g = np.mean([v for d in pick for v in gbm_days[d]])
        diffs.append(e - g)
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

d_lo, d_hi = diff_block_boot_ci(ens_days, gbm_days)
print(f"  차이값 95% CI : [{d_lo:.4f}, {d_hi:.4f}]  "
      f"→ {'0 미포함, 유의' if d_lo > 0 else '0 포함'}")
print(f"  → p<0.05 이면 앙상블 우위가 통계적으로 유의함")
