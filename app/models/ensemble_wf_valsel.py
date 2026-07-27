"""
앙상블 Walk-Forward — 검증구간 기반 하이퍼파라미터 선택 (test 무오염 버전)

ensemble_wf.py와의 차이:
  - 기존: GBM_MIN / TOP_N을 test 적중률을 보고 선택 (test set 오염)
  - 본안: 학습구간을 3분할하여 val_sel(보정에 쓰이지 않은 구간)에서만
          임계값을 선택하고, test는 평가에만 사용

  학습구간 분할 (폴드별, 날짜 기준)
    core    ~ p70   : 모델 학습
    val_es  p70~p85 : early stopping + 확률 보정(GBM) / best weight(LSTM)
    val_sel p85~    : GBM_MIN·TOP_N 선택
    test            : 최종 평가 (선택 과정에서 일절 참조하지 않음)

  폴드마다 독립적으로 임계값을 선택한다(폴드 간 정보 공유 없음).

실행: python -m app.models.ensemble_wf_valsel
"""
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

random.seed(42); np.random.seed(42); torch.manual_seed(42)

from app.config.config import GBM_FEATURE_COLS, LSTM_FEATURE_COLS
from app.models.lstm_model import DualLSTMModel, SingleLSTMModel
from lightgbm import LGBMClassifier, early_stopping
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
from scipy import stats

LSTMClass = SingleLSTMModel if os.environ.get('LSTM_ARCH') == 'single' else DualLSTMModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = device.type == 'cuda'

_TAG = os.environ.get('DATA_TAG', '20260623')
GBM_CSV = f"feature__indicator_{_TAG}.csv"
LSTM_CSV = f"feature__indicator_lstm{_TAG}.csv"
print(f"[LSTM 아키텍처] {LSTMClass.__name__}   [데이터셋] {_TAG}   [device] {device}")

SEQ20, SEQ60 = 20, 60
# 학습구간 3분할 지점(퍼센타일) — 환경변수로 조정 가능
P_CORE = int(os.environ.get('P_CORE', 70))
P_SEL = int(os.environ.get('P_SEL', 85))
FIX_GMIN, FIX_TOPN = 0.50, 8    # 사전 고정 설정(대조군)
GBM_MIN_GRID = [0.40, 0.45, 0.50, 0.55, 0.60]
TOP_N_GRID = [5, 6, 8, 10]

FOLDS = [
    ('2024-07-01', '2025-01-01'),
    ('2025-01-01', '2025-07-01'),
    ('2025-07-01', '2026-01-01'),
    ('2026-01-01', '2026-07-01'),
]

gbm_df = pd.read_csv(GBM_CSV);  gbm_df['date'] = pd.to_datetime(gbm_df['date'])
lstm_df = pd.read_csv(LSTM_CSV); lstm_df['date'] = pd.to_datetime(lstm_df['date'])
gfeat, lfeat = GBM_FEATURE_COLS, LSTM_FEATURE_COLS


# ---------------------------------------------------
# GBM: core 학습 → val_es 보정 → val_sel·test 예측 반환
# ---------------------------------------------------
def gbm_fold(tr_end, te_end):
    d = gbm_df[gbm_df['date'] < te_end].copy()
    tr = d[d['date'] < tr_end]
    te = d[d['date'] >= tr_end]
    if len(te) < 50:
        return None, None

    c1 = tr['date'].quantile(P_CORE / 100)
    c2 = tr['date'].quantile(P_SEL / 100)
    core = tr[tr['date'] <= c1]
    v_es = tr[(tr['date'] > c1) & (tr['date'] <= c2)]
    v_sel = tr[tr['date'] > c2]
    if len(v_es) < 50 or len(v_sel) < 50:
        return None, None

    m = LGBMClassifier(
        n_estimators=2000, learning_rate=0.005, max_depth=6, num_leaves=31,
        min_data_in_leaf=50, feature_fraction=0.8, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, lambda_l1=0.05, lambda_l2=0.05, objective='binary',
        boosting_type='gbdt', force_col_wise=True, random_state=42,
        scale_pos_weight=2.0, verbose=-1,
    )
    m.fit(core[gfeat], core['label'],
          eval_set=[(v_es[gfeat], v_es['label'])],
          callbacks=[early_stopping(50, verbose=False)])
    cal = CalibratedClassifierCV(m, method='sigmoid', cv='prefit')
    cal.fit(v_es[gfeat], v_es['label'])          # 보정은 val_es에서만

    def _pack(df_):
        o = df_[['date', 'ticker', 'label']].copy()
        o['prob_lgb'] = cal.predict_proba(df_[gfeat])[:, 1]
        return o
    return _pack(v_sel), _pack(te)


# ---------------------------------------------------
# LSTM: core 학습 → val_es로 early stopping → val_sel·test 예측
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
            X20.append(f[i - SEQ20 + 1:i + 1]); X60.append(f[i - SEQ60 + 1:i + 1])
            y.append(t[i]); tks.append(tk); dts.append(dd[i])
    return (np.array(X20, np.float32), np.array(X60, np.float32),
            np.array(y, np.float32), np.array(tks), np.array(dts))


def dl(x20, x60, y, sh):
    ds = TensorDataset(torch.from_numpy(x20), torch.from_numpy(x60), torch.from_numpy(y))
    return DataLoader(ds, batch_size=128, shuffle=sh, pin_memory=use_amp)


def _infer(model, x20, x60, y):
    model.eval(); out = []
    with torch.no_grad():
        for a, b, _ in dl(x20, x60, y, False):
            a, b = a.to(device), b.to(device)
            with torch.autocast(device.type, enabled=use_amp):
                out += torch.sigmoid(model(a, b)).cpu().tolist()
    return out


def lstm_fold(tr_end, te_end):
    d = scale_fold(lstm_df[lstm_df['date'] < te_end].copy(), tr_end)
    X20, X60, y, tks, dts = make_seq(d)
    dts = pd.to_datetime(dts)
    trm = dts < pd.Timestamp(tr_end)
    tem = dts >= pd.Timestamp(tr_end)
    if tem.sum() < 50 or len(np.unique(y[trm])) < 2:
        return None, None

    tr_ns = dts[trm].astype(np.int64).values
    c1 = np.percentile(tr_ns, P_CORE)
    c2 = np.percentile(tr_ns, P_SEL)
    m_core = tr_ns <= c1
    m_es = (tr_ns > c1) & (tr_ns <= c2)
    m_sel = tr_ns > c2
    if m_es.sum() < 50 or m_sel.sum() < 50:
        return None, None

    model = LSTMClass(len(lfeat)).to(device)
    yc = y[trm][m_core]
    cw = compute_class_weight('balanced', classes=np.unique(yc), y=yc)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cw[1] / cw[0]], dtype=torch.float32).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', factor=0.5, patience=3, min_lr=1e-5)
    sc_amp = torch.amp.GradScaler('cuda', enabled=use_amp)

    tl = dl(X20[trm][m_core], X60[trm][m_core], y[trm][m_core], True)
    best, best_w, pat = 0, None, 0
    for ep in range(100):
        model.train()
        for a, b, c in tl:
            a, b, c = a.to(device), b.to(device), c.to(device)
            opt.zero_grad()
            with torch.autocast(device.type, enabled=use_amp):
                loss = crit(model(a, b), c)
            sc_amp.scale(loss).backward(); sc_amp.step(opt); sc_amp.update()
        vp = _infer(model, X20[trm][m_es], X60[trm][m_es], y[trm][m_es])
        va = roc_auc_score(y[trm][m_es], vp); sch.step(va)
        if va > best:
            best, best_w, pat = va, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 15:
                break
    model.load_state_dict(best_w)

    def _pack(mask_or_test, is_test=False):
        idx = tem if is_test else None
        if is_test:
            xs20, xs60, ys, tk, dt = X20[tem], X60[tem], y[tem], tks[tem], dts[tem]
        else:
            sub = np.zeros(len(y), bool); sub[np.where(trm)[0][mask_or_test]] = True
            xs20, xs60, ys, tk, dt = X20[sub], X60[sub], y[sub], tks[sub], dts[sub]
        return pd.DataFrame({'date': dt, 'ticker': tk,
                             'prob_lstm': _infer(model, xs20, xs60, ys), 'label_lstm': ys})

    return _pack(m_sel), _pack(None, is_test=True)


# ---------------------------------------------------
# 재정렬 적중률 계산
# ---------------------------------------------------
def rerank_hits(e, gmin, topn):
    """(적중 라벨 리스트, 일별 적중률 리스트)"""
    picks, daily = [], []
    for _, grp in e.groupby('date'):
        pool = grp[grp['prob_lgb'] >= gmin]
        cand = pool.sort_values('prob_lgb', ascending=False).head(topn)
        r3 = cand.sort_values('prob_lstm', ascending=False).head(3)['label'].tolist()
        if r3:
            picks.append(r3); daily.append(np.mean(r3))
    return picks, daily


def diff_block_boot_ci(a, b, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    D = len(a); idx = np.arange(D); out = []
    for _ in range(n):
        pick = rng.choice(idx, size=D, replace=True)
        e = np.mean([v for d_ in pick for v in a[d_]])
        g = np.mean([v for d_ in pick for v in b[d_]])
        out.append(e - g)
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def summarize(tag, rows, ens_days, gbm_days):
    hits = [r['hit'] for r in rows]; gs = [r['g'] for r in rows]; bs = [r['base'] for r in rows]
    mults = [h / b for h, b in zip(hits, bs) if b > 0]
    d_e = np.array([np.mean(p) for p in ens_days])
    d_g = np.array([np.mean(p) for p in gbm_days])
    diff = d_e - d_g
    t_stat, t_p = stats.ttest_rel(d_e, d_g)
    try:
        _, w_p = stats.wilcoxon(d_e, d_g)
    except ValueError:
        w_p = float('nan')
    lo, hi = diff_block_boot_ci(ens_days, gbm_days)
    print(f"\n----- {tag} -----")
    print(f"재정렬 {np.mean(hits):.4f} ({np.mean(mults):.2f}배, 편차 {np.std(mults):.3f})   "
          f"GBM단독 {np.mean(gs):.4f}   베이스 {np.mean(bs):.4f}")
    print(f"폴드별 재정렬: " + " / ".join(f"{r['hit']:.4f}" for r in rows))
    print(f"차이 평균 {diff.mean():+.4f} ({len(diff)}일)  t={t_stat:.3f} p={t_p:.4f}  "
          f"Wilcoxon p={w_p:.4f}  CI [{lo:.4f}, {hi:.4f}] "
          f"→ {'유의' if lo > 0 else '유의 X'}")


print(f"\n===== 분할: core {P_CORE}% / val_es {P_SEL-P_CORE}% / val_sel {100-P_SEL}% =====")
rows_sel, rows_fix = [], []
ens_sel, gbm_sel, ens_fix, gbm_fix = [], [], [], []

for tr_end, te_end in FOLDS:
    g_sel, g_te = gbm_fold(tr_end, te_end)
    l_sel, l_te = lstm_fold(tr_end, te_end)
    if g_sel is None or l_sel is None:
        print(f"{tr_end}~{te_end}  스킵"); continue

    e_sel = pd.merge(g_sel, l_sel, on=['date', 'ticker'], how='inner')
    e_te = pd.merge(g_te, l_te, on=['date', 'ticker'], how='inner')
    base = float(e_te['label'].mean())

    gbm_only = []
    for _, grp in e_te.groupby('date'):
        g3 = grp.sort_values('prob_lgb', ascending=False).head(3)['label'].tolist()
        if g3:
            gbm_only.append(g3)
    g_hit = float(np.mean([v for p in gbm_only for v in p]))

    # (A) val_sel 기반 선택
    best_cfg, best_score = (FIX_GMIN, FIX_TOPN), -1
    for gmin in GBM_MIN_GRID:
        for topn in TOP_N_GRID:
            _, daily = rerank_hits(e_sel, gmin, topn)
            if len(daily) < 10:
                continue
            s = float(np.mean(daily))
            if s > best_score:
                best_score, best_cfg = s, (gmin, topn)
    picks_s, _ = rerank_hits(e_te, *best_cfg)
    hit_s = float(np.mean([v for p in picks_s for v in p]))

    # (B) 사전 고정 설정
    picks_f, _ = rerank_hits(e_te, FIX_GMIN, FIX_TOPN)
    hit_f = float(np.mean([v for p in picks_f for v in p]))

    n_s = min(len(picks_s), len(gbm_only)); n_f = min(len(picks_f), len(gbm_only))
    ens_sel += picks_s[:n_s]; gbm_sel += gbm_only[:n_s]
    ens_fix += picks_f[:n_f]; gbm_fix += gbm_only[:n_f]
    rows_sel.append({'hit': hit_s, 'g': g_hit, 'base': base, 'cfg': best_cfg})
    rows_fix.append({'hit': hit_f, 'g': g_hit, 'base': base})

    print(f"{tr_end}~{te_end}  선택({best_cfg[0]:.2f},{best_cfg[1]}) test {hit_s:.4f}  |  "
          f"고정(0.50,8) test {hit_f:.4f}  |  GBM단독 {g_hit:.4f}  베이스 {base:.4f}")

summarize("A. val_sel 기반 선택 (폴드별 상이)", rows_sel, ens_sel, gbm_sel)
print(f"   선택된 설정: " + ", ".join(f"({r['cfg'][0]:.2f},{r['cfg'][1]})" for r in rows_sel))
summarize(f"B. 사전 고정 ({FIX_GMIN}, {FIX_TOPN}) — 선택 과정 없음", rows_fix, ens_fix, gbm_fix)
