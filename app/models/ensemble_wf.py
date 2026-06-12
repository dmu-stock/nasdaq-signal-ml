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

from app.config.config import GBM_FEATURE_COLS, LSTM_FEATURE_COLS
from app.models.lstm_model import DualLSTMModel
from lightgbm import LGBMClassifier, early_stopping
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = device.type == 'cuda'

# 최신 CSV 파일명으로 맞추세요
GBM_CSV  = "feature__indicator_20260608.csv"
LSTM_CSV = "feature__indicator_lstm20260608.csv"

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

    out = te[['date', 'ticker', 'label']].copy()
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

    model = DualLSTMModel(len(lfeat)).to(device)
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
    return pd.DataFrame({'date': dts[tem], 'ticker': tks[tem], 'prob_lstm': tp})


# ---------------------------------------------------
# 폴드 루프
# ---------------------------------------------------
print("===== 앙상블 Walk-Forward =====")
print(f"가드레일: LGBM >= {LGBM_TH}, LSTM >= {LSTM_TH}\n")
rows = []
for tr_end, te_end in FOLDS:
    g = gbm_fold(tr_end, te_end)
    l = lstm_fold(tr_end, te_end)
    if g is None or l is None:
        print(f"{tr_end} ~ {te_end}  스킵"); continue

    e = pd.merge(g, l, on=['date', 'ticker'], how='inner')
    e['final'] = 2*(e['prob_lgb']*e['prob_lstm']) / (e['prob_lgb']+e['prob_lstm']+1e-9)
    base = e['label'].mean()

    top3 = []
    for date, grp in e.groupby('date'):
        f = grp[(grp['prob_lgb'] >= LGBM_TH) & (grp['prob_lstm'] >= LSTM_TH)]
        f = f.sort_values('final', ascending=False).head(3)
        if not f.empty:
            top3 += f['label'].tolist()
    hit = np.mean(top3) if top3 else 0.0
    mult = hit/base if base > 0 else 0
    rows.append((f'{tr_end}~{te_end}', hit, len(top3), base, mult))
    print(f"{tr_end} ~ {te_end}  Top3={hit:.4f}  ({len(top3)}회)  베이스{base:.4f}  → {mult:.2f}배")

if rows:
    hits  = [r[1] for r in rows]
    mults = [r[4] for r in rows]
    print(f"\n평균 Top3 타율: {np.mean(hits):.4f}")
    print(f"평균 배수: {np.mean(mults):.2f}배  (편차 {np.std(mults):.3f})")
