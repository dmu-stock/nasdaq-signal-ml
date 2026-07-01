"""
모델 비교 Walk-Forward (논문 표용)
같은 나스닥 데이터·같은 폴드·같은 학습설정에서 인코더만 바꿔 AUC 비교:
  - Dual-LSTM (제안)
  - Single-LSTM (20일 단일 채널) → Dual 효과 검증
  - Dual-GRU (LSTM→GRU)          → LSTM vs GRU 검증

실행: python -m app.models.model_compare_wf
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random

from app.config.config import LSTM_FEATURE_COLS
from app.models.lstm_model import DualLSTMModel, SingleLSTMModel, DualGRUModel, DualTransformerModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = device.type == 'cuda'

CSV = "feature__indicator_lstm20260623.csv"
df = pd.read_csv(CSV)
df['date'] = pd.to_datetime(df['date'])
feat, tgt = LSTM_FEATURE_COLS, 'label'
SEQ20, SEQ60 = 20, 60

FOLDS = [
    ('2024-07-01', '2024-07-01', '2025-01-01'),
    ('2025-01-01', '2025-01-01', '2025-07-01'),
    ('2025-07-01', '2025-07-01', '2026-01-01'),
    ('2026-01-01', '2026-01-01', '2026-07-01'),
]

def scale_fold(d, train_end):
    d = d.sort_values(['ticker','date']).reset_index(drop=True).copy()
    d[feat] = d[feat].replace([np.inf,-np.inf], np.nan)
    d = d.dropna(subset=feat).reset_index(drop=True)
    tr = d['date'] < train_end
    for tk, _ in d.groupby('ticker'):
        m = d['ticker'] == tk
        if (m & tr).sum() == 0: continue
        sc = StandardScaler().fit(d.loc[m & tr, feat])
        d.loc[m, feat] = sc.transform(d.loc[m, feat])
    return d

def make_seq(d):
    X20, X60, y, dts = [], [], [], []
    for tk, g in d.groupby('ticker'):
        g = g.sort_values('date')
        if len(g) < SEQ60: continue
        f, t, dd = g[feat].values, g[tgt].values, g['date'].values
        for i in range(SEQ60-1, len(g)):
            X20.append(f[i-SEQ20+1:i+1]); X60.append(f[i-SEQ60+1:i+1])
            y.append(t[i]); dts.append(dd[i])
    return (np.array(X20,np.float32), np.array(X60,np.float32),
            np.array(y,np.float32), np.array(dts))

def loader(x20, x60, y, sh):
    ds = TensorDataset(torch.from_numpy(x20), torch.from_numpy(x60), torch.from_numpy(y))
    return DataLoader(ds, batch_size=128, shuffle=sh, pin_memory=use_amp)

def train_eval(ModelClass, tr_end, te_start, te_end, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    d = scale_fold(df[df['date'] < te_end].copy(), tr_end)
    X20, X60, y, dts = make_seq(d)
    dts = pd.to_datetime(dts)
    trm = dts < pd.Timestamp(tr_end)
    tem = (dts >= pd.Timestamp(te_start)) & (dts < pd.Timestamp(te_end))
    if tem.sum() < 50 or len(np.unique(y[tem])) < 2:
        return None
    tr_ns = dts[trm].astype(np.int64).values
    cut = np.percentile(tr_ns, 80)
    fin = tr_ns <= cut; vm = tr_ns > cut
    X20tr, X60tr, ytr = X20[trm][fin], X60[trm][fin], y[trm][fin]
    X20v, X60v, yv = X20[trm][vm], X60[trm][vm], y[trm][vm]

    model = ModelClass(len(feat)).to(device)
    cw = compute_class_weight('balanced', classes=np.unique(ytr), y=ytr)
    pw = torch.tensor([cw[1]/cw[0]], dtype=torch.float32).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', factor=0.5, patience=3, min_lr=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    tl = loader(X20tr, X60tr, ytr, True)
    vl = loader(X20v, X60v, yv, False)
    best, best_w, pat = 0, None, 0
    for ep in range(100):
        model.train()
        for a,b,c in tl:
            a,b,c = a.to(device),b.to(device),c.to(device)
            opt.zero_grad()
            with torch.autocast(device.type, enabled=use_amp):
                loss = crit(model(a,b), c)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        model.eval(); vp, vy = [], []
        with torch.no_grad():
            for a,b,c in vl:
                a,b = a.to(device),b.to(device)
                with torch.autocast(device.type, enabled=use_amp):
                    p = torch.sigmoid(model(a,b))
                vp += p.cpu().tolist(); vy += c.tolist()
        va = roc_auc_score(vy, vp); sched.step(va)
        if va > best: best, best_w, pat = va, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 15: break
    model.load_state_dict(best_w)

    model.eval(); tp = []
    teL = loader(X20[tem], X60[tem], y[tem], False)
    with torch.no_grad():
        for a,b,c in teL:
            a,b = a.to(device),b.to(device)
            with torch.autocast(device.type, enabled=use_amp):
                tp += torch.sigmoid(model(a,b)).cpu().tolist()
    return roc_auc_score(y[tem], tp)


MODELS = [
    ("Single-LSTM (20일)", SingleLSTMModel),
    ("Dual-GRU",           DualGRUModel),
    ("Dual-Transformer",   DualTransformerModel),
    ("Dual-LSTM (제안)",    DualLSTMModel),
]

print("===== 모델 비교 Walk-Forward (동일 데이터·폴드·설정) =====\n")
print(f"{'모델':<22}{'평균 AUC':<10}{'폴드별 AUC'}")
results = {}
for name, M in MODELS:
    aucs = []
    for tr_end, te_s, te_e in FOLDS:
        a = train_eval(M, tr_end, te_s, te_e)
        aucs.append(a if a is not None else float('nan'))
    results[name] = aucs
    folds_str = ' / '.join(f'{a:.4f}' for a in aucs)
    print(f"{name:<22}{np.nanmean(aucs):<10.4f}{folds_str}")

print("\n해석: Dual-LSTM이 가장 높으면 '왜 Dual-LSTM'을 성능으로 증명")
