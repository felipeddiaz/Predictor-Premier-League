# -*- coding: utf-8 -*-
"""
experimento_xgb3.py
Optuna 200 trials sobre XGB con features base — guardar mejor resultado
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from optuna.samplers import TPESampler

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from utils import (
    agregar_xg_rolling, agregar_features_tabla,
    agregar_features_cuotas_derivadas, agregar_features_asian_handicap,
)
from config import ALL_FEATURES, PESOS_OPTIMOS, ARCHIVO_FEATURES, ROLLING_WINDOW

# ============================================================================
# CARGAR DATOS
# ============================================================================
print("Cargando datos...")
df = pd.read_csv(ARCHIVO_FEATURES)
df = agregar_xg_rolling(df)
df = agregar_features_tabla(df)
df = agregar_features_cuotas_derivadas(df)
df = agregar_features_asian_handicap(df)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)

ROLLING = ROLLING_WINDOW

# Agregar features nuevas que demostraron aportar en experimento anterior
for col in ['HT_Goals_Diff','AT_Goals_Diff','AT_HTR_Rate']:
    df[col] = np.nan

for team in df['HomeTeam'].unique():
    idx_h = df[df['HomeTeam'] == team].sort_values('Date').index.tolist()
    idx_a = df[df['AwayTeam'] == team].sort_values('Date').index.tolist()
    gd_h, gd_a, htr_a = [], [], []

    for i in idx_h:
        if gd_h: df.at[i, 'HT_Goals_Diff'] = np.mean(gd_h[-ROLLING:])
        gh = df.at[i, 'FTHG'] if pd.notna(df.at[i, 'FTHG']) else 0
        ga = df.at[i, 'FTAG'] if pd.notna(df.at[i, 'FTAG']) else 0
        gd_h.append(gh - ga)

    for i in idx_a:
        if gd_a:  df.at[i, 'AT_Goals_Diff'] = np.mean(gd_a[-ROLLING:])
        if htr_a: df.at[i, 'AT_HTR_Rate']   = np.mean(htr_a[-ROLLING:])
        gh   = df.at[i, 'FTHG'] if pd.notna(df.at[i, 'FTHG']) else 0
        ga   = df.at[i, 'FTAG'] if pd.notna(df.at[i, 'FTAG']) else 0
        gd_a.append(ga - gh)
        hthg = df.at[i, 'HTHG'] if pd.notna(df.at[i, 'HTHG']) else 0
        htag = df.at[i, 'HTAG'] if pd.notna(df.at[i, 'HTAG']) else 0
        htr_a.append(1 if htag > hthg else 0)

for col in ['HT_Goals_Diff','AT_Goals_Diff','AT_HTR_Rate']:
    df[col] = df[col].fillna(0)

# Pinnacle vs avg (sharp money)
df['PS_vs_Avg_H'] = (df['PSCH'] - df['AvgCH']).fillna(0)

# Features finales
features_base = [f for f in ALL_FEATURES if f in df.columns]
features_extra = ['HT_Goals_Diff', 'AT_Goals_Diff', 'AT_HTR_Rate', 'PS_vs_Avg_H']
features = list(dict.fromkeys(features_base + features_extra))
features = [f for f in features if f in df.columns]

X = df[features].fillna(0)
y = df['FTR_numeric']

# Split: 80/20 temporal
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
tscv5 = TimeSeriesSplit(n_splits=5)

print(f"Features: {len(features)}  Train: {len(X_tr)}  Test: {len(X_te)}")

# ============================================================================
# OPTUNA 200 TRIALS
# ============================================================================
print("\nOptuna 200 trials (puede tardar ~15 min)...")

def objective(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 100, 800),
        'max_depth':        trial.suggest_int('max_depth', 3, 10),
        'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.20, log=True),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'colsample_bylevel':trial.suggest_float('colsample_bylevel', 0.4, 1.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda':       trial.suggest_float('reg_lambda', 0.1, 5.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'gamma':            trial.suggest_float('gamma', 0.0, 2.0),
        'random_state': 42, 'n_jobs': -1, 'eval_metric': 'mlogloss',
    }
    w_local     = trial.suggest_float('peso_local',     0.5, 3.0)
    w_empate    = trial.suggest_float('peso_empate',    1.0, 6.0)
    w_visitante = trial.suggest_float('peso_visitante', 0.5, 3.0)

    fold_scores = []
    for tr_idx, val_idx in tscv5.split(X_tr):
        Xf_tr, Xf_val = X_tr.iloc[tr_idx], X_tr.iloc[val_idx]
        yf_tr, yf_val = y_tr.iloc[tr_idx], y_tr.iloc[val_idx]
        sw = compute_sample_weight(
            class_weight={0: w_local, 1: w_empate, 2: w_visitante}, y=yf_tr
        )
        m = XGBClassifier(**params)
        m.fit(Xf_tr, yf_tr, sample_weight=sw)
        fold_scores.append(f1_score(yf_val, m.predict(Xf_val), average='weighted'))
    return np.mean(fold_scores)

study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=200, show_progress_bar=True)

best = study.best_params
print(f"\nMejor CV5 F1: {study.best_value:.4f}")

# ============================================================================
# EVALUAR MEJOR MODELO EN TEST
# ============================================================================
best_model_params = {k: v for k, v in best.items()
                     if k not in ['peso_local','peso_empate','peso_visitante']}
best_model_params.update({'random_state': 42, 'n_jobs': -1, 'eval_metric': 'mlogloss'})

sw_tr = compute_sample_weight(
    class_weight={0: best['peso_local'], 1: best['peso_empate'], 2: best['peso_visitante']},
    y=y_tr
)
best_xgb = XGBClassifier(**best_model_params)
best_xgb.fit(X_tr, y_tr, sample_weight=sw_tr)
pred = best_xgb.predict(X_te)

f1_test  = f1_score(y_te, pred, average='weighted')
acc_test = accuracy_score(y_te, pred)
cm       = confusion_matrix(y_te, pred)

print(f"\nF1  test (20%):  {f1_test:.4f}")
print(f"Acc test (20%):  {acc_test:.4f}")
print(f"Objetivo: 0.5460  {'*** SUPERADO ***' if f1_test > 0.5460 else 'no superado (pero revisa CV)'}")
print(f"\nMatriz de confusion:")
print(f"             Local  Empate  Visit")
print(f"Real Local   {cm[0,0]:>5}  {cm[0,1]:>6}  {cm[0,2]:>5}")
print(f"Real Empate  {cm[1,0]:>5}  {cm[1,1]:>6}  {cm[1,2]:>5}")
print(f"Real Visit   {cm[2,0]:>5}  {cm[2,1]:>6}  {cm[2,2]:>5}")

r_l = cm[0,0]/cm[0].sum(); r_e = cm[1,1]/cm[1].sum(); r_v = cm[2,2]/cm[2].sum()
print(f"\nRecall  Local={r_l:.2%}  Empate={r_e:.2%}  Visitante={r_v:.2%}")

# Top features
imp = sorted(zip(features, best_xgb.feature_importances_), key=lambda x: -x[1])
print("\nTop 15 features:")
for f, v in imp[:15]:
    print(f"  {f:<35} {v:.4f}")

# ============================================================================
# COMPARAR CON DISTINTOS TEST SIZES
# ============================================================================
print("\n" + "="*70)
print("F1 POR VENTANA DE TEST CON EL MODELO OPTIMIZADO")
print("="*70)
for test_pct in [0.10, 0.15, 0.20, 0.25]:
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y, test_size=test_pct, shuffle=False)
    sw2 = compute_sample_weight(
        class_weight={0: best['peso_local'], 1: best['peso_empate'], 2: best['peso_visitante']},
        y=y_tr2
    )
    m2 = XGBClassifier(**best_model_params)
    m2.fit(X_tr2, y_tr2, sample_weight=sw2)
    f1_2 = f1_score(y_te2, m2.predict(X_te2), average='weighted')
    fecha_corte = df['Date'].iloc[len(X_tr2)].date()
    print(f"  test={test_pct:.0%} ({len(X_te2)} partidos desde {fecha_corte})  F1={f1_2:.4f}  {'OK' if f1_2>0.5460 else ''}")

# ============================================================================
# BLOQUE PARA COPIAR A config.py
# ============================================================================
print("\n" + "="*70)
print("COPIAR A config.py SI LOS RESULTADOS SON BUENOS:")
print("="*70)
print(f"""
PESOS_OPTIMOS = {{
    0: {best['peso_local']:.4f},   # Local
    1: {best['peso_empate']:.4f},  # Empate
    2: {best['peso_visitante']:.4f}   # Visitante
}}

PARAMS_XGB = {{
    'n_estimators':      {best['n_estimators']},
    'max_depth':         {best['max_depth']},
    'learning_rate':     {best['learning_rate']:.5f},
    'subsample':         {best['subsample']:.4f},
    'colsample_bytree':  {best['colsample_bytree']:.4f},
    'colsample_bylevel': {best['colsample_bylevel']:.4f},
    'reg_alpha':         {best['reg_alpha']:.4f},
    'reg_lambda':        {best['reg_lambda']:.4f},
    'min_child_weight':  {best['min_child_weight']},
    'gamma':             {best['gamma']:.4f},
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'mlogloss',
}}
""")
print(f"Features nuevas a agregar a config.py + utils.py: {features_extra}")
print(f"CV5 optimo: {study.best_value:.4f}  |  F1 test 20%: {f1_test:.4f}")
