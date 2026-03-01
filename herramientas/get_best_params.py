# -*- coding: utf-8 -*-
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
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from utils import (
    agregar_xg_rolling, agregar_features_tabla,
    agregar_features_cuotas_derivadas, agregar_features_asian_handicap,
)
from config import ALL_FEATURES, PESOS_OPTIMOS, ARCHIVO_FEATURES, ROLLING_WINDOW

ROLLING = ROLLING_WINDOW

print("Cargando datos...")
df = pd.read_csv(ARCHIVO_FEATURES)
df = agregar_xg_rolling(df)
df = agregar_features_tabla(df)
df = agregar_features_cuotas_derivadas(df)
df = agregar_features_asian_handicap(df)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)

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

df['PS_vs_Avg_H'] = (df['PSCH'] - df['AvgCH']).fillna(0)

features_base = [f for f in ALL_FEATURES if f in df.columns]
features_extra = ['HT_Goals_Diff', 'AT_Goals_Diff', 'AT_HTR_Rate', 'PS_vs_Avg_H']
features = list(dict.fromkeys(features_base + features_extra))
features = [f for f in features if f in df.columns]

X = df[features].fillna(0)
y = df['FTR_numeric']

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
tscv5 = TimeSeriesSplit(n_splits=5)

print(f"Features: {len(features)}  Train: {len(X_tr)}  Test: {len(X_te)}")

# Correr solo 50 trials enfocados (seed=42 reproduce los mismos primeros trials)
print("\nOptuna 50 trials rapido...")

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
study.optimize(objective, n_trials=50, show_progress_bar=False)

best = study.best_params
print(f"Mejor CV5 F1: {study.best_value:.4f}")

# Evaluar en test
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
f1_test = f1_score(y_te, pred, average='weighted')

print(f"F1 test 20%: {f1_test:.4f}  {'*** SUPERADO ***' if f1_test > 0.5460 else 'no superado aun'}")
print(f"\n=== MEJORES PARAMS (trial {study.best_trial.number}) ===")
print(f"PESOS_OPTIMOS = {{")
print(f"    0: {best['peso_local']:.4f},   # Local")
print(f"    1: {best['peso_empate']:.4f},  # Empate")
print(f"    2: {best['peso_visitante']:.4f}   # Visitante")
print(f"}}")
print(f"")
print(f"PARAMS_XGB = {{")
print(f"    'n_estimators':      {best['n_estimators']},")
print(f"    'max_depth':         {best['max_depth']},")
print(f"    'learning_rate':     {best['learning_rate']:.5f},")
print(f"    'subsample':         {best['subsample']:.4f},")
print(f"    'colsample_bytree':  {best['colsample_bytree']:.4f},")
print(f"    'colsample_bylevel': {best['colsample_bylevel']:.4f},")
print(f"    'reg_alpha':         {best['reg_alpha']:.4f},")
print(f"    'reg_lambda':        {best['reg_lambda']:.4f},")
print(f"    'min_child_weight':  {best['min_child_weight']},")
print(f"    'gamma':             {best['gamma']:.4f},")
print(f"    'random_state': 42,")
print(f"    'n_jobs': -1,")
print(f"    'eval_metric': 'mlogloss',")
print(f"}}")
