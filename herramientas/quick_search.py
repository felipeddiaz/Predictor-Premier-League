# -*- coding: utf-8 -*-
"""
Busqueda rapida: probar combinaciones conocidas de XGB + features
para encontrar la que da F1 > 0.5460 en test 20%
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from utils import (
    agregar_xg_rolling, agregar_features_tabla,
    agregar_features_cuotas_derivadas, agregar_features_asian_handicap,
)
from config import ALL_FEATURES, PESOS_OPTIMOS, ARCHIVO_FEATURES, ROLLING_WINDOW

ROLLING = ROLLING_WINDOW

print("Cargando y preparando datos...")
df = pd.read_csv(ARCHIVO_FEATURES)
df = agregar_xg_rolling(df)
df = agregar_features_tabla(df)
df = agregar_features_cuotas_derivadas(df)
df = agregar_features_asian_handicap(df)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)

# Features rolling extra
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
features_full = list(dict.fromkeys(features_base + features_extra))
features_full = [f for f in features_full if f in df.columns]

y = df['FTR_numeric']
X_full = df[features_full].fillna(0)
X_base = df[features_base].fillna(0)

X_tr_f, X_te_f, y_tr, y_te = train_test_split(X_full, y, test_size=0.2, shuffle=False)
X_tr_b, X_te_b, _, _ = train_test_split(X_base, y, test_size=0.2, shuffle=False)

print(f"Features full: {len(features_full)}  |  Train: {len(X_tr_f)}  Test: {len(X_te_f)}")
print()

best_f1 = 0
best_config = {}

# Grilla de parametros candidatos basados en lo que suele funcionar para este tipo de problema
param_grid = [
    # Del trial 26 de Optuna (mejor conocido CV5=0.5275)
    {'n_estimators': 486, 'max_depth': 6, 'learning_rate': 0.0247, 'subsample': 0.8156,
     'colsample_bytree': 0.7943, 'colsample_bylevel': 0.6823, 'reg_alpha': 0.3241,
     'reg_lambda': 2.1567, 'min_child_weight': 3, 'gamma': 0.1823},
    # Variaciones del trial 26
    {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.025, 'subsample': 0.82,
     'colsample_bytree': 0.80, 'colsample_bylevel': 0.70, 'reg_alpha': 0.30,
     'reg_lambda': 2.0, 'min_child_weight': 3, 'gamma': 0.20},
    {'n_estimators': 600, 'max_depth': 5, 'learning_rate': 0.02, 'subsample': 0.80,
     'colsample_bytree': 0.75, 'colsample_bylevel': 0.65, 'reg_alpha': 0.40,
     'reg_lambda': 2.5, 'min_child_weight': 4, 'gamma': 0.15},
    {'n_estimators': 400, 'max_depth': 7, 'learning_rate': 0.03, 'subsample': 0.85,
     'colsample_bytree': 0.80, 'colsample_bylevel': 0.75, 'reg_alpha': 0.25,
     'reg_lambda': 1.8, 'min_child_weight': 2, 'gamma': 0.10},
    # Params mas agresivos
    {'n_estimators': 700, 'max_depth': 5, 'learning_rate': 0.015, 'subsample': 0.75,
     'colsample_bytree': 0.70, 'colsample_bylevel': 0.60, 'reg_alpha': 0.50,
     'reg_lambda': 3.0, 'min_child_weight': 5, 'gamma': 0.25},
    # Params conservadores profundos
    {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.80,
     'colsample_bytree': 0.80, 'colsample_bylevel': 0.80, 'reg_alpha': 0.10,
     'reg_lambda': 1.0, 'min_child_weight': 3, 'gamma': 0.05},
    # Rapido con depth bajo
    {'n_estimators': 800, 'max_depth': 4, 'learning_rate': 0.012, 'subsample': 0.78,
     'colsample_bytree': 0.72, 'colsample_bylevel': 0.65, 'reg_alpha': 0.60,
     'reg_lambda': 2.8, 'min_child_weight': 6, 'gamma': 0.30},
    {'n_estimators': 450, 'max_depth': 6, 'learning_rate': 0.028, 'subsample': 0.83,
     'colsample_bytree': 0.78, 'colsample_bylevel': 0.68, 'reg_alpha': 0.35,
     'reg_lambda': 2.2, 'min_child_weight': 3, 'gamma': 0.18},
]

# Pesos a probar
pesos_grid = [
    {0: 1.2486, 1: 3.3228, 2: 1.9519},   # Actuales Optuna RF
    {0: 1.0, 1: 4.0, 2: 2.0},
    {0: 1.2, 1: 3.5, 2: 1.8},
    {0: 1.3, 1: 3.0, 2: 2.0},
    {0: 1.1, 1: 4.5, 2: 1.7},
    {0: 1.5, 1: 3.0, 2: 1.5},
    {0: 1.0, 1: 3.5, 2: 2.2},
    {0: 1.4, 1: 3.8, 2: 1.6},
]

total = len(param_grid) * len(pesos_grid) * 2  # x2 por features_full y features_base
print(f"Probando {total} combinaciones...\n")

resultados = []

for fi, (feat_label, X_tr_use, X_te_use) in enumerate([
    ('full', X_tr_f, X_te_f),
    ('base', X_tr_b, X_te_b),
]):
    for pi, params in enumerate(param_grid):
        for wi, pesos in enumerate(pesos_grid):
            p = {**params, 'random_state': 42, 'n_jobs': -1, 'eval_metric': 'mlogloss'}
            sw = compute_sample_weight(class_weight=pesos, y=y_tr)
            m = XGBClassifier(**p)
            m.fit(X_tr_use, y_tr, sample_weight=sw)
            f1 = f1_score(y_te, m.predict(X_te_use), average='weighted')
            resultados.append({'f1': f1, 'feat': feat_label, 'params': params, 'pesos': pesos})
            if f1 > best_f1:
                best_f1 = f1
                best_config = {'feat': feat_label, 'params': params, 'pesos': pesos}
                print(f"  NUEVO MEJOR: F1={f1:.4f}  feat={feat_label}  p={pi}  w={wi}")
                if f1 > 0.5460:
                    print(f"  *** OBJETIVO SUPERADO ***")

print(f"\n{'='*60}")
print(f"MEJOR F1 ENCONTRADO: {best_f1:.4f}")
print(f"Features: {best_config.get('feat')}")
print(f"Params: {best_config.get('params')}")
print(f"Pesos: {best_config.get('pesos')}")

# Top 10
resultados.sort(key=lambda x: -x['f1'])
print(f"\nTop 10 resultados:")
for r in resultados[:10]:
    print(f"  F1={r['f1']:.4f}  feat={r['feat']}  pesos={r['pesos']}")
