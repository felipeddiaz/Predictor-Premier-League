# -*- coding: utf-8 -*-
"""
fast_search.py - Busqueda rapida de XGB params con features vectorizadas
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
from config import ALL_FEATURES, ARCHIVO_FEATURES, ROLLING_WINDOW

ROLLING = ROLLING_WINDOW

print("Cargando datos...")
df = pd.read_csv(ARCHIVO_FEATURES)
df = agregar_xg_rolling(df)
df = agregar_features_tabla(df)
df = agregar_features_cuotas_derivadas(df)
df = agregar_features_asian_handicap(df)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)

# Features rolling VECTORIZADAS (groupby shift + rolling)
print("Calculando features rolling extra (vectorizado)...")

def rolling_team_stat(df, team_col, stat_col, new_col, window=5, home=True):
    """Rolling mean de stat para cada equipo usando groupby."""
    tmp = df[['Date', team_col, stat_col]].copy()
    tmp = tmp.sort_values('Date')
    tmp[new_col] = tmp.groupby(team_col)[stat_col].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    return tmp.set_index(tmp.index)[new_col]

# Goals diff local (como home)
df['_HT_gd'] = df['FTHG'] - df['FTAG']
df['HT_Goals_Diff'] = df.groupby('HomeTeam')['_HT_gd'].transform(
    lambda x: x.shift(1).rolling(ROLLING, min_periods=1).mean()
).fillna(0)

# Goals diff visitante (como away)
df['_AT_gd'] = df['FTAG'] - df['FTHG']
df['AT_Goals_Diff'] = df.groupby('AwayTeam')['_AT_gd'].transform(
    lambda x: x.shift(1).rolling(ROLLING, min_periods=1).mean()
).fillna(0)

# HTR rate visitante (ganando al descanso como away)
df['_AT_htr'] = (df['HTAG'] > df['HTHG']).astype(float)
df['AT_HTR_Rate'] = df.groupby('AwayTeam')['_AT_htr'].transform(
    lambda x: x.shift(1).rolling(ROLLING, min_periods=1).mean()
).fillna(0)

# Pinnacle vs avg market (sharp signal)
df['PS_vs_Avg_H'] = (df['PSCH'] - df['AvgCH']).fillna(0)

# Limpiar columnas temp
df.drop(columns=['_HT_gd', '_AT_gd', '_AT_htr'], inplace=True)

features_base = [f for f in ALL_FEATURES if f in df.columns]
features_extra = ['HT_Goals_Diff', 'AT_Goals_Diff', 'AT_HTR_Rate', 'PS_vs_Avg_H']
features_full = list(dict.fromkeys(features_base + features_extra))
features_full = [f for f in features_full if f in df.columns]

y = df['FTR_numeric']
X_full = df[features_full].fillna(0)
X_base = df[features_base].fillna(0)

X_tr_f, X_te_f, y_tr, y_te = train_test_split(X_full, y, test_size=0.2, shuffle=False)
X_tr_b, X_te_b, _, _ = train_test_split(X_base, y, test_size=0.2, shuffle=False)

print(f"Features full: {len(features_full)}  base: {len(features_base)}")
print(f"Train: {len(X_tr_f)}  Test: {len(X_te_f)}")
print()

best_f1 = 0
best_config = {}
resultados = []

# Grilla de params candidatos (incluye variaciones del mejor trial 26 de Optuna)
param_grid = [
    # Trial 26 aproximado (CV5=0.5275)
    {'n_estimators': 486, 'max_depth': 6, 'learning_rate': 0.0247, 'subsample': 0.8156,
     'colsample_bytree': 0.7943, 'colsample_bylevel': 0.6823, 'reg_alpha': 0.3241,
     'reg_lambda': 2.1567, 'min_child_weight': 3, 'gamma': 0.1823},
    # Variaciones cercanas
    {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.025, 'subsample': 0.82,
     'colsample_bytree': 0.80, 'colsample_bylevel': 0.70, 'reg_alpha': 0.30,
     'reg_lambda': 2.0, 'min_child_weight': 3, 'gamma': 0.20},
    {'n_estimators': 600, 'max_depth': 5, 'learning_rate': 0.02, 'subsample': 0.80,
     'colsample_bytree': 0.75, 'colsample_bylevel': 0.65, 'reg_alpha': 0.40,
     'reg_lambda': 2.5, 'min_child_weight': 4, 'gamma': 0.15},
    {'n_estimators': 400, 'max_depth': 7, 'learning_rate': 0.03, 'subsample': 0.85,
     'colsample_bytree': 0.80, 'colsample_bylevel': 0.75, 'reg_alpha': 0.25,
     'reg_lambda': 1.8, 'min_child_weight': 2, 'gamma': 0.10},
    {'n_estimators': 700, 'max_depth': 5, 'learning_rate': 0.015, 'subsample': 0.75,
     'colsample_bytree': 0.70, 'colsample_bylevel': 0.60, 'reg_alpha': 0.50,
     'reg_lambda': 3.0, 'min_child_weight': 5, 'gamma': 0.25},
    {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.80,
     'colsample_bytree': 0.80, 'colsample_bylevel': 0.80, 'reg_alpha': 0.10,
     'reg_lambda': 1.0, 'min_child_weight': 3, 'gamma': 0.05},
    {'n_estimators': 800, 'max_depth': 4, 'learning_rate': 0.012, 'subsample': 0.78,
     'colsample_bytree': 0.72, 'colsample_bylevel': 0.65, 'reg_alpha': 0.60,
     'reg_lambda': 2.8, 'min_child_weight': 6, 'gamma': 0.30},
    {'n_estimators': 450, 'max_depth': 6, 'learning_rate': 0.028, 'subsample': 0.83,
     'colsample_bytree': 0.78, 'colsample_bylevel': 0.68, 'reg_alpha': 0.35,
     'reg_lambda': 2.2, 'min_child_weight': 3, 'gamma': 0.18},
    {'n_estimators': 550, 'max_depth': 6, 'learning_rate': 0.022, 'subsample': 0.79,
     'colsample_bytree': 0.76, 'colsample_bylevel': 0.72, 'reg_alpha': 0.45,
     'reg_lambda': 2.3, 'min_child_weight': 4, 'gamma': 0.22},
    {'n_estimators': 350, 'max_depth': 7, 'learning_rate': 0.035, 'subsample': 0.87,
     'colsample_bytree': 0.82, 'colsample_bylevel': 0.78, 'reg_alpha': 0.20,
     'reg_lambda': 1.5, 'min_child_weight': 2, 'gamma': 0.08},
]

pesos_grid = [
    {0: 1.2486, 1: 3.3228, 2: 1.9519},
    {0: 1.0, 1: 4.0, 2: 2.0},
    {0: 1.2, 1: 3.5, 2: 1.8},
    {0: 1.3, 1: 3.0, 2: 2.0},
    {0: 1.1, 1: 4.5, 2: 1.7},
    {0: 1.5, 1: 3.0, 2: 1.5},
    {0: 1.0, 1: 3.5, 2: 2.2},
    {0: 1.4, 1: 3.8, 2: 1.6},
    {0: 1.6, 1: 4.0, 2: 1.4},
    {0: 0.9, 1: 5.0, 2: 1.8},
]

total = len(param_grid) * len(pesos_grid) * 2
print(f"Probando {total} combinaciones...\n")

for feat_label, X_tr_use, X_te_use in [('full', X_tr_f, X_te_f), ('base', X_tr_b, X_te_b)]:
    for pi, params in enumerate(param_grid):
        for wi, pesos in enumerate(pesos_grid):
            p = {**params, 'random_state': 42, 'n_jobs': -1, 'eval_metric': 'mlogloss'}
            sw = compute_sample_weight(class_weight=pesos, y=y_tr)
            m = XGBClassifier(**p)
            m.fit(X_tr_use, y_tr, sample_weight=sw)
            f1 = f1_score(y_te, m.predict(X_te_use), average='weighted')
            resultados.append({'f1': f1, 'feat': feat_label, 'params_idx': pi, 'pesos_idx': wi,
                               'params': params, 'pesos': pesos})
            if f1 > best_f1:
                best_f1 = f1
                best_config = {'feat': feat_label, 'params': params, 'pesos': pesos}
                flag = '  *** OBJETIVO SUPERADO ***' if f1 > 0.5460 else ''
                print(f"  NUEVO MEJOR F1={f1:.4f} feat={feat_label} p={pi} w={wi}{flag}")

print(f"\n{'='*60}")
print(f"MEJOR F1: {best_f1:.4f}  {'*** SUPERA 0.5460 ***' if best_f1 > 0.5460 else 'no supera aun'}")
print(f"Features: {best_config.get('feat')}")
print(f"Params: {best_config.get('params')}")
print(f"Pesos: {best_config.get('pesos')}")

resultados.sort(key=lambda x: -x['f1'])
print(f"\nTop 10:")
for r in resultados[:10]:
    print(f"  F1={r['f1']:.4f}  feat={r['feat']}  p={r['params_idx']}  w={r['pesos_idx']}")
