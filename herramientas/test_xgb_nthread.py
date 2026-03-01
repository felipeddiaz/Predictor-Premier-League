# -*- coding: utf-8 -*-
"""
test_xgb_nthread.py
===================
Investiga si XGBoost con nthread=1 da resultado consistente
despues de entrenar RF en memoria (sin depender de estado de threads).
"""
import sys, warnings, json
import unittest.mock as mock

sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import (
    ARCHIVO_FEATURES, PARAMS_XGB, PESOS_XGB,
    PARAMS_OPTIMOS, PESOS_OPTIMOS,
    FEATURES_SELECCIONADAS, RANDOM_SEED,
)
import utils

# Cargar datos
df = pd.read_csv(ARCHIVO_FEATURES)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
with mock.patch('builtins.print'):
    df = utils.agregar_xg_rolling(df)
    df = utils.agregar_features_tabla(df)
    df = utils.agregar_features_cuotas_derivadas(df)
    df = utils.agregar_features_asian_handicap(df)
    df = utils.agregar_features_rolling_extra(df)
    df = utils.agregar_features_forma_momentum(df)
    df = utils.agregar_features_pinnacle_move(df)
    df = utils.agregar_features_arbitro(df)

feats = [f for f in FEATURES_SELECCIONADAS if f in df.columns]
X = df[feats].fillna(0)
y = df['FTR_numeric']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

with open('modelos/optuna_xgb_nuevas_feats.json') as f:
    best = json.load(f)['best_params']
cw_xgb = {0: best['w0'], 1: best['w1'], 2: best['w2']}

print(f"Train={len(X_train)} Test={len(X_test)} Feats={len(feats)}")

# RF Optuna primero (igual que el pipeline)
rf = RandomForestClassifier(**PARAMS_OPTIMOS)
rf.fit(X_train, y_train)
f1_rf = f1_score(y_test, rf.predict(X_test), average='weighted')
print(f"RF Optuna: {f1_rf:.4f}")

del rf  # liberar memoria

# XGB con nthread=1 (determinista, sin threading externo)
params_1t = {k: v for k, v in PARAMS_XGB.items() if k != 'n_jobs'}
params_1t['nthread'] = 1
sw = compute_sample_weight(cw_xgb, y_train)
xgb_1t = XGBClassifier(**params_1t, verbosity=0)
xgb_1t.fit(X_train, y_train, sample_weight=sw)
f1_1t = f1_score(y_test, xgb_1t.predict(X_test), average='weighted')
print(f"XGB nthread=1: {f1_1t:.4f}")

# XGB con n_jobs=-1 despues del RF
sw = compute_sample_weight(cw_xgb, y_train)
xgb_mt = XGBClassifier(**PARAMS_XGB, verbosity=0)
xgb_mt.fit(X_train, y_train, sample_weight=sw)
f1_mt = f1_score(y_test, xgb_mt.predict(X_test), average='weighted')
print(f"XGB n_jobs=-1: {f1_mt:.4f}")

print("\n--- RESUMEN ---")
print(f"RF Optuna      : {f1_rf:.4f}")
print(f"XGB nthread=1  : {f1_1t:.4f}")
print(f"XGB n_jobs=-1  : {f1_mt:.4f}")
print(f"Baseline       : 0.5084")
print(f"Mejor          : {max(f1_rf, f1_1t, f1_mt):.4f}")
