# -*- coding: utf-8 -*-
"""
diagnostico_xgb.py
==================
Investiga la discrepancia entre F1=0.5109 (buscar_pesos_xgb.py)
y F1=0.5020 (02_entrenar_modelo.py) para XGBoost con los mismos params.

Hipotesis a verificar:
  1. Los params de config.py no coinciden con los del JSON (actualizacion parcial)
  2. El orden de features difiere entre los dos scripts
  3. El split de datos difiere (fillna, sort, etc.)
  4. n_jobs=-1 vs n_jobs=1 afecta al resultado (raro pero posible)
"""
import sys, warnings, json
import unittest.mock as mock

sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import (
    ARCHIVO_FEATURES, PARAMS_XGB, PESOS_XGB,
    FEATURES_SELECCIONADAS, RANDOM_SEED,
)
import utils

JSON_PATH = 'modelos/optuna_xgb_nuevas_feats.json'

# ============================================================================
# Cargar datos (identico a 02_entrenar_modelo.py)
# ============================================================================
print("Cargando datos...")
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

feats_pipeline = [f for f in FEATURES_SELECCIONADAS if f in df.columns]
X = df[feats_pipeline].fillna(0)
y = df['FTR_numeric']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
print(f"Train={len(X_train)} Test={len(X_test)} Feats={len(feats_pipeline)}")

# ============================================================================
# Cargar JSON con los mejores params
# ============================================================================
with open(JSON_PATH) as f:
    data = json.load(f)
best = data['best_params']
feats_json = data['features']

print("\n--- COMPARACION DE PARAMS ---")
print(f"JSON  : n_est={best['n_estimators']} depth={best['max_depth']} lr={best['learning_rate']:.5f} n_jobs=1")
print(f"Config: n_est={PARAMS_XGB['n_estimators']} depth={PARAMS_XGB['max_depth']} lr={PARAMS_XGB['learning_rate']:.5f} n_jobs={PARAMS_XGB['n_jobs']}")
print(f"JSON pesos : w0={best['w0']:.4f} w1={best['w1']:.4f} w2={best['w2']:.4f}")
print(f"Config pesos: w0={PESOS_XGB[0]:.4f} w1={PESOS_XGB[1]:.4f} w2={PESOS_XGB[2]:.4f}")
print(f"\nFeats pipeline == feats JSON: {feats_pipeline == feats_json}")
if feats_pipeline != feats_json:
    en_json_no_pipe = [f for f in feats_json if f not in feats_pipeline]
    en_pipe_no_json = [f for f in feats_pipeline if f not in feats_json]
    print(f"  En JSON pero no en pipeline: {en_json_no_pipe}")
    print(f"  En pipeline pero no en JSON: {en_pipe_no_json}")
    dif_orden = [(i, a, b) for i, (a, b) in enumerate(zip(feats_pipeline, feats_json)) if a != b]
    if dif_orden:
        print(f"  Diferencias de orden (idx, pipeline, json): {dif_orden[:5]}")

cw_json = {0: best['w0'], 1: best['w1'], 2: best['w2']}
cw_cfg  = PESOS_XGB

# ============================================================================
# EXPERIMENTO 1: params del JSON, features pipeline, n_jobs=-1
# ============================================================================
print("\n--- EXPERIMENTO 1: params JSON, feats pipeline, n_jobs=-1 ---")
sw = compute_sample_weight(cw_json, y_train)
m1 = XGBClassifier(
    n_estimators=best['n_estimators'], max_depth=best['max_depth'],
    learning_rate=best['learning_rate'], subsample=best['subsample'],
    colsample_bytree=best['colsample_bytree'], colsample_bylevel=best['colsample_bylevel'],
    reg_alpha=best['reg_alpha'], reg_lambda=best['reg_lambda'],
    min_child_weight=best['min_child_weight'], gamma=best['gamma'],
    random_state=RANDOM_SEED, n_jobs=-1, eval_metric='mlogloss', verbosity=0,
)
m1.fit(X_train, y_train, sample_weight=sw)
f1_e1 = f1_score(y_test, m1.predict(X_test), average='weighted')
print(f"F1 = {f1_e1:.4f}")

# ============================================================================
# EXPERIMENTO 2: params JSON, features JSON exactas (mismo orden), n_jobs=1
# ============================================================================
print("\n--- EXPERIMENTO 2: params JSON, feats JSON exactas, n_jobs=1 ---")
X_train_j = X_train[feats_json]
X_test_j  = X_test[feats_json]
sw = compute_sample_weight(cw_json, y_train)
m2 = XGBClassifier(
    n_estimators=best['n_estimators'], max_depth=best['max_depth'],
    learning_rate=best['learning_rate'], subsample=best['subsample'],
    colsample_bytree=best['colsample_bytree'], colsample_bylevel=best['colsample_bylevel'],
    reg_alpha=best['reg_alpha'], reg_lambda=best['reg_lambda'],
    min_child_weight=best['min_child_weight'], gamma=best['gamma'],
    random_state=RANDOM_SEED, n_jobs=1, eval_metric='mlogloss', verbosity=0,
)
m2.fit(X_train_j, y_train, sample_weight=sw)
f1_e2 = f1_score(y_test, m2.predict(X_test_j), average='weighted')
print(f"F1 = {f1_e2:.4f}")

# ============================================================================
# EXPERIMENTO 3: params config.py, features pipeline
# ============================================================================
print("\n--- EXPERIMENTO 3: params config.py, feats pipeline (lo que hace el pipeline) ---")
sw = compute_sample_weight(cw_cfg, y_train)
m3 = XGBClassifier(**PARAMS_XGB, verbosity=0)
m3.fit(X_train, y_train, sample_weight=sw)
f1_e3 = f1_score(y_test, m3.predict(X_test), average='weighted')
print(f"F1 = {f1_e3:.4f}")

# ============================================================================
# EXPERIMENTO 4: params JSON, n_jobs=1 (como en buscar_pesos_xgb.py)
# ============================================================================
print("\n--- EXPERIMENTO 4: params JSON, feats pipeline, n_jobs=1 (como buscar_pesos_xgb) ---")
sw = compute_sample_weight(cw_json, y_train)
m4 = XGBClassifier(
    n_estimators=best['n_estimators'], max_depth=best['max_depth'],
    learning_rate=best['learning_rate'], subsample=best['subsample'],
    colsample_bytree=best['colsample_bytree'], colsample_bylevel=best['colsample_bylevel'],
    reg_alpha=best['reg_alpha'], reg_lambda=best['reg_lambda'],
    min_child_weight=best['min_child_weight'], gamma=best['gamma'],
    random_state=RANDOM_SEED, n_jobs=1, eval_metric='mlogloss', verbosity=0,
)
m4.fit(X_train, y_train, sample_weight=sw)
f1_e4 = f1_score(y_test, m4.predict(X_test), average='weighted')
print(f"F1 = {f1_e4:.4f}")

# ============================================================================
# EXPERIMENTO 5: best model del JSON replicando buscar_pesos_xgb.py exactamente
# (train_test_split identico, mismas features, mismos pesos)
# ============================================================================
print("\n--- EXPERIMENTO 5: replica exacta de buscar_pesos_xgb.py ---")
# buscar_pesos_xgb.py carga features desde JSON, luego hace train_test_split
from config import ALL_FEATURES
feats_all = [f for f in ALL_FEATURES if f in df.columns]
X_all = df[feats_all].fillna(0)
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_all, y, test_size=0.20, shuffle=False
)
X_train_r = X_train_all[feats_json]
X_test_r  = X_test_all[feats_json]
sw = compute_sample_weight(cw_json, y_train_all)
m5 = XGBClassifier(
    n_estimators=best['n_estimators'], max_depth=best['max_depth'],
    learning_rate=best['learning_rate'], subsample=best['subsample'],
    colsample_bytree=best['colsample_bytree'], colsample_bylevel=best['colsample_bylevel'],
    reg_alpha=best['reg_alpha'], reg_lambda=best['reg_lambda'],
    min_child_weight=best['min_child_weight'], gamma=best['gamma'],
    random_state=RANDOM_SEED, n_jobs=1, eval_metric='mlogloss', verbosity=0,
)
m5.fit(X_train_r, y_train_all, sample_weight=sw)
f1_e5 = f1_score(y_test_all, m5.predict(X_test_r), average='weighted')
print(f"F1 = {f1_e5:.4f}  (esperado: 0.5109)")

# ============================================================================
# RESUMEN
# ============================================================================
print("\n" + "=" * 60)
print("  RESUMEN DE DISCREPANCIA")
print("=" * 60)
print(f"  E1 params JSON, feats pipeline, n_jobs=-1   : {f1_e1:.4f}")
print(f"  E2 params JSON, feats JSON, n_jobs=1        : {f1_e2:.4f}")
print(f"  E3 params config, feats pipeline (pipeline) : {f1_e3:.4f}")
print(f"  E4 params JSON, feats pipeline, n_jobs=1    : {f1_e4:.4f}")
print(f"  E5 replica exacta buscar_pesos_xgb.py       : {f1_e5:.4f}")
print(f"  Pipeline actual (config)                    : 0.5020")
print(f"  buscar_pesos_xgb.py reportado               : 0.5109")
print("=" * 60)
mejor = max(f1_e1, f1_e2, f1_e3, f1_e4, f1_e5)
print(f"  Mejor encontrado: {mejor:.4f}")
if mejor >= 0.5084:
    print("  -> Supera baseline 0.5084")
else:
    print("  -> No supera baseline con estos experimentos")
print("=" * 60)
