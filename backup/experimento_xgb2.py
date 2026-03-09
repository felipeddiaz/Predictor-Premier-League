# -*- coding: utf-8 -*-
"""
experimento_xgb2.py
- Diagnostico de varianza en test set
- Optuna sobre XGBoost con features base (que es lo que mejor funciona)
- Prueba con diferentes ventanas de test (ultimo 10%, 15%, 20%, 25%)
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
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from utils import (
    agregar_xg_rolling, agregar_features_tabla,
    agregar_features_cuotas_derivadas, agregar_features_asian_handicap,
)
from config import ALL_FEATURES, PARAMS_XGB, PESOS_OPTIMOS, ARCHIVO_FEATURES, ROLLING_WINDOW

ROLLING = ROLLING_WINDOW

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

features = [f for f in ALL_FEATURES if f in df.columns]
X = df[features].fillna(0)
y = df['FTR_numeric']

print(f"Features: {len(features)}  Partidos: {len(df)}")
print(f"Rango fechas: {df['Date'].min().date()} -> {df['Date'].max().date()}")

# ============================================================================
# DIAGNOSTICO: F1 en diferentes ventanas de test
# ============================================================================
print("\n" + "="*70)
print("DIAGNOSTICO: VARIANZA SEGUN VENTANA DE TEST")
print("="*70)

for test_pct in [0.10, 0.15, 0.20, 0.25, 0.30]:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_pct, shuffle=False)
    sw = compute_sample_weight(class_weight=PESOS_OPTIMOS, y=y_tr)
    m = XGBClassifier(**PARAMS_XGB)
    m.fit(X_tr, y_tr, sample_weight=sw)
    f1 = f1_score(y_te, m.predict(X_te), average='weighted')
    n_test = len(X_te)
    fecha_corte = df['Date'].iloc[len(X_tr)].date()
    print(f"  test={test_pct:.0%}  ({n_test} partidos desde {fecha_corte})  F1={f1:.4f}")

# ============================================================================
# DIAGNOSTICO: CV temporal con 3, 5, 7 folds
# ============================================================================
print("\n" + "="*70)
print("DIAGNOSTICO: CV TEMPORAL CON DISTINTOS FOLDS")
print("="*70)

from sklearn.model_selection import cross_val_score
for n_splits in [3, 5, 7]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    sw_full = compute_sample_weight(class_weight=PESOS_OPTIMOS, y=y)

    # XGB con CV manual (para pasar sample_weight por fold)
    scores = []
    for tr_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        sw_tr = compute_sample_weight(class_weight=PESOS_OPTIMOS, y=y_tr)
        m = XGBClassifier(**PARAMS_XGB)
        m.fit(X_tr, y_tr, sample_weight=sw_tr)
        scores.append(f1_score(y_val, m.predict(X_val), average='weighted'))
    print(f"  XGB CV{n_splits}: mean={np.mean(scores):.4f}  std={np.std(scores):.4f}  folds={scores}")

# ============================================================================
# OPTUNA: OPTIMIZAR XGB SOBRE FEATURES BASE
# ============================================================================
print("\n" + "="*70)
print("OPTUNA: OPTIMIZAR XGB (100 trials, CV5 temporal)")
print("="*70)

tscv5 = TimeSeriesSplit(n_splits=5)
X_tr_opt, X_te_opt, y_tr_opt, y_te_opt = train_test_split(X, y, test_size=0.2, shuffle=False)

def objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 100, 600),
        'max_depth':         trial.suggest_int('max_depth', 3, 9),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.20, log=True),
        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 0.0, 3.0),
        'reg_lambda':        trial.suggest_float('reg_lambda', 0.5, 4.0),
        'min_child_weight':  trial.suggest_int('min_child_weight', 1, 10),
        'gamma':             trial.suggest_float('gamma', 0.0, 1.0),
        'random_state': 42, 'n_jobs': -1, 'eval_metric': 'mlogloss',
    }
    # Pesos de clase como hiperparametros tambien
    w_local     = trial.suggest_float('peso_local',     0.5, 2.5)
    w_empate    = trial.suggest_float('peso_empate',    1.0, 5.0)
    w_visitante = trial.suggest_float('peso_visitante', 0.5, 2.5)

    fold_scores = []
    for tr_idx, val_idx in tscv5.split(X_tr_opt):
        X_f_tr, X_f_val = X_tr_opt.iloc[tr_idx], X_tr_opt.iloc[val_idx]
        y_f_tr, y_f_val = y_tr_opt.iloc[tr_idx], y_tr_opt.iloc[val_idx]
        sw_f = compute_sample_weight(
            class_weight={0: w_local, 1: w_empate, 2: w_visitante}, y=y_f_tr
        )
        m = XGBClassifier(**params)
        m.fit(X_f_tr, y_f_tr, sample_weight=sw_f)
        fold_scores.append(f1_score(y_f_val, m.predict(X_f_val), average='weighted'))
    return np.mean(fold_scores)

study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=100, show_progress_bar=True)

best = study.best_params
print(f"\nMejor CV F1: {study.best_value:.4f}")
print(f"Params: {best}")

# Evaluar en test
best_model_params = {k: v for k, v in best.items()
                     if k not in ['peso_local','peso_empate','peso_visitante']}
best_model_params.update({'random_state': 42, 'n_jobs': -1, 'eval_metric': 'mlogloss'})
sw_tr = compute_sample_weight(
    class_weight={0: best['peso_local'], 1: best['peso_empate'], 2: best['peso_visitante']},
    y=y_tr_opt
)
best_xgb = XGBClassifier(**best_model_params)
best_xgb.fit(X_tr_opt, y_tr_opt, sample_weight=sw_tr)
f1_final = f1_score(y_te_opt, best_xgb.predict(X_te_opt), average='weighted')

print(f"\nF1 en test set (20%): {f1_final:.4f}")
print(f"Objetivo: 0.5460  {'*** SUPERADO ***' if f1_final > 0.5460 else 'no superado'}")

# Feature importances
imp = sorted(zip(features, best_xgb.feature_importances_), key=lambda x: -x[1])
print("\nTop 15 features:")
for f, v in imp[:15]:
    print(f"  {f:<35} {v:.4f}")
print("\nBottom 10 features:")
for f, v in imp[-10:]:
    print(f"  {f:<35} {v:.4f}")

print("\n" + "="*70)
print("BLOQUE PARA COPIAR A config.py si mejora:")
print("="*70)
print(f"""
PESOS_OPTIMOS = {{
    0: {best['peso_local']:.4f},   # Local
    1: {best['peso_empate']:.4f},  # Empate
    2: {best['peso_visitante']:.4f}   # Visitante
}}

PARAMS_XGB = {{
    'n_estimators':     {best['n_estimators']},
    'max_depth':        {best['max_depth']},
    'learning_rate':    {best['learning_rate']:.4f},
    'subsample':        {best['subsample']:.4f},
    'colsample_bytree': {best['colsample_bytree']:.4f},
    'reg_alpha':        {best['reg_alpha']:.4f},
    'reg_lambda':       {best['reg_lambda']:.4f},
    'min_child_weight': {best['min_child_weight']},
    'gamma':            {best['gamma']:.4f},
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'mlogloss',
}}
""")
