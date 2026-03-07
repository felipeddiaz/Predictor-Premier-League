# -*- coding: utf-8 -*-
"""
Optimización de pesos e hiperparámetros con Optuna.

Fase 3 del audit ML:
  - Métrica: Log Loss (minimizar) + penalización de varianza entre folds
    score = mean(log_loss_por_fold) + 0.5 * std(log_loss_por_fold)
  - MedianPruner para explorar 500+ trials eficientemente
  - XGBoost con early_stopping_rounds=50 dentro de cada fold CV
  - Rangos corregidos: min_child_weight [5,30], learning_rate [0.003,0.15]
  - Co-optimización de pesos de clase

Modos de ejecución (flags principales abajo):
  MODO_SIN_CUOTAS=False, MODO_XGB=False → RF  con cuotas (modelo 02)
  MODO_SIN_CUOTAS=True,  MODO_XGB=False → RF  sin cuotas (modelo 03)
  MODO_SIN_CUOTAS=False, MODO_XGB=True  → XGB con cuotas (modelo 02)
  MODO_SIN_CUOTAS=True,  MODO_XGB=True  → XGB sin cuotas (modelo 03)

Al terminar imprime el bloque listo para copiar a config.py.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import unittest.mock as mock
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, brier_score_loss, confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from config import (
    ARCHIVO_FEATURES,
    ALL_FEATURES,
    FEATURES_ESTRUCTURALES,
    FEATURES_BASE,
    FEATURES_CUOTAS,
    FEATURES_CUOTAS_DERIVADAS,
    FEATURES_XG,
    FEATURES_XG_GLOBAL,
    FEATURES_MULTI_ESCALA,
    FEATURES_H2H,
    FEATURES_H2H_DERIVADAS,
    FEATURES_TABLA,
    FEATURES_ASIAN_HANDICAP,
    FEATURES_ROLLING_EXTRA,
    FEATURES_PINNACLE,
    FEATURES_REFEREE,
    FEATURES_FORMA_MOMENTUM,
    FEATURES_DESCANSO,
    RANDOM_SEED,
)
import utils

# ============================================================================
# FLAGS PRINCIPALES — cambia aquí antes de ejecutar
# ============================================================================

MODO_SIN_CUOTAS = False   # False = datos CON cuotas  |  True = datos SIN cuotas
MODO_XGB = True           # False = Random Forest     |  True = XGBoost

N_TRIALS = 200            # MedianPruner permite más trials sin costo extra
N_SPLITS_CV = 5           # Folds para TimeSeriesSplit

# ============================================================================
# CARGAR Y PREPARAR DATOS
# ============================================================================

if MODO_XGB and MODO_SIN_CUOTAS:
    modo_label = "XGBOOST SIN CUOTAS (modelo 03)"
elif MODO_XGB:
    modo_label = "XGBOOST CON CUOTAS (modelo 02)"
elif MODO_SIN_CUOTAS:
    modo_label = "RF SIN CUOTAS (modelo 03)"
else:
    modo_label = "RF CON CUOTAS (modelo 02)"

print("=" * 70)
print(f"OPTIMIZACION DE HIPERPARAMETROS CON OPTUNA")
print(f"Modo: {modo_label}")
print(f"Metrica: Log Loss (minimizar) + 0.5*STD penalty")
print(f"Pruner: MedianPruner | Trials: {N_TRIALS} | CV: {N_SPLITS_CV}-fold TimeSeriesSplit")
print("=" * 70)

df = pd.read_csv(ARCHIVO_FEATURES)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
print(f"\n   Cargados: {len(df)} partidos")

# Features calculadas en memoria
with mock.patch('builtins.print'):
    df = utils.agregar_xg_rolling(df)
    df = utils.agregar_features_tabla(df)
    if not MODO_SIN_CUOTAS:
        df = utils.agregar_features_cuotas_derivadas(df)
        df = utils.agregar_features_asian_handicap(df)
        df = utils.agregar_features_rolling_extra(df)
        df = utils.agregar_features_pinnacle_move(df)
    df = utils.agregar_features_multi_escala(df)
    df = utils.agregar_features_forma_momentum(df)
    df = utils.agregar_features_arbitro(df)
    df = utils.agregar_features_descanso(df)

if MODO_SIN_CUOTAS:
    # Filtrar solo partidos con H2H
    if 'H2H_Available' in df.columns:
        antes = len(df)
        df = df[df['H2H_Available'] == 1].copy()
        print(f"   Filtro H2H: {antes} -> {len(df)} partidos con historial")
    df = df.reset_index(drop=True)

    features = [f for f in FEATURES_ESTRUCTURALES if f in df.columns]
    print(f"\n   Features: {len(features)} (SIN cuotas)")
else:
    features = [f for f in ALL_FEATURES if f in df.columns]
    print(f"\n   Features: {len(features)} (CON cuotas)")

# X con NaN para XGBoost, X_filled para RF
X_raw = df[features]
X_filled = X_raw.fillna(0)
y = df['FTR_numeric']

# Split temporal
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, shuffle=False
)
X_train_filled = X_filled.loc[X_train_raw.index]
X_test_filled = X_filled.loc[X_test_raw.index]

print(f"   Train: {len(X_train_raw)} | Test: {len(X_test_raw)}")


# ============================================================================
# HELPERS
# ============================================================================

def _brier_multiclase(y_true, probs):
    bs = 0.0
    for clase in range(3):
        y_bin = (y_true == clase).astype(int)
        bs += brier_score_loss(y_bin, probs[:, clase])
    return bs / 3


# ============================================================================
# OBJETIVO OPTUNA
# ============================================================================

def objective(trial):
    # --- Pesos de clase (co-optimización) ---
    if MODO_SIN_CUOTAS:
        w_local     = trial.suggest_float('peso_local',     0.5, 3.0)
        w_empate    = trial.suggest_float('peso_empate',    1.0, 5.0)
        w_visitante = trial.suggest_float('peso_visitante', 0.5, 3.0)
    else:
        w_local     = trial.suggest_float('peso_local',     0.5, 3.0)
        w_empate    = trial.suggest_float('peso_empate',    1.0, 6.0)
        w_visitante = trial.suggest_float('peso_visitante', 0.5, 3.0)

    class_weight = {0: w_local, 1: w_empate, 2: w_visitante}

    if MODO_XGB:
        # --- XGBoost hyperparams (rangos corregidos 3.4) ---
        n_estimators      = trial.suggest_int  ('n_estimators',      200, 800, step=50)
        max_depth         = trial.suggest_int  ('max_depth',           3,  10)
        learning_rate     = trial.suggest_float('learning_rate',   0.003, 0.15, log=True)
        subsample         = trial.suggest_float('subsample',         0.5,  1.0)
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5,  1.0)
        colsample_bylevel = trial.suggest_float('colsample_bylevel',0.5,  1.0)
        reg_alpha         = trial.suggest_float('reg_alpha',         0.0,  3.0)
        reg_lambda        = trial.suggest_float('reg_lambda',        0.5,  5.0)
        min_child_weight  = trial.suggest_int  ('min_child_weight',    5,  30)
        gamma             = trial.suggest_float('gamma',             0.0,  2.0)

        # Cross-val manual con early stopping (3.3)
        tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
        X_tr = X_train_raw  # XGBoost maneja NaN
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_tr)):
            Xf_tr, Xf_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
            yf_tr, yf_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            sw_tr = compute_sample_weight(class_weight=class_weight, y=yf_tr)

            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                colsample_bylevel=colsample_bylevel,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                min_child_weight=min_child_weight,
                gamma=gamma,
                early_stopping_rounds=50,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                eval_metric='mlogloss',
                verbosity=0,
            )
            model.fit(
                Xf_tr, yf_tr,
                sample_weight=sw_tr,
                eval_set=[(Xf_val, yf_val)],
                verbose=False,
            )
            probs = model.predict_proba(Xf_val)
            ll = log_loss(yf_val, probs)
            fold_scores.append(ll)

            # Pruning: reportar score intermedio para MedianPruner
            trial.report(np.mean(fold_scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # 3.1: penalizar varianza entre folds
        return np.mean(fold_scores) + 0.5 * np.std(fold_scores)

    else:
        # --- Random Forest ---
        n_estimators     = trial.suggest_int('n_estimators',     100, 600, step=50)
        max_depth        = trial.suggest_int('max_depth',          4,  15)
        min_samples_leaf = trial.suggest_int('min_samples_leaf',   2,  15)
        max_features     = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

        tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
        X_tr = X_train_filled  # RF no soporta NaN
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_tr)):
            Xf_tr, Xf_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
            yf_tr, yf_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight=class_weight,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
            model.fit(Xf_tr, yf_tr)
            probs = model.predict_proba(Xf_val)
            ll = log_loss(yf_val, probs)
            fold_scores.append(ll)

            trial.report(np.mean(fold_scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(fold_scores) + 0.5 * np.std(fold_scores)


# ============================================================================
# EJECUTAR OPTIMIZACIÓN
# ============================================================================

print("\n" + "=" * 70)
print(f"EJECUTANDO OPTIMIZACION ({N_TRIALS} trials) — modo: {modo_label}")
print("=" * 70)

# 3.2: MedianPruner — poda trials que no superan la mediana en folds intermedios
study = optuna.create_study(
    direction='minimize',  # 3.1: minimizar log_loss (no maximizar f1)
    sampler=TPESampler(seed=RANDOM_SEED),
    pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=1),
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# ============================================================================
# RESULTADOS
# ============================================================================

print("\n" + "=" * 70)
print("RESULTADOS DE OPTIMIZACION")
print("=" * 70)

best = study.best_params
n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

print(f"\n   Trials completados: {n_complete} | Podados: {n_pruned} | Total: {N_TRIALS}")
print(f"   Mejor score (LL + 0.5*STD): {study.best_value:.4f}")

print(f"\n   MEJORES PARAMETROS ({modo_label}):")
print(f"   Peso Local:        {best['peso_local']:.4f}")
print(f"   Peso Empate:       {best['peso_empate']:.4f}")
print(f"   Peso Visitante:    {best['peso_visitante']:.4f}")

if MODO_XGB:
    print(f"   n_estimators:      {best['n_estimators']}")
    print(f"   max_depth:         {best['max_depth']}")
    print(f"   learning_rate:     {best['learning_rate']:.5f}")
    print(f"   subsample:         {best['subsample']:.4f}")
    print(f"   colsample_bytree:  {best['colsample_bytree']:.4f}")
    print(f"   colsample_bylevel: {best['colsample_bylevel']:.4f}")
    print(f"   reg_alpha:         {best['reg_alpha']:.4f}")
    print(f"   reg_lambda:        {best['reg_lambda']:.4f}")
    print(f"   min_child_weight:  {best['min_child_weight']}")
    print(f"   gamma:             {best['gamma']:.4f}")
else:
    print(f"   n_estimators:      {best['n_estimators']}")
    print(f"   max_depth:         {best['max_depth']}")
    print(f"   min_samples_leaf:  {best['min_samples_leaf']}")
    print(f"   max_features:      {best['max_features']}")

# ============================================================================
# EVALUAR EN TEST
# ============================================================================

print("\n" + "=" * 70)
print("EVALUACION EN TEST SET")
print("=" * 70)

best_weights = {0: best['peso_local'], 1: best['peso_empate'], 2: best['peso_visitante']}

if MODO_XGB:
    sw_final = compute_sample_weight(class_weight=best_weights, y=y_train)
    modelo_opt = XGBClassifier(
        n_estimators=best['n_estimators'],
        max_depth=best['max_depth'],
        learning_rate=best['learning_rate'],
        subsample=best['subsample'],
        colsample_bytree=best['colsample_bytree'],
        colsample_bylevel=best['colsample_bylevel'],
        reg_alpha=best['reg_alpha'],
        reg_lambda=best['reg_lambda'],
        min_child_weight=best['min_child_weight'],
        gamma=best['gamma'],
        early_stopping_rounds=50,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric='mlogloss',
        verbosity=0,
    )
    modelo_opt.fit(
        X_train_raw, y_train,
        sample_weight=sw_final,
        eval_set=[(X_test_raw, y_test)],
        verbose=False,
    )
    X_te = X_test_raw
    if hasattr(modelo_opt, 'best_iteration'):
        print(f"\n   Early stopping: mejor iteracion = {modelo_opt.best_iteration}")
else:
    modelo_opt = RandomForestClassifier(
        n_estimators=best['n_estimators'],
        max_depth=best['max_depth'],
        min_samples_leaf=best['min_samples_leaf'],
        max_features=best['max_features'],
        class_weight=best_weights,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    modelo_opt.fit(X_train_filled, y_train)
    X_te = X_test_filled

pred_opt = modelo_opt.predict(X_te)
probs_opt = modelo_opt.predict_proba(X_te)
acc = accuracy_score(y_test, pred_opt)
f1 = f1_score(y_test, pred_opt, average='weighted')
ll = log_loss(y_test, probs_opt)
bs = _brier_multiclase(y_test, probs_opt)

print(f"\n   MODELO OPTIMIZADO ({modo_label}):")
print(f"   Log Loss:    {ll:.4f}  (primaria)")
print(f"   Brier Score: {bs:.4f}  (secundaria)")
print(f"   F1-Score:    {f1:.4f}  (referencia)")
print(f"   Accuracy:    {acc:.2%}  (referencia)")

cm = confusion_matrix(y_test, pred_opt)
print(f"\n   Matriz de Confusion:")
print(f"                    Prediccion")
print(f"                 Local  Empate  Visit")
print(f"   Real Local    {cm[0,0]:>5}  {cm[0,1]:>6}  {cm[0,2]:>5}")
print(f"   Real Empate   {cm[1,0]:>5}  {cm[1,1]:>6}  {cm[1,2]:>5}")
print(f"   Real Visit    {cm[2,0]:>5}  {cm[2,1]:>6}  {cm[2,2]:>5}")

recall_l = cm[0,0] / cm[0].sum() if cm[0].sum() > 0 else 0
recall_e = cm[1,1] / cm[1].sum() if cm[1].sum() > 0 else 0
recall_v = cm[2,2] / cm[2].sum() if cm[2].sum() > 0 else 0
print(f"\n   Recall por clase:")
print(f"   Local:     {recall_l:.2%}")
print(f"   Empate:    {recall_e:.2%}")
print(f"   Visitante: {recall_v:.2%}")

# ============================================================================
# COMPARACIÓN CON BASELINES
# ============================================================================

print("\n" + "=" * 70)
print("COMPARACION CON BASELINES")
print("=" * 70)

# Baseline: RF básico
rf_base = RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                  random_state=RANDOM_SEED, n_jobs=-1)
rf_base.fit(X_train_filled, y_train)
probs_base = rf_base.predict_proba(X_test_filled)
ll_base = log_loss(y_test, probs_base)
bs_base = _brier_multiclase(y_test, probs_base)
f1_base = f1_score(y_test, rf_base.predict(X_test_filled), average='weighted')

# Baseline: RF balanceado
rf_bal = RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                 class_weight='balanced',
                                 random_state=RANDOM_SEED, n_jobs=-1)
rf_bal.fit(X_train_filled, y_train)
probs_bal = rf_bal.predict_proba(X_test_filled)
ll_bal = log_loss(y_test, probs_bal)
bs_bal = _brier_multiclase(y_test, probs_bal)
f1_bal = f1_score(y_test, rf_bal.predict(X_test_filled), average='weighted')

etiqueta_opt = f"Optuna ({modo_label})"

print(f"\n{'Modelo':<40} {'Log Loss':>9} {'Brier':>7} {'F1':>7}")
print("-" * 65)
print(f"{'RF Basico (baseline)':<40} {ll_base:>9.4f} {bs_base:>7.4f} {f1_base:>7.4f}")
print(f"{'RF Balanceado (auto)':<40} {ll_bal:>9.4f} {bs_bal:>7.4f} {f1_bal:>7.4f}")
print(f"{etiqueta_opt:<40} {ll:>9.4f} {bs:>7.4f} {f1:>7.4f}")

# ============================================================================
# CÓDIGO LISTO PARA COPIAR A config.py
# ============================================================================

print("\n" + "=" * 70)
if MODO_XGB and MODO_SIN_CUOTAS:
    print("COPIA ESTO EN config.py  ->  PESOS_XGB_VB  y  PARAMS_XGB_VB")
elif MODO_XGB:
    print("COPIA ESTO EN config.py  ->  PESOS_XGB  y  PARAMS_XGB")
elif MODO_SIN_CUOTAS:
    print("COPIA ESTO EN config.py  ->  PESOS_OPTIMOS_VB  y  PARAMS_OPTIMOS_VB")
else:
    print("COPIA ESTO EN config.py  ->  PESOS_OPTIMOS  y  PARAMS_OPTIMOS")
print("=" * 70)

if MODO_XGB:
    nombre_pesos  = "PESOS_XGB_VB" if MODO_SIN_CUOTAS else "PESOS_XGB"
    nombre_params = "PARAMS_XGB_VB" if MODO_SIN_CUOTAS else "PARAMS_XGB"
    print(f"""
{nombre_pesos} = {{
    0: {best['peso_local']:.4f},   # Local
    1: {best['peso_empate']:.4f},  # Empate
    2: {best['peso_visitante']:.4f},  # Visitante
}}

{nombre_params} = {{
    'n_estimators': {best['n_estimators']},
    'max_depth': {best['max_depth']},
    'learning_rate': {best['learning_rate']:.5f},
    'subsample': {best['subsample']:.4f},
    'colsample_bytree': {best['colsample_bytree']:.4f},
    'colsample_bylevel': {best['colsample_bylevel']:.4f},
    'reg_alpha': {best['reg_alpha']:.4f},
    'reg_lambda': {best['reg_lambda']:.4f},
    'min_child_weight': {best['min_child_weight']},
    'gamma': {best['gamma']:.4f},
    'early_stopping_rounds': 50,
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    'eval_metric': 'mlogloss',
}}
""")
else:
    nombre_pesos  = "PESOS_OPTIMOS_VB" if MODO_SIN_CUOTAS else "PESOS_OPTIMOS"
    nombre_params = "PARAMS_OPTIMOS_VB" if MODO_SIN_CUOTAS else "PARAMS_OPTIMOS"
    print(f"""
{nombre_pesos} = {{
    0: {best['peso_local']:.4f},   # Local
    1: {best['peso_empate']:.4f},  # Empate
    2: {best['peso_visitante']:.4f},  # Visitante
}}

{nombre_params} = {{
    'n_estimators': {best['n_estimators']},
    'max_depth': {best['max_depth']},
    'min_samples_leaf': {best['min_samples_leaf']},
    'max_features': {repr(best['max_features'])},
    'class_weight': {nombre_pesos},
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
}}
""")
