# -*- coding: utf-8 -*-
"""
optimizar_todo_optuna.py
========================
Re-optimiza SIMULTANEAMENTE:
  - Pesos de clase (RF y XGBoost)
  - Hiperparametros de Random Forest
  - Hiperparametros de XGBoost

Usa Optuna con TimeSeriesSplit para respetar el orden temporal.
Al terminar, imprime los mejores parametros para copiar a config.py.

Uso:
    python herramientas/optimizar_todo_optuna.py
    python herramientas/optimizar_todo_optuna.py --trials 150
"""

import argparse
import sys
import os
import warnings
import unittest.mock as mock

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from config import ARCHIVO_FEATURES, ALL_FEATURES, RANDOM_SEED
import utils

# ---------------------------------------------------------------------------
# Cargar y preparar datos UNA SOLA VEZ
# ---------------------------------------------------------------------------

def cargar_datos():
    df = pd.read_csv(ARCHIVO_FEATURES)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    with mock.patch('builtins.print'):
        df = utils.agregar_xg_rolling(df)
        df = utils.agregar_features_tabla(df)
        df = utils.agregar_features_cuotas_derivadas(df)
        df = utils.agregar_features_asian_handicap(df)
        df = utils.agregar_features_rolling_extra(df)

    features = [f for f in ALL_FEATURES if f in df.columns]
    X = df[features].fillna(0)
    y = df['FTR_numeric']
    return X, y, features


# ---------------------------------------------------------------------------
# Objetivos Optuna
# ---------------------------------------------------------------------------

def objetivo_rf(trial, X_train, y_train, tscv):
    """Optimiza pesos + hiperparametros de Random Forest."""
    w0 = trial.suggest_float('w0', 0.5, 3.0)
    w1 = trial.suggest_float('w1', 1.0, 6.0)
    w2 = trial.suggest_float('w2', 0.5, 3.0)

    n_estimators    = trial.suggest_int('n_estimators', 100, 600, step=50)
    max_depth       = trial.suggest_int('max_depth', 4, 15)
    min_samples_leaf= trial.suggest_int('min_samples_leaf', 1, 10)
    max_features    = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

    class_weight = {0: w0, 1: w1, 2: w2}
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        Xtr, Xval = X_train.iloc[train_idx], X_train.iloc[val_idx]
        ytr, yval = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(Xtr, ytr)
        f1 = f1_score(yval, model.predict(Xval), average='weighted')
        scores.append(f1)

    return np.mean(scores)


def objetivo_xgb(trial, X_train, y_train, tscv):
    """Optimiza pesos + hiperparametros de XGBoost."""
    w0 = trial.suggest_float('w0', 0.5, 3.0)
    w1 = trial.suggest_float('w1', 1.0, 6.0)
    w2 = trial.suggest_float('w2', 0.5, 3.0)

    n_estimators      = trial.suggest_int('n_estimators', 200, 800, step=50)
    max_depth         = trial.suggest_int('max_depth', 3, 10)
    learning_rate     = trial.suggest_float('learning_rate', 0.005, 0.1, log=True)
    subsample         = trial.suggest_float('subsample', 0.6, 1.0)
    colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    colsample_bylevel = trial.suggest_float('colsample_bylevel', 0.5, 1.0)
    reg_alpha         = trial.suggest_float('reg_alpha', 0.0, 2.0)
    reg_lambda        = trial.suggest_float('reg_lambda', 0.5, 5.0)
    min_child_weight  = trial.suggest_int('min_child_weight', 1, 10)
    gamma             = trial.suggest_float('gamma', 0.0, 0.5)

    class_weight = {0: w0, 1: w1, 2: w2}
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
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric='mlogloss',
        verbosity=0,
    )

    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        Xtr, Xval = X_train.iloc[train_idx], X_train.iloc[val_idx]
        ytr, yval = y_train.iloc[train_idx], y_train.iloc[val_idx]
        sw = compute_sample_weight(class_weight=class_weight, y=ytr)
        model.fit(Xtr, ytr, sample_weight=sw)
        f1 = f1_score(yval, model.predict(Xval), average='weighted')
        scores.append(f1)

    return np.mean(scores)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_trials: int = 100):
    print("=" * 65)
    print("  OPTIMIZACION COMPLETA CON OPTUNA")
    print(f"  Trials por modelo: {n_trials}")
    print("=" * 65)

    print("\n[1/4] Cargando datos...")
    X, y, features = cargar_datos()
    print(f"      {len(X)} partidos | {len(features)} features")

    # Split temporal 80/20 — el test NO participa en la optimizacion
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=False
    )
    print(f"      Train={len(X_train)} | Test={len(X_test)} (test no se toca)")

    # TimeSeriesSplit sobre el train solamente
    tscv = TimeSeriesSplit(n_splits=5)

    # ---- Optimizar RF ----
    print(f"\n[2/4] Optimizando Random Forest ({n_trials} trials)...")
    study_rf = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_SEED),
    )
    study_rf.optimize(
        lambda trial: objetivo_rf(trial, X_train, y_train, tscv),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best_rf = study_rf.best_params
    print(f"      Mejor F1-CV RF: {study_rf.best_value:.4f}")

    # ---- Optimizar XGBoost ----
    print(f"\n[3/4] Optimizando XGBoost ({n_trials} trials)...")
    study_xgb = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_SEED),
    )
    study_xgb.optimize(
        lambda trial: objetivo_xgb(trial, X_train, y_train, tscv),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best_xgb = study_xgb.best_params
    print(f"      Mejor F1-CV XGB: {study_xgb.best_value:.4f}")

    # ---- Evaluar en test ----
    print(f"\n[4/4] Evaluando en test set ({len(X_test)} partidos)...")

    # RF optimo
    rf_cw  = {0: best_rf['w0'], 1: best_rf['w1'], 2: best_rf['w2']}
    rf_opt = RandomForestClassifier(
        n_estimators    = best_rf['n_estimators'],
        max_depth       = best_rf['max_depth'],
        min_samples_leaf= best_rf['min_samples_leaf'],
        max_features    = best_rf['max_features'],
        class_weight    = rf_cw,
        random_state    = RANDOM_SEED,
        n_jobs          = -1,
    )
    rf_opt.fit(X_train, y_train)
    f1_rf = f1_score(y_test, rf_opt.predict(X_test), average='weighted')
    acc_rf = (rf_opt.predict(X_test) == y_test).mean()

    # XGB optimo
    xgb_cw  = {0: best_xgb['w0'], 1: best_xgb['w1'], 2: best_xgb['w2']}
    sw_test = compute_sample_weight(class_weight=xgb_cw, y=y_train)
    xgb_opt = XGBClassifier(
        n_estimators      = best_xgb['n_estimators'],
        max_depth         = best_xgb['max_depth'],
        learning_rate     = best_xgb['learning_rate'],
        subsample         = best_xgb['subsample'],
        colsample_bytree  = best_xgb['colsample_bytree'],
        colsample_bylevel = best_xgb['colsample_bylevel'],
        reg_alpha         = best_xgb['reg_alpha'],
        reg_lambda        = best_xgb['reg_lambda'],
        min_child_weight  = best_xgb['min_child_weight'],
        gamma             = best_xgb['gamma'],
        random_state      = RANDOM_SEED,
        n_jobs            = -1,
        eval_metric       = 'mlogloss',
        verbosity         = 0,
    )
    xgb_opt.fit(X_train, y_train, sample_weight=sw_test)
    f1_xgb = f1_score(y_test, xgb_opt.predict(X_test), average='weighted')
    acc_xgb = (xgb_opt.predict(X_test) == y_test).mean()

    print(f"\n      RF  optimo  -> F1={f1_rf:.4f}  Acc={acc_rf:.4f}")
    print(f"      XGB optimo -> F1={f1_xgb:.4f}  Acc={acc_xgb:.4f}")

    # ---- Imprimir parametros para config.py ----
    print("\n" + "=" * 65)
    print("  PARAMETROS OPTIMOS -> copiar a config.py")
    print("=" * 65)

    print(f"""
# --- RF OPTUNA ---
PESOS_OPTIMOS = {{
    0: {best_rf['w0']:.4f},  # Local
    1: {best_rf['w1']:.4f},  # Empate
    2: {best_rf['w2']:.4f},  # Visitante
}}
PARAMS_OPTIMOS = {{
    'n_estimators':     {best_rf['n_estimators']},
    'max_depth':        {best_rf['max_depth']},
    'min_samples_leaf': {best_rf['min_samples_leaf']},
    'max_features':     {repr(best_rf['max_features'])},
    'class_weight':     PESOS_OPTIMOS,
    'random_state':     RANDOM_SEED,
    'n_jobs':           -1,
}}

# --- XGB OPTUNA ---
PESOS_XGB = {{
    0: {best_xgb['w0']:.4f},
    1: {best_xgb['w1']:.4f},
    2: {best_xgb['w2']:.4f},
}}
PARAMS_XGB = {{
    'n_estimators':      {best_xgb['n_estimators']},
    'max_depth':         {best_xgb['max_depth']},
    'learning_rate':     {best_xgb['learning_rate']:.5f},
    'subsample':         {best_xgb['subsample']:.4f},
    'colsample_bytree':  {best_xgb['colsample_bytree']:.4f},
    'colsample_bylevel': {best_xgb['colsample_bylevel']:.4f},
    'reg_alpha':         {best_xgb['reg_alpha']:.4f},
    'reg_lambda':        {best_xgb['reg_lambda']:.4f},
    'min_child_weight':  {best_xgb['min_child_weight']},
    'gamma':             {best_xgb['gamma']:.4f},
    'random_state':      RANDOM_SEED,
    'n_jobs':            -1,
    'eval_metric':       'mlogloss',
}}
""")

    # Guardar resultados para uso posterior
    resultados = {
        'f1_rf':   f1_rf,
        'f1_xgb':  f1_xgb,
        'acc_rf':  acc_rf,
        'acc_xgb': acc_xgb,
        'best_rf':  best_rf,
        'best_xgb': best_xgb,
        'ganador':  'RF' if f1_rf >= f1_xgb else 'XGB',
    }

    import json
    ruta_json = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'modelos', 'optuna_best_params.json')
    with open(ruta_json, 'w') as f:
        json.dump(resultados, f, indent=2)
    print(f"\n  Resultados guardados en: {os.path.abspath(ruta_json)}")
    print(f"  Ganador: {resultados['ganador']} (F1={max(f1_rf, f1_xgb):.4f})")

    return resultados


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=100,
                        help='Numero de trials Optuna por modelo (default: 100)')
    args = parser.parse_args()
    main(n_trials=args.trials)
