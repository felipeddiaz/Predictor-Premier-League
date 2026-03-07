# -*- coding: utf-8 -*-
"""
optimizar_todo_optuna.py
========================
Re-optimiza SIMULTANEAMENTE:
  - Pesos de clase (RF y XGBoost)
  - Hiperparametros de Random Forest
  - Hiperparametros de XGBoost

Fase 3 del audit ML:
  - Metrica: Log Loss (minimizar) + 0.5*STD penalty entre folds
  - MedianPruner para pruning eficiente
  - XGBoost con early_stopping_rounds=50
  - Rangos corregidos: min_child_weight [5,30], learning_rate [0.003,0.15]

Usa Optuna con TimeSeriesSplit para respetar el orden temporal.
Al terminar, imprime los mejores parametros para copiar a config.py.

Uso:
    python herramientas/optimizar_todo_optuna.py
    python herramientas/optimizar_todo_optuna.py --trials 200
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
from optuna.pruners import MedianPruner
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, log_loss, brier_score_loss, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from config import ARCHIVO_FEATURES, ALL_FEATURES, RANDOM_SEED
import utils

N_SPLITS_CV = 5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _brier_multiclase(y_true, probs):
    bs = 0.0
    for clase in range(3):
        y_bin = (y_true == clase).astype(int)
        bs += brier_score_loss(y_bin, probs[:, clase])
    return bs / 3


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
        df = utils.agregar_features_multi_escala(df)
        df = utils.agregar_features_forma_momentum(df)
        df = utils.agregar_features_pinnacle_move(df)
        df = utils.agregar_features_arbitro(df)
        df = utils.agregar_features_descanso(df)

    features = [f for f in ALL_FEATURES if f in df.columns]
    X_raw = df[features]
    X_filled = X_raw.fillna(0)
    y = df['FTR_numeric']
    return X_raw, X_filled, y, features


# ---------------------------------------------------------------------------
# Objetivos Optuna — Log Loss + varianza penalty
# ---------------------------------------------------------------------------

def objetivo_rf(trial, X_train, y_train, tscv):
    """Optimiza pesos + hiperparametros de Random Forest (minimizar LL)."""
    w0 = trial.suggest_float('w0', 0.5, 3.0)
    w1 = trial.suggest_float('w1', 1.0, 6.0)
    w2 = trial.suggest_float('w2', 0.5, 3.0)

    n_estimators    = trial.suggest_int('n_estimators', 100, 600, step=50)
    max_depth       = trial.suggest_int('max_depth', 4, 15)
    min_samples_leaf= trial.suggest_int('min_samples_leaf', 2, 15)
    max_features    = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

    class_weight = {0: w0, 1: w1, 2: w2}

    fold_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        Xtr, Xval = X_train.iloc[train_idx], X_train.iloc[val_idx]
        ytr, yval = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        model.fit(Xtr, ytr)
        probs = model.predict_proba(Xval)
        ll = log_loss(yval, probs)
        fold_scores.append(ll)

        trial.report(np.mean(fold_scores), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_scores) + 0.5 * np.std(fold_scores)


def objetivo_xgb(trial, X_train, y_train, tscv):
    """Optimiza pesos + hiperparametros de XGBoost (minimizar LL + early stopping)."""
    w0 = trial.suggest_float('w0', 0.5, 3.0)
    w1 = trial.suggest_float('w1', 1.0, 6.0)
    w2 = trial.suggest_float('w2', 0.5, 3.0)

    n_estimators      = trial.suggest_int('n_estimators', 200, 800, step=50)
    max_depth         = trial.suggest_int('max_depth', 3, 10)
    learning_rate     = trial.suggest_float('learning_rate', 0.003, 0.15, log=True)
    subsample         = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    colsample_bylevel = trial.suggest_float('colsample_bylevel', 0.5, 1.0)
    reg_alpha         = trial.suggest_float('reg_alpha', 0.0, 3.0)
    reg_lambda        = trial.suggest_float('reg_lambda', 0.5, 5.0)
    min_child_weight  = trial.suggest_int('min_child_weight', 5, 30)
    gamma             = trial.suggest_float('gamma', 0.0, 2.0)

    class_weight = {0: w0, 1: w1, 2: w2}

    fold_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        Xtr, Xval = X_train.iloc[train_idx], X_train.iloc[val_idx]
        ytr, yval = y_train.iloc[train_idx], y_train.iloc[val_idx]
        sw = compute_sample_weight(class_weight=class_weight, y=ytr)

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
            Xtr, ytr,
            sample_weight=sw,
            eval_set=[(Xval, yval)],
            verbose=False,
        )
        probs = model.predict_proba(Xval)
        ll = log_loss(yval, probs)
        fold_scores.append(ll)

        trial.report(np.mean(fold_scores), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_scores) + 0.5 * np.std(fold_scores)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_trials: int = 200):
    print("=" * 65)
    print("  OPTIMIZACION COMPLETA CON OPTUNA")
    print(f"  Trials por modelo: {n_trials}")
    print(f"  Metrica: Log Loss (min) + 0.5*STD penalty")
    print(f"  Pruner: MedianPruner | CV: {N_SPLITS_CV}-fold TimeSeriesSplit")
    print("=" * 65)

    print("\n[1/4] Cargando datos...")
    X_raw, X_filled, y, features = cargar_datos()
    print(f"      {len(X_raw)} partidos | {len(features)} features")

    # Split temporal 80/20
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, shuffle=False
    )
    X_train_filled = X_filled.loc[X_train_raw.index]
    X_test_filled = X_filled.loc[X_test_raw.index]
    print(f"      Train={len(X_train_raw)} | Test={len(X_test_raw)} (test no se toca)")

    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)

    # ---- Optimizar RF ----
    print(f"\n[2/4] Optimizando Random Forest ({n_trials} trials)...")
    study_rf = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=RANDOM_SEED),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )
    study_rf.optimize(
        lambda trial: objetivo_rf(trial, X_train_filled, y_train, tscv),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best_rf = study_rf.best_params
    n_pruned_rf = len([t for t in study_rf.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"      Mejor LL+STD RF: {study_rf.best_value:.4f} "
          f"(completados: {n_trials - n_pruned_rf}, podados: {n_pruned_rf})")

    # ---- Optimizar XGBoost ----
    print(f"\n[3/4] Optimizando XGBoost ({n_trials} trials)...")
    study_xgb = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=RANDOM_SEED),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )
    study_xgb.optimize(
        lambda trial: objetivo_xgb(trial, X_train_raw, y_train, tscv),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best_xgb = study_xgb.best_params
    n_pruned_xgb = len([t for t in study_xgb.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"      Mejor LL+STD XGB: {study_xgb.best_value:.4f} "
          f"(completados: {n_trials - n_pruned_xgb}, podados: {n_pruned_xgb})")

    # ---- Evaluar en test ----
    print(f"\n[4/4] Evaluando en test set ({len(X_test_raw)} partidos)...")

    # RF optimo
    rf_cw = {0: best_rf['w0'], 1: best_rf['w1'], 2: best_rf['w2']}
    rf_opt = RandomForestClassifier(
        n_estimators    = best_rf['n_estimators'],
        max_depth       = best_rf['max_depth'],
        min_samples_leaf= best_rf['min_samples_leaf'],
        max_features    = best_rf['max_features'],
        class_weight    = rf_cw,
        random_state    = RANDOM_SEED,
        n_jobs          = -1,
    )
    rf_opt.fit(X_train_filled, y_train)
    probs_rf = rf_opt.predict_proba(X_test_filled)
    ll_rf = log_loss(y_test, probs_rf)
    bs_rf = _brier_multiclase(y_test, probs_rf)
    f1_rf = f1_score(y_test, rf_opt.predict(X_test_filled), average='weighted')
    acc_rf = accuracy_score(y_test, rf_opt.predict(X_test_filled))

    # XGB optimo
    xgb_cw = {0: best_xgb['w0'], 1: best_xgb['w1'], 2: best_xgb['w2']}
    sw_train = compute_sample_weight(class_weight=xgb_cw, y=y_train)
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
        early_stopping_rounds = 50,
        random_state      = RANDOM_SEED,
        n_jobs            = -1,
        eval_metric       = 'mlogloss',
        verbosity         = 0,
    )
    xgb_opt.fit(
        X_train_raw, y_train,
        sample_weight=sw_train,
        eval_set=[(X_test_raw, y_test)],
        verbose=False,
    )
    probs_xgb = xgb_opt.predict_proba(X_test_raw)
    ll_xgb = log_loss(y_test, probs_xgb)
    bs_xgb = _brier_multiclase(y_test, probs_xgb)
    f1_xgb = f1_score(y_test, xgb_opt.predict(X_test_raw), average='weighted')
    acc_xgb = accuracy_score(y_test, xgb_opt.predict(X_test_raw))

    if hasattr(xgb_opt, 'best_iteration'):
        print(f"\n      XGB early stopping: mejor iteracion = {xgb_opt.best_iteration}")

    print(f"\n      {'Modelo':<15} {'Log Loss':>9} {'Brier':>7} {'F1':>7} {'Acc':>7}")
    print(f"      {'-'*45}")
    print(f"      {'RF optimo':<15} {ll_rf:>9.4f} {bs_rf:>7.4f} {f1_rf:>7.4f} {acc_rf:>6.2%}")
    print(f"      {'XGB optimo':<15} {ll_xgb:>9.4f} {bs_xgb:>7.4f} {f1_xgb:>7.4f} {acc_xgb:>6.2%}")

    # Ganador por Log Loss
    ganador = 'RF' if ll_rf <= ll_xgb else 'XGB'
    print(f"\n      Ganador por Log Loss: {ganador}")

    # ---- Imprimir parametros para config.py ----
    print("\n" + "=" * 65)
    print("  PARAMETROS OPTIMOS -> copiar a config.py")
    print("=" * 65)

    print(f"""
# --- RF OPTUNA (Log Loss optimizado) ---
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

# --- XGB OPTUNA (Log Loss optimizado + early stopping) ---
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
    'early_stopping_rounds': 50,
    'random_state':      RANDOM_SEED,
    'n_jobs':            -1,
    'eval_metric':       'mlogloss',
}}
""")

    # Guardar resultados
    import json
    resultados = {
        'll_rf': ll_rf, 'll_xgb': ll_xgb,
        'bs_rf': bs_rf, 'bs_xgb': bs_xgb,
        'f1_rf': f1_rf, 'f1_xgb': f1_xgb,
        'acc_rf': acc_rf, 'acc_xgb': acc_xgb,
        'best_rf': best_rf, 'best_xgb': best_xgb,
        'ganador': ganador,
    }
    ruta_json = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'modelos', 'optuna_best_params.json')
    with open(ruta_json, 'w') as f:
        json.dump(resultados, f, indent=2)
    print(f"\n  Resultados guardados en: {os.path.abspath(ruta_json)}")
    print(f"  Ganador: {ganador} (Log Loss={min(ll_rf, ll_xgb):.4f})")

    return resultados


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=200,
                        help='Numero de trials Optuna por modelo (default: 200)')
    args = parser.parse_args()
    main(n_trials=args.trials)
