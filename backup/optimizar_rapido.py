# -*- coding: utf-8 -*-
"""
optimizar_rapido.py
===================
Optimizacion rapida RF+XGB con 60 trials cada uno.
Guarda INMEDIATAMENTE los mejores params en JSON tras cada estudio.
"""

import argparse, sys, os, json, warnings, unittest.mock as mock
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

RUTA_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'modelos', 'optuna_best_params.json')


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
    return X, y


def objetivo_rf(trial, X_train, y_train, tscv):
    w0 = trial.suggest_float('w0', 0.5, 3.0)
    w1 = trial.suggest_float('w1', 1.0, 5.0)
    w2 = trial.suggest_float('w2', 0.5, 3.0)
    n_estimators     = trial.suggest_int('n_estimators', 150, 500, step=50)
    max_depth        = trial.suggest_int('max_depth', 5, 12)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 8)
    max_features     = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, max_features=max_features,
        class_weight={0: w0, 1: w1, 2: w2},
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    scores = []
    for tr, val in tscv.split(X_train):
        model.fit(X_train.iloc[tr], y_train.iloc[tr])
        scores.append(f1_score(y_train.iloc[val],
                               model.predict(X_train.iloc[val]),
                               average='weighted'))
    return float(np.mean(scores))


def objetivo_xgb(trial, X_train, y_train, tscv):
    w0 = trial.suggest_float('w0', 0.5, 3.0)
    w1 = trial.suggest_float('w1', 1.0, 5.0)
    w2 = trial.suggest_float('w2', 0.5, 3.0)
    n_estimators      = trial.suggest_int('n_estimators', 200, 600, step=50)
    max_depth         = trial.suggest_int('max_depth', 3, 8)
    learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.08, log=True)
    subsample         = trial.suggest_float('subsample', 0.65, 1.0)
    colsample_bytree  = trial.suggest_float('colsample_bytree', 0.55, 1.0)
    colsample_bylevel = trial.suggest_float('colsample_bylevel', 0.55, 1.0)
    reg_alpha         = trial.suggest_float('reg_alpha', 0.0, 1.5)
    reg_lambda        = trial.suggest_float('reg_lambda', 0.5, 4.0)
    min_child_weight  = trial.suggest_int('min_child_weight', 1, 8)
    gamma             = trial.suggest_float('gamma', 0.0, 0.4)
    cw = {0: w0, 1: w1, 2: w2}
    model = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=subsample,
        colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel,
        reg_alpha=reg_alpha, reg_lambda=reg_lambda,
        min_child_weight=min_child_weight, gamma=gamma,
        random_state=RANDOM_SEED, n_jobs=-1, eval_metric='mlogloss', verbosity=0,
    )
    scores = []
    for tr, val in tscv.split(X_train):
        sw = compute_sample_weight(class_weight=cw, y=y_train.iloc[tr])
        model.fit(X_train.iloc[tr], y_train.iloc[tr], sample_weight=sw)
        scores.append(f1_score(y_train.iloc[val],
                               model.predict(X_train.iloc[val]),
                               average='weighted'))
    return float(np.mean(scores))


def main(n_trials=60):
    print("=" * 60)
    print(f"  OPTIMIZACION RAPIDA  ({n_trials} trials/modelo)")
    print("=" * 60)

    print("\n[Datos] Cargando...")
    X, y = cargar_datos()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=False)
    tscv = TimeSeriesSplit(n_splits=5)
    print(f"  Train={len(X_train)} | Test={len(X_test)}")

    # ── RF ──────────────────────────────────────────────────────────
    print(f"\n[RF] Optimizando {n_trials} trials...")
    study_rf = optuna.create_study(direction='maximize',
                                   sampler=TPESampler(seed=RANDOM_SEED))
    study_rf.optimize(
        lambda t: objetivo_rf(t, X_train, y_train, tscv),
        n_trials=n_trials, show_progress_bar=True,
    )
    best_rf = study_rf.best_params
    print(f"  Mejor F1-CV RF: {study_rf.best_value:.4f}")

    # Guardar RF inmediatamente
    parcial = {'best_rf': best_rf, 'f1_cv_rf': study_rf.best_value}
    with open(RUTA_JSON, 'w') as f:
        json.dump(parcial, f, indent=2)
    print("  RF params guardados en JSON (parcial)")

    # ── XGB ─────────────────────────────────────────────────────────
    print(f"\n[XGB] Optimizando {n_trials} trials...")
    study_xgb = optuna.create_study(direction='maximize',
                                    sampler=TPESampler(seed=RANDOM_SEED))
    study_xgb.optimize(
        lambda t: objetivo_xgb(t, X_train, y_train, tscv),
        n_trials=n_trials, show_progress_bar=True,
    )
    best_xgb = study_xgb.best_params
    print(f"  Mejor F1-CV XGB: {study_xgb.best_value:.4f}")

    # ── Evaluar en test ──────────────────────────────────────────────
    print("\n[Test] Evaluando ambos modelos...")

    rf_cw = {0: best_rf['w0'], 1: best_rf['w1'], 2: best_rf['w2']}
    rf_opt = RandomForestClassifier(
        n_estimators=best_rf['n_estimators'], max_depth=best_rf['max_depth'],
        min_samples_leaf=best_rf['min_samples_leaf'],
        max_features=best_rf['max_features'],
        class_weight=rf_cw, random_state=RANDOM_SEED, n_jobs=-1,
    )
    rf_opt.fit(X_train, y_train)
    f1_rf  = f1_score(y_test, rf_opt.predict(X_test), average='weighted')
    acc_rf = (rf_opt.predict(X_test) == y_test).mean()
    print(f"  RF  -> F1={f1_rf:.4f}  Acc={acc_rf:.4f}")

    xgb_cw = {0: best_xgb['w0'], 1: best_xgb['w1'], 2: best_xgb['w2']}
    sw = compute_sample_weight(class_weight=xgb_cw, y=y_train)
    xgb_opt = XGBClassifier(
        n_estimators=best_xgb['n_estimators'], max_depth=best_xgb['max_depth'],
        learning_rate=best_xgb['learning_rate'], subsample=best_xgb['subsample'],
        colsample_bytree=best_xgb['colsample_bytree'],
        colsample_bylevel=best_xgb['colsample_bylevel'],
        reg_alpha=best_xgb['reg_alpha'], reg_lambda=best_xgb['reg_lambda'],
        min_child_weight=best_xgb['min_child_weight'], gamma=best_xgb['gamma'],
        random_state=RANDOM_SEED, n_jobs=-1, eval_metric='mlogloss', verbosity=0,
    )
    xgb_opt.fit(X_train, y_train, sample_weight=sw)
    f1_xgb  = f1_score(y_test, xgb_opt.predict(X_test), average='weighted')
    acc_xgb = (xgb_opt.predict(X_test) == y_test).mean()
    print(f"  XGB -> F1={f1_xgb:.4f}  Acc={acc_xgb:.4f}")

    ganador = 'RF' if f1_rf >= f1_xgb else 'XGB'
    print(f"\n  Ganador: {ganador}  (F1 test={max(f1_rf, f1_xgb):.4f})")

    # ── Guardar JSON completo ────────────────────────────────────────
    resultados = {
        'f1_rf': f1_rf, 'f1_xgb': f1_xgb,
        'acc_rf': acc_rf, 'acc_xgb': acc_xgb,
        'f1_cv_rf': study_rf.best_value, 'f1_cv_xgb': study_xgb.best_value,
        'best_rf': best_rf, 'best_xgb': best_xgb,
        'ganador': ganador,
    }
    with open(RUTA_JSON, 'w') as f:
        json.dump(resultados, f, indent=2)
    print(f"\n  JSON completo guardado: {os.path.abspath(RUTA_JSON)}")

    # ── Imprimir params para config.py ───────────────────────────────
    print("\n" + "=" * 60)
    print("  COPIAR EN config.py:")
    print("=" * 60)
    print(f"""
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=60)
    args = parser.parse_args()
    main(args.trials)
