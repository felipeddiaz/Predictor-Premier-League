# -*- coding: utf-8 -*-
"""
optimizar_xgb_rapido.py
=======================
Optimizacion XGB de 30 trials con espacio reducido (mas rapido).
Lee el JSON parcial existente (best_rf ya guardado) y agrega best_xgb.
"""

import sys, os, json, warnings, unittest.mock as mock
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


def objetivo_xgb(trial, X_train, y_train, tscv):
    w0 = trial.suggest_float('w0', 0.5, 2.5)
    w1 = trial.suggest_float('w1', 1.5, 4.5)
    w2 = trial.suggest_float('w2', 0.8, 2.5)
    # Espacio reducido centrado en valores prometedores
    n_estimators      = trial.suggest_int('n_estimators', 300, 600, step=100)
    max_depth         = trial.suggest_int('max_depth', 4, 7)
    learning_rate     = trial.suggest_float('learning_rate', 0.02, 0.06, log=True)
    subsample         = trial.suggest_float('subsample', 0.70, 0.95)
    colsample_bytree  = trial.suggest_float('colsample_bytree', 0.60, 0.90)
    colsample_bylevel = trial.suggest_float('colsample_bylevel', 0.60, 0.90)
    reg_alpha         = trial.suggest_float('reg_alpha', 0.1, 1.0)
    reg_lambda        = trial.suggest_float('reg_lambda', 1.0, 4.0)
    min_child_weight  = trial.suggest_int('min_child_weight', 2, 6)
    gamma             = trial.suggest_float('gamma', 0.0, 0.3)

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


def main():
    print("=" * 60)
    print("  XGB OPTUNA - 30 trials (espacio reducido)")
    print("=" * 60)

    # Cargar RF ya guardado
    with open(RUTA_JSON) as f:
        datos = json.load(f)
    best_rf = datos['best_rf']
    f1_cv_rf = datos['f1_cv_rf']
    print(f"\n  RF ya optimizado: F1-CV={f1_cv_rf:.4f}")

    print("\nCargando datos...")
    X, y = cargar_datos()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=False)
    tscv = TimeSeriesSplit(n_splits=5)
    print(f"  Train={len(X_train)} | Test={len(X_test)}")

    print("\n[XGB] Optimizando 30 trials...")
    study_xgb = optuna.create_study(direction='maximize',
                                    sampler=TPESampler(seed=RANDOM_SEED))
    study_xgb.optimize(
        lambda t: objetivo_xgb(t, X_train, y_train, tscv),
        n_trials=30, show_progress_bar=True,
    )
    best_xgb = study_xgb.best_params
    print(f"  Mejor F1-CV XGB: {study_xgb.best_value:.4f}")

    # ── Evaluar en test ──────────────────────────────────────────────
    print("\n[Test] Evaluando RF y XGB...")

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
        'f1_cv_rf': f1_cv_rf, 'f1_cv_xgb': study_xgb.best_value,
        'best_rf': best_rf, 'best_xgb': best_xgb,
        'ganador': ganador,
    }
    with open(RUTA_JSON, 'w') as f:
        json.dump(resultados, f, indent=2)
    print(f"\n  JSON completo guardado: {os.path.abspath(RUTA_JSON)}")

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
    main()
