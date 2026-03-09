# -*- coding: utf-8 -*-
"""
optimizar_xgb_25feat.py
=======================
Optimiza XGBoost sobre las 25 features seleccionadas por permutation importance.
Usa Optuna con TPESampler + penalizacion por sobreajuste en el fold mas reciente.

- 150 trials (warm start con los params actuales de config.py)
- Espacio de busqueda amplio
- Guarda JSON con mejores params en modelos/optuna_xgb_25feat.json
- Imprime bloque listo para copiar en config.py
"""

import sys, os, json, warnings, unittest.mock as mock
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from config import ARCHIVO_FEATURES, FEATURES_SELECCIONADAS, RANDOM_SEED, PARAMS_XGB, PESOS_XGB
import utils

RUTA_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'modelos', 'optuna_xgb_25feat.json')

_P = PARAMS_XGB
_W = PESOS_XGB


def cargar_datos():
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
    # XGBoost no maneja NaN nativamente: fillna(0)
    X = df[feats].fillna(0)
    y = df['FTR_numeric']
    return X, y, feats


def objetivo(trial, X_train, y_train, tscv):
    """
    Maximiza F1-CV ponderado con penalizacion por sobreajuste temporal.
    """
    w0 = trial.suggest_float('w0', 0.4, 2.0)
    w1 = trial.suggest_float('w1', 0.8, 3.5)
    w2 = trial.suggest_float('w2', 0.4, 2.0)

    n_estimators      = trial.suggest_int('n_estimators', 100, 600, step=50)
    max_depth         = trial.suggest_int('max_depth', 3, 10)
    learning_rate     = trial.suggest_float('learning_rate', 0.005, 0.15, log=True)
    subsample         = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree  = trial.suggest_float('colsample_bytree', 0.4, 1.0)
    colsample_bylevel = trial.suggest_float('colsample_bylevel', 0.4, 1.0)
    reg_alpha         = trial.suggest_float('reg_alpha', 0.0, 5.0)
    reg_lambda        = trial.suggest_float('reg_lambda', 0.3, 6.0)
    min_child_weight  = trial.suggest_int('min_child_weight', 1, 15)
    gamma             = trial.suggest_float('gamma', 0.0, 0.5)

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
        sw = compute_sample_weight(cw, y_train.iloc[tr])
        model.fit(X_train.iloc[tr], y_train.iloc[tr], sample_weight=sw)
        scores.append(f1_score(y_train.iloc[val], model.predict(X_train.iloc[val]),
                               average='weighted'))

    mean_cv = float(np.mean(scores))
    last_fold = scores[-1]

    # Penalizacion: gap > 1.5% penaliza fuerte
    gap = mean_cv - last_fold
    penalizacion = max(0.0, gap - 0.015) * 2.0

    return mean_cv - penalizacion


def warm_start_params():
    """Primer trial = params actuales de config.py (warm start)."""
    return {
        'w0': _W[0], 'w1': _W[1], 'w2': _W[2],
        'n_estimators':      _P['n_estimators'],
        'max_depth':         _P['max_depth'],
        'learning_rate':     _P['learning_rate'],
        'subsample':         _P['subsample'],
        'colsample_bytree':  _P['colsample_bytree'],
        'colsample_bylevel': _P['colsample_bylevel'],
        'reg_alpha':         _P['reg_alpha'],
        'reg_lambda':        _P['reg_lambda'],
        'min_child_weight':  _P['min_child_weight'],
        'gamma':             _P['gamma'],
    }


def main(n_trials=150):
    print("=" * 65)
    print(f"  XGB OPTUNA — 25 features ({n_trials} trials)")
    print(f"  Espacio amplio + penalizacion por sobreajuste temporal")
    print("=" * 65)

    X, y, feats = cargar_datos()
    print(f"\n  Features usadas: {len(feats)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
    tscv = TimeSeriesSplit(n_splits=5)
    print(f"  Train={len(X_train)} | Test={len(X_test)}")

    # Baseline: params actuales con las 25 features
    sw_base = compute_sample_weight(_W, y_train)
    base_model = XGBClassifier(**{k: v for k, v in _P.items()}, verbosity=0)
    base_model.fit(X_train, y_train, sample_weight=sw_base)
    base_f1 = f1_score(y_test, base_model.predict(X_test), average='weighted')
    print(f"  Baseline (params actuales, 25 feats): F1-test = {base_f1:.4f}")
    print()

    sampler = TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.enqueue_trial(warm_start_params())

    def guardar_si_mejora(study, trial):
        if trial.number == study.best_trial.number:
            with open(RUTA_JSON, 'w') as f:
                json.dump({'best_params': study.best_params,
                           'f1_cv_penalizado': study.best_value,
                           'baseline_f1_test': base_f1,
                           'features': feats,
                           'n_features': len(feats),
                           'trials_completados': trial.number + 1}, f, indent=2)

    print(f"Optimizando {n_trials} trials (warm start incluido)...")
    study.optimize(
        lambda t: objetivo(t, X_train, y_train, tscv),
        n_trials=n_trials, show_progress_bar=True,
        callbacks=[guardar_si_mejora],
    )
    best = study.best_params
    print(f"\n  Mejor F1-CV (penalizado): {study.best_value:.4f}")

    # Evaluar en test
    cw = {0: best['w0'], 1: best['w1'], 2: best['w2']}
    sw = compute_sample_weight(cw, y_train)
    xgb_opt = XGBClassifier(
        n_estimators=best['n_estimators'], max_depth=best['max_depth'],
        learning_rate=best['learning_rate'], subsample=best['subsample'],
        colsample_bytree=best['colsample_bytree'], colsample_bylevel=best['colsample_bylevel'],
        reg_alpha=best['reg_alpha'], reg_lambda=best['reg_lambda'],
        min_child_weight=best['min_child_weight'], gamma=best['gamma'],
        random_state=RANDOM_SEED, n_jobs=-1, eval_metric='mlogloss', verbosity=0,
    )
    xgb_opt.fit(X_train, y_train, sample_weight=sw)
    preds = xgb_opt.predict(X_test)
    f1_test = f1_score(y_test, preds, average='weighted')
    acc_test = (preds == y_test).mean()

    mejora = f1_test - base_f1
    print(f"\n  F1 test={f1_test:.4f}  Acc={acc_test:.4f}  (vs baseline 25feat: {mejora:+.4f})")
    print()
    print(classification_report(y_test, preds, target_names=['Local', 'Empate', 'Visitante']))

    resultado = {
        'best_params': best, 'f1_cv_penalizado': study.best_value,
        'f1_test': f1_test, 'acc_test': acc_test,
        'baseline_f1_test': base_f1, 'mejora': mejora,
        'features': feats, 'n_features': len(feats),
    }
    with open(RUTA_JSON, 'w') as f:
        json.dump(resultado, f, indent=2)
    print(f"  JSON guardado: {os.path.abspath(RUTA_JSON)}")

    if f1_test > base_f1:
        print("\n  *** MEJORA ENCONTRADA — copiar en config.py: ***")
    else:
        print("\n  (no supera el baseline — mantener params actuales)")

    print("\n" + "=" * 65)
    print("  PARAMS PARA config.py (XGB 25 features):")
    print("=" * 65)
    print(f"""
PESOS_XGB = {{
    0: {best['w0']:.4f},
    1: {best['w1']:.4f},
    2: {best['w2']:.4f},
}}
PARAMS_XGB = {{
    'n_estimators':      {best['n_estimators']},
    'max_depth':         {best['max_depth']},
    'learning_rate':     {best['learning_rate']:.5f},
    'subsample':         {best['subsample']:.4f},
    'colsample_bytree':  {best['colsample_bytree']:.4f},
    'colsample_bylevel': {best['colsample_bylevel']:.4f},
    'reg_alpha':         {best['reg_alpha']:.4f},
    'reg_lambda':        {best['reg_lambda']:.4f},
    'min_child_weight':  {best['min_child_weight']},
    'gamma':             {best['gamma']:.4f},
    'random_state':      RANDOM_SEED,
    'n_jobs':            -1,
    'eval_metric':       'mlogloss',
}}
""")
    return resultado


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=150)
    args = parser.parse_args()
    main(args.trials)
