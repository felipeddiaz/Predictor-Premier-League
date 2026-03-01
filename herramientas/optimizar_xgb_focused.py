# -*- coding: utf-8 -*-
"""
optimizar_xgb_focused.py
========================
Re-optimiza XGBoost sobre las 27 features con un espacio de busqueda ESTRECHO
centrado en los parametros actuales de PARAMS_XGB (que dan F1-test=0.5187).

Diferencias respecto a optimizar_xgb_27feat.py:
- Espacio de busqueda ~50% mas estrecho (menos sobreajuste al CV)
- Penalizacion cuando F1-CV >> F1 de la ultima particion (proxy de overfitting)
- 100+ trials por defecto
- Warm start: el primer trial usa los params actuales de config.py
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

from config import ARCHIVO_FEATURES, ALL_FEATURES, RANDOM_SEED, PARAMS_XGB, PESOS_XGB
import utils

RUTA_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'modelos', 'optuna_xgb_focused.json')

FEATURES_OPTIMAS = [
    'Home_Advantage_Prob', 'HT_xG_Avg', 'Market_Confidence', 'AT_GoalsFor5',
    'AT_xGA_Avg', 'B365CH', 'HT_Form_W', 'Position_Diff', 'Prob_Move_D',
    'AT_Points', 'Match_Type', 'HT_Points', 'HT_Pressure', 'AT_Pressure',
    'Position_Diff_Weighted', 'Season_Progress', 'xG_Total', 'Prob_A',
    'AT_AwayWinRate5', 'H2H_Total_Goals_Avg', 'H2H_Away_Goals_Avg',
    'H2H_Home_Win_Rate', 'HT_GoalsFor5', 'AT_WinRate5', 'AT_AwayGoals5',
    'AT_AvgGoals', 'AT_Form_W',
]

# Params actuales que dan F1-test=0.5187 — centro del espacio de busqueda
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
    all_f = [f for f in ALL_FEATURES if f in df.columns]
    feats = [f for f in all_f if f in FEATURES_OPTIMAS]
    X = df[feats].fillna(0)
    y = df['FTR_numeric']
    return X, y, feats


def objetivo(trial, X_train, y_train, tscv):
    """
    Optimiza F1-CV con penalizacion por sobreajuste.
    Penalizacion: si el score de la ultima particion temporal es mucho menor
    que el promedio, el trial es penalizado (indica sobreajuste al pasado).
    """
    # Espacio estrecho centrado en los params actuales
    w0 = trial.suggest_float('w0', max(0.5, _W[0] * 0.6), _W[0] * 1.6)
    w1 = trial.suggest_float('w1', max(1.0, _W[1] * 0.6), _W[1] * 1.6)
    w2 = trial.suggest_float('w2', max(0.5, _W[2] * 0.6), _W[2] * 1.6)

    n_estimators      = trial.suggest_int('n_estimators',
                                           max(200, _P['n_estimators'] - 150),
                                           _P['n_estimators'] + 150, step=50)
    max_depth         = trial.suggest_int('max_depth',
                                           max(3, _P['max_depth'] - 2),
                                           min(10, _P['max_depth'] + 2))
    learning_rate     = trial.suggest_float('learning_rate',
                                             _P['learning_rate'] * 0.5,
                                             _P['learning_rate'] * 2.5, log=True)
    subsample         = trial.suggest_float('subsample',
                                             max(0.55, _P['subsample'] - 0.20),
                                             min(1.0,  _P['subsample'] + 0.15))
    colsample_bytree  = trial.suggest_float('colsample_bytree',
                                             max(0.45, _P['colsample_bytree'] - 0.20),
                                             min(1.0,  _P['colsample_bytree'] + 0.20))
    colsample_bylevel = trial.suggest_float('colsample_bylevel',
                                             max(0.45, _P['colsample_bylevel'] - 0.20),
                                             min(1.0,  _P['colsample_bylevel'] + 0.20))
    reg_alpha         = trial.suggest_float('reg_alpha',
                                             max(0.0, _P['reg_alpha'] - 0.60),
                                             _P['reg_alpha'] + 0.80)
    reg_lambda        = trial.suggest_float('reg_lambda',
                                             max(0.3, _P['reg_lambda'] - 0.80),
                                             _P['reg_lambda'] + 1.20)
    min_child_weight  = trial.suggest_int('min_child_weight',
                                           max(1, _P['min_child_weight'] - 3),
                                           _P['min_child_weight'] + 3)
    gamma             = trial.suggest_float('gamma',
                                             max(0.0, _P['gamma'] - 0.05),
                                             _P['gamma'] + 0.15)

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
    last_fold = scores[-1]  # particion mas reciente = mejor proxy del test real

    # Penalizacion si la ultima particion esta muy por debajo del promedio
    # (sugiere que el modelo generaliza mal hacia el futuro)
    gap = mean_cv - last_fold
    penalizacion = max(0.0, gap - 0.015) * 2.0  # tolera gap de 1.5%, penaliza el resto

    return mean_cv - penalizacion


def warm_start_params():
    """Devuelve los params actuales de config.py como punto de partida."""
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


def main(n_trials=85):
    print("=" * 60)
    print(f"  XGB FOCUSED OPTUNA — 27 features ({n_trials} trials)")
    print(f"  Espacio estrecho + penalizacion overfitting")
    print("=" * 60)

    X, y, feats = cargar_datos()
    print(f"\n  Features usadas: {len(feats)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
    tscv = TimeSeriesSplit(n_splits=5)
    print(f"  Train={len(X_train)} | Test={len(X_test)}")

    # Evaluar baseline (params actuales) en test
    sw_base = compute_sample_weight(_W, y_train)
    base_model = XGBClassifier(**{k: v for k, v in _P.items()},
                                verbosity=0)
    base_model.fit(X_train, y_train, sample_weight=sw_base)
    base_f1 = f1_score(y_test, base_model.predict(X_test), average='weighted')
    print(f"  Baseline (params actuales): F1-test = {base_f1:.4f}")
    print()

    sampler = TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction='maximize', sampler=sampler)

    # Warm start con los params actuales
    study.enqueue_trial(warm_start_params())

    # Callback: guarda en disco cada vez que se encuentra un nuevo mejor trial
    def guardar_si_mejora(study, trial):
        if trial.number == study.best_trial.number:
            with open(RUTA_JSON, 'w') as f:
                json.dump({'best_params': study.best_params,
                           'f1_cv_penalizado': study.best_value,
                           'baseline_f1_test': base_f1, 'features': feats,
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
    print(f"\n  F1 test={f1_test:.4f}  Acc={acc_test:.4f}  (vs baseline: {mejora:+.4f})")
    print()
    print(classification_report(y_test, preds, target_names=['Local', 'Empate', 'Visitante']))

    resultado = {
        'best_params': best, 'f1_cv_penalizado': study.best_value,
        'f1_test': f1_test, 'acc_test': acc_test,
        'baseline_f1_test': base_f1, 'mejora': mejora,
        'features': feats,
    }
    with open(RUTA_JSON, 'w') as f:
        json.dump(resultado, f, indent=2)
    print(f"  JSON guardado: {os.path.abspath(RUTA_JSON)}")

    if f1_test > base_f1:
        print("\n  *** MEJORA ENCONTRADA — copiar en config.py: ***")
    else:
        print("\n  (no supera el baseline — mantener params actuales)")

    print("\n" + "=" * 60)
    print("  PARAMS PARA config.py:")
    print("=" * 60)
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
    parser.add_argument('--trials', type=int, default=100)
    args = parser.parse_args()
    main(args.trials)
