# -*- coding: utf-8 -*-
"""
optimizar_xgb_27feat.py
=======================
Re-optimiza XGBoost sobre las 27 features seleccionadas por permutation importance.
Guarda los resultados en modelos/optuna_xgb_27feat.json
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

from config import ARCHIVO_FEATURES, ALL_FEATURES, RANDOM_SEED
import utils

RUTA_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'modelos', 'optuna_xgb_27feat.json')

# Las 27 features seleccionadas por permutation importance positiva
FEATURES_OPTIMAS = [
    'Home_Advantage_Prob', 'HT_xG_Avg', 'Market_Confidence', 'AT_GoalsFor5',
    'AT_xGA_Avg', 'B365CH', 'HT_Form_W', 'Position_Diff', 'Prob_Move_D',
    'AT_Points', 'Match_Type', 'HT_Points', 'HT_Pressure', 'AT_Pressure',
    'Position_Diff_Weighted', 'Season_Progress', 'xG_Total', 'Prob_A',
    'AT_AwayWinRate5', 'H2H_Total_Goals_Avg', 'H2H_Away_Goals_Avg',
    'H2H_Home_Win_Rate', 'HT_GoalsFor5', 'AT_WinRate5', 'AT_AwayGoals5',
    'AT_AvgGoals', 'AT_Form_W',
]


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
    w0 = trial.suggest_float('w0', 0.5, 2.5)
    w1 = trial.suggest_float('w1', 1.5, 5.0)
    w2 = trial.suggest_float('w2', 0.8, 2.5)
    n_estimators      = trial.suggest_int('n_estimators', 200, 700, step=50)
    max_depth         = trial.suggest_int('max_depth', 3, 9)
    learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
    subsample         = trial.suggest_float('subsample', 0.60, 1.0)
    colsample_bytree  = trial.suggest_float('colsample_bytree', 0.50, 1.0)
    colsample_bylevel = trial.suggest_float('colsample_bylevel', 0.50, 1.0)
    reg_alpha         = trial.suggest_float('reg_alpha', 0.0, 2.0)
    reg_lambda        = trial.suggest_float('reg_lambda', 0.5, 5.0)
    min_child_weight  = trial.suggest_int('min_child_weight', 1, 10)
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
    return float(np.mean(scores))


def main(n_trials=60):
    print("=" * 60)
    print(f"  XGB OPTUNA — 27 features seleccionadas ({n_trials} trials)")
    print("=" * 60)

    X, y, feats = cargar_datos()
    print(f"\n  Features usadas: {len(feats)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
    tscv = TimeSeriesSplit(n_splits=5)
    print(f"  Train={len(X_train)} | Test={len(X_test)}")

    print(f"\nOptimizando {n_trials} trials...")
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_SEED))
    study.optimize(
        lambda t: objetivo(t, X_train, y_train, tscv),
        n_trials=n_trials, show_progress_bar=True,
    )
    best = study.best_params
    print(f"\n  Mejor F1-CV: {study.best_value:.4f}")

    # Guardar parcial
    with open(RUTA_JSON, 'w') as f:
        json.dump({'best_params': best, 'f1_cv': study.best_value, 'features': feats}, f, indent=2)
    print("  Params guardados (parcial)")

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

    print(f"\n  F1 test={f1_test:.4f}  Acc={acc_test:.4f}")
    print()
    print(classification_report(y_test, preds, target_names=['Local', 'Empate', 'Visitante']))

    resultado = {
        'best_params': best, 'f1_cv': study.best_value,
        'f1_test': f1_test, 'acc_test': acc_test, 'features': feats,
    }
    with open(RUTA_JSON, 'w') as f:
        json.dump(resultado, f, indent=2)
    print(f"  JSON guardado: {os.path.abspath(RUTA_JSON)}")

    print("\n" + "=" * 60)
    print("  COPIAR EN config.py:")
    print("=" * 60)
    print(f"""
FEATURES_SELECCIONADAS = {feats}

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
    parser.add_argument('--trials', type=int, default=60)
    args = parser.parse_args()
    main(args.trials)
