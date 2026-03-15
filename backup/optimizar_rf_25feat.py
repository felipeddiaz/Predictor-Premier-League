# -*- coding: utf-8 -*-
"""
optimizar_rf_25feat.py
======================
Optimiza Random Forest sobre las 25 features seleccionadas por permutation importance.
Usa Optuna TPESampler + penalizacion por sobreajuste en el fold mas reciente.

- 150 trials (warm start con params actuales de config.py)
- Split 80/20 temporal
- Guarda JSON con mejores params en modelos/optuna_rf_25feat.json
- Imprime bloque listo para copiar en config.py
"""

import sys, os, json, warnings, unittest.mock as mock
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from config import ARCHIVO_FEATURES, FEATURES_SELECCIONADAS, RANDOM_SEED, PARAMS_OPTIMOS, PESOS_OPTIMOS
import utils

RUTA_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'modelos', 'optuna_rf_25feat.json')

_P = PARAMS_OPTIMOS
_W = PESOS_OPTIMOS


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
    X = df[feats].fillna(0)
    y = df['FTR_numeric']
    return X, y, feats


def objetivo(trial, X_train, y_train, tscv):
    """Maximiza F1-CV con penalizacion por sobreajuste temporal."""
    w0 = trial.suggest_float('w0', 0.5, 4.0)
    w1 = trial.suggest_float('w1', 1.0, 6.0)
    w2 = trial.suggest_float('w2', 0.5, 4.0)

    n_estimators     = trial.suggest_int('n_estimators', 100, 800, step=50)
    max_depth        = trial.suggest_int('max_depth', 3, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 15)
    min_samples_split= trial.suggest_int('min_samples_split', 2, 20)
    max_features     = trial.suggest_categorical('max_features', ['sqrt', 'log2'])

    cw = {0: w0, 1: w1, 2: w2}
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=max_features,
        class_weight=cw,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    scores = []
    for tr, val in tscv.split(X_train):
        model.fit(X_train.iloc[tr], y_train.iloc[tr])
        scores.append(f1_score(y_train.iloc[val], model.predict(X_train.iloc[val]),
                               average='weighted'))

    mean_cv = float(np.mean(scores))
    last_fold = scores[-1]
    gap = mean_cv - last_fold
    penalizacion = max(0.0, gap - 0.015) * 2.0
    return mean_cv - penalizacion


def warm_start_params():
    return {
        'w0': _W[0], 'w1': _W[1], 'w2': _W[2],
        'n_estimators':      _P['n_estimators'],
        'max_depth':         _P['max_depth'],
        'min_samples_leaf':  _P['min_samples_leaf'],
        'min_samples_split': _P.get('min_samples_split', 2),
        'max_features':      _P.get('max_features', 'sqrt'),
    }


def main(n_trials=150):
    print("=" * 65)
    print(f"  RF OPTUNA — 25 features ({n_trials} trials)")
    print(f"  Espacio amplio + penalizacion sobreajuste temporal")
    print("=" * 65)

    X, y, feats = cargar_datos()
    print(f"\n  Features usadas: {len(feats)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
    tscv = TimeSeriesSplit(n_splits=5)
    print(f"  Train={len(X_train)} | Test={len(X_test)}")

    # Baseline con params actuales
    rf_base = RandomForestClassifier(**_P)
    rf_base.fit(X_train, y_train)
    base_f1 = f1_score(y_test, rf_base.predict(X_test), average='weighted')
    print(f"  Baseline (params actuales): F1-test = {base_f1:.4f}\n")

    sampler = TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.enqueue_trial(warm_start_params())

    def guardar_si_mejora(study, trial):
        if trial.number == study.best_trial.number:
            with open(RUTA_JSON, 'w') as f:
                json.dump({'best_params': study.best_params,
                           'f1_cv_penalizado': study.best_value,
                           'baseline_f1_test': base_f1,
                           'features': feats, 'n_features': len(feats),
                           'trials_completados': trial.number + 1}, f, indent=2)

    print(f"Optimizando {n_trials} trials...")
    study.optimize(
        lambda t: objetivo(t, X_train, y_train, tscv),
        n_trials=n_trials, show_progress_bar=True,
        callbacks=[guardar_si_mejora],
    )
    best = study.best_params
    print(f"\n  Mejor F1-CV (penalizado): {study.best_value:.4f}")

    # Evaluar en test
    cw = {0: best['w0'], 1: best['w1'], 2: best['w2']}
    rf_opt = RandomForestClassifier(
        n_estimators=best['n_estimators'],
        max_depth=best['max_depth'],
        min_samples_leaf=best['min_samples_leaf'],
        min_samples_split=best['min_samples_split'],
        max_features=best['max_features'],
        class_weight=cw,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf_opt.fit(X_train, y_train)
    preds = rf_opt.predict(X_test)
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
        'features': feats, 'n_features': len(feats),
    }
    with open(RUTA_JSON, 'w') as f:
        json.dump(resultado, f, indent=2)
    print(f"  JSON guardado: {os.path.abspath(RUTA_JSON)}")

    if f1_test > base_f1:
        print("\n  *** MEJORA ENCONTRADA — copiar en config.py: ***")
    else:
        print("\n  (no supera el baseline — revisar si mejora en ensemble)")

    print("\n" + "=" * 65)
    print("  PARAMS PARA config.py (RF 25 features):")
    print("=" * 65)
    print(f"""
PESOS_OPTIMOS = {{
    0: {best['w0']:.4f},
    1: {best['w1']:.4f},
    2: {best['w2']:.4f},
}}
PARAMS_OPTIMOS = {{
    'n_estimators':      {best['n_estimators']},
    'max_depth':         {best['max_depth']},
    'min_samples_leaf':  {best['min_samples_leaf']},
    'min_samples_split': {best['min_samples_split']},
    'max_features':      '{best['max_features']}',
    'class_weight':      PESOS_OPTIMOS,
    'random_state':      RANDOM_SEED,
    'n_jobs':            -1,
}}
""")
    return resultado


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=150)
    args = parser.parse_args()
    main(args.trials)
