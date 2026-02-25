# -*- coding: utf-8 -*-
"""
optimizar_xgb_optuna.py
=======================
Busca los mejores hiperparámetros para XGBoost usando Optuna (200 trials).

Optimiza SIMULTÁNEAMENTE:
  - Hiperparámetros del modelo (n_estimators, max_depth, learning_rate, etc.)
  - Pesos de clase (peso_local, peso_empate, peso_visitante)

Métrica objetivo: F1-weighted sobre TimeSeriesSplit(n_splits=5)
Split de evaluación final: 85/15 temporal (igual que 02_entrenar_modelo.py)

Al terminar imprime el bloque exacto para copiar a config.py.

Uso:
    venv/Scripts/python.exe herramientas/optimizar_xgb_optuna.py

Tiempo estimado: ~20 minutos (200 trials × ~6s/trial)
"""

import sys
import os
import time
import warnings

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import (
    ARCHIVO_FEATURES,
    ALL_FEATURES,
    FEATURES_ROLLING_EXTRA,
    TEST_SIZE,
    ROLLING_WINDOW,
)
from utils import (
    agregar_xg_rolling,
    agregar_features_tabla,
    agregar_features_cuotas_derivadas,
    agregar_features_asian_handicap,
    agregar_features_rolling_extra,
)

N_TRIALS    = 200
N_CV_SPLITS = 5
SEED        = 42

# ============================================================================
# PREPARAR DATOS
# ============================================================================

def preparar_datos():
    print("=" * 70)
    print("CARGANDO Y PREPARANDO DATOS")
    print("=" * 70)

    t0 = time.time()
    df = pd.read_csv(ARCHIVO_FEATURES)
    df = agregar_xg_rolling(df)
    df = agregar_features_tabla(df)
    df = agregar_features_cuotas_derivadas(df)
    df = agregar_features_asian_handicap(df)
    df = agregar_features_rolling_extra(df)

    features = [f for f in ALL_FEATURES if f in df.columns]
    print(f"\n✅ Datos listos en {time.time()-t0:.1f}s")
    print(f"   Partidos : {len(df)}")
    print(f"   Features : {len(features)}")
    print(f"   Incluye  : {[f for f in FEATURES_ROLLING_EXTRA if f in features]}")

    X = df[features].fillna(0)
    y = df['FTR_numeric']

    # Split igual que 02_entrenar_modelo.py
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False
    )
    fecha_corte = df['Date'].iloc[len(X_tr)] if 'Date' in df.columns else '?'
    print(f"\n   Train : {len(X_tr)} partidos")
    print(f"   Test  : {len(X_te)} partidos (desde {fecha_corte})")

    return X_tr, X_te, y_tr, y_te, features


# ============================================================================
# FUNCIÓN OBJETIVO OPTUNA
# ============================================================================

def make_objective(X_tr, y_tr, n_splits):
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        # --- Hiperparámetros XGBoost ---
        params = {
            'n_estimators'     : trial.suggest_int  ('n_estimators',      100,  800),
            'max_depth'        : trial.suggest_int  ('max_depth',           3,   10),
            'learning_rate'    : trial.suggest_float('learning_rate',   0.005, 0.20, log=True),
            'subsample'        : trial.suggest_float('subsample',        0.50,  1.0),
            'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.40,  1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel',0.40,  1.0),
            'reg_alpha'        : trial.suggest_float('reg_alpha',        0.0,   5.0),
            'reg_lambda'       : trial.suggest_float('reg_lambda',       0.1,   5.0),
            'min_child_weight' : trial.suggest_int  ('min_child_weight',   1,   15),
            'gamma'            : trial.suggest_float('gamma',            0.0,   2.0),
            'random_state': SEED,
            'n_jobs': -1,
            'eval_metric': 'mlogloss',
        }

        # --- Pesos de clase (optimizados junto con los hiperparámetros) ---
        w_local     = trial.suggest_float('peso_local',     0.8, 2.5)
        w_empate    = trial.suggest_float('peso_empate',    1.5, 6.0)
        w_visitante = trial.suggest_float('peso_visitante', 0.8, 3.0)
        pesos = {0: w_local, 1: w_empate, 2: w_visitante}

        # --- Validación cruzada temporal (5 folds) ---
        fold_scores = []
        for tr_idx, val_idx in tscv.split(X_tr):
            X_f_tr  = X_tr.iloc[tr_idx]
            X_f_val = X_tr.iloc[val_idx]
            y_f_tr  = y_tr.iloc[tr_idx]
            y_f_val = y_tr.iloc[val_idx]

            sw = compute_sample_weight(class_weight=pesos, y=y_f_tr)
            m  = XGBClassifier(**params)
            m.fit(X_f_tr, y_f_tr, sample_weight=sw)

            score = f1_score(y_f_val, m.predict(X_f_val), average='weighted')
            fold_scores.append(score)

        return float(np.mean(fold_scores))

    return objective


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("  OPTIMIZACIÓN XGBoost CON OPTUNA")
    print(f"  {N_TRIALS} trials  |  CV{N_CV_SPLITS} temporal  |  seed={SEED}")
    print("=" * 70 + "\n")

    # 1. Datos
    X_tr, X_te, y_tr, y_te, features = preparar_datos()

    # 2. Estudio Optuna
    print(f"\n{'=' * 70}")
    print(f"OPTUNA — {N_TRIALS} TRIALS (puede tardar ~20 min)")
    print(f"{'=' * 70}")
    print("Progreso:\n")

    sampler = TPESampler(seed=SEED)
    study   = optuna.create_study(direction='maximize', sampler=sampler)

    t_start = time.time()
    study.optimize(
        make_objective(X_tr, y_tr, N_CV_SPLITS),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )
    elapsed = time.time() - t_start

    best      = study.best_params
    best_cv   = study.best_value
    best_trial= study.best_trial.number

    print(f"\n✅ Optuna finalizado en {elapsed/60:.1f} min")
    print(f"   Mejor trial    : #{best_trial}")
    print(f"   Mejor CV5 F1   : {best_cv:.4f}")

    # 3. Evaluar en test set con los mejores params
    print(f"\n{'=' * 70}")
    print("EVALUACIÓN EN TEST SET (85/15 temporal)")
    print(f"{'=' * 70}")

    model_params = {k: v for k, v in best.items()
                    if k not in ('peso_local', 'peso_empate', 'peso_visitante')}
    model_params.update({'random_state': SEED, 'n_jobs': -1, 'eval_metric': 'mlogloss'})

    pesos_best = {
        0: best['peso_local'],
        1: best['peso_empate'],
        2: best['peso_visitante'],
    }

    sw_tr = compute_sample_weight(class_weight=pesos_best, y=y_tr)
    best_model = XGBClassifier(**model_params)
    best_model.fit(X_tr, y_tr, sample_weight=sw_tr)

    pred    = best_model.predict(X_te)
    f1_test = f1_score(y_te, pred, average='weighted')
    acc_test= (pred == y_te).mean()

    print(f"\n   F1 test (15%)  : {f1_test:.4f}")
    print(f"   Accuracy       : {acc_test:.2%}")
    print(f"   Objetivo       : 0.5460  {'✅ SUPERADO' if f1_test > 0.5460 else '❌ no superado'}")
    print(f"   Referencia     : 0.5726  {'✅ MEJORADO' if f1_test > 0.5726 else '(no mejorado aun)'}")

    print(f"\n   Reporte por clase:")
    print(classification_report(y_te, pred,
                                 target_names=['Local', 'Empate', 'Visitante'],
                                 digits=4))

    # Top 15 features por importancia
    imp = sorted(zip(features, best_model.feature_importances_),
                 key=lambda x: -x[1])
    print("   Top 10 features:")
    for fname, fval in imp[:10]:
        print(f"     {fname:<35} {fval:.4f}")

    # 4. Bloque para copiar a config.py
    print(f"\n{'=' * 70}")
    print("COPIAR A config.py SI LOS RESULTADOS SON BUENOS:")
    print(f"{'=' * 70}")
    print(f"""
# Optimizados por Optuna ({N_TRIALS} trials, CV{N_CV_SPLITS}, F1_test={f1_test:.4f})
PESOS_XGB = {{
    0: {best['peso_local']:.4f},   # Local
    1: {best['peso_empate']:.4f},  # Empate
    2: {best['peso_visitante']:.4f},   # Visitante
}}

PARAMS_XGB = {{
    'n_estimators'     : {model_params['n_estimators']},
    'max_depth'        : {model_params['max_depth']},
    'learning_rate'    : {model_params['learning_rate']:.5f},
    'subsample'        : {model_params['subsample']:.4f},
    'colsample_bytree' : {model_params['colsample_bytree']:.4f},
    'colsample_bylevel': {model_params['colsample_bylevel']:.4f},
    'reg_alpha'        : {model_params['reg_alpha']:.4f},
    'reg_lambda'       : {model_params['reg_lambda']:.4f},
    'min_child_weight' : {model_params['min_child_weight']},
    'gamma'            : {model_params['gamma']:.4f},
    'random_state'     : 42,
    'n_jobs'           : -1,
    'eval_metric'      : 'mlogloss',
}}
""")
    print(f"   CV5 optimo : {best_cv:.4f}")
    print(f"   F1 test    : {f1_test:.4f}")
    print(f"   Trials     : {N_TRIALS}")
    print(f"   Tiempo     : {elapsed/60:.1f} min")
    print()


if __name__ == '__main__':
    main()
