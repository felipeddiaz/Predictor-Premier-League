# -*- coding: utf-8 -*-
"""
optimizar_xgb_optuna.py
=======================
Busca los mejores hiperparámetros para XGBoost usando Optuna (200 trials).

Optimiza SIMULTÁNEAMENTE:
  - Hiperparámetros del modelo (n_estimators, max_depth, learning_rate, etc.)
  - Pesos de clase (peso_local, peso_empate, peso_visitante)

Métrica objetivo: Walk-Forward Validation
  - 3 ventanas deslizantes donde el fold de validación son siempre
    los datos MÁS RECIENTES de cada ventana (mismo tipo de datos que el test real)
  - Esto evita el problema del CV5 clásico que optimiza para folds internos
    que no representan bien el período de test real (ene-2025 en adelante)

Split de evaluación final: 85/15 temporal (igual que 02_entrenar_modelo.py)

Al terminar imprime el bloque exacto para copiar a config.py.

Uso:
    venv/Scripts/python.exe herramientas/optimizar_xgb_optuna.py

Tiempo estimado: ~20-30 minutos (200 trials × ~6s/trial × 3 folds WF)
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import (
    ARCHIVO_FEATURES,
    ALL_FEATURES,
    FEATURES_ROLLING_EXTRA,
    TEST_SIZE,
)
from utils import (
    agregar_xg_rolling,
    agregar_features_tabla,
    agregar_features_cuotas_derivadas,
    agregar_features_asian_handicap,
    agregar_features_rolling_extra,
)

N_TRIALS   = 200
SEED       = 42

# Walk-forward: tamaños de validación (% del training set)
# Cada fold valida sobre los últimos WF_VAL_SIZE datos del bloque de training
# Ejemplo con 3 folds y WF_VAL_SIZE=0.15:
#   fold 1: train=primeros 55%  val=siguientes 15%
#   fold 2: train=primeros 70%  val=siguientes 15%
#   fold 3: train=primeros 85%  val=últimos 15%   ← más cercano al test real
WF_N_FOLDS   = 3
WF_VAL_SIZE  = 0.15   # tamaño de cada ventana de validación

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
    print(f"   Rolling  : {[f for f in FEATURES_ROLLING_EXTRA if f in features]}")

    X = df[features].fillna(0)
    y = df['FTR_numeric']

    # Split igual que 02_entrenar_modelo.py (85/15)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False
    )
    fecha_corte = df['Date'].iloc[len(X_tr)] if 'Date' in df.columns else '?'
    print(f"\n   Train : {len(X_tr)} partidos")
    print(f"   Test  : {len(X_te)} partidos (desde {fecha_corte})")

    return X_tr, X_te, y_tr, y_te, features, df


# ============================================================================
# WALK-FORWARD SPLITS
# ============================================================================

def make_wf_splits(n, n_folds, val_size):
    """
    Genera índices para walk-forward validation.

    Cada fold usa como validación un bloque de tamaño val_size*n al final
    de un sub-conjunto creciente del training data.

    fold 1: train=[0 .. n*(1 - n_folds*val_size)]    val=[n*(1-n_folds*v) .. n*(1-(n_folds-1)*v)]
    fold 2: train=[0 .. n*(1 - (n_folds-1)*val_size)] val=[...]
    ...
    fold k: train=[0 .. n*(1 - val_size)]             val=[n*(1-val_size) .. n]  ← más reciente

    El último fold (más reciente) tiene el mayor peso porque replica
    exactamente el tipo de predicción que haremos en producción.
    """
    splits = []
    val_n = int(n * val_size)
    for fold in range(n_folds):
        # El fold más reciente es el último
        val_end   = n - fold * val_n
        val_start = val_end - val_n
        if val_start <= 0:
            continue
        tr_idx  = list(range(0, val_start))
        val_idx = list(range(val_start, val_end))
        splits.append((tr_idx, val_idx))

    # Invertir para que el fold más reciente sea el último
    splits.reverse()
    return splits


# ============================================================================
# FUNCIÓN OBJETIVO OPTUNA — WALK-FORWARD
# ============================================================================

def make_objective(X_tr, y_tr, wf_splits):
    """
    La métrica de Optuna es el F1 promedio ponderado de los 3 folds walk-forward,
    con mayor peso al fold más reciente (que replica mejor el test real).
    """
    # Pesos de los folds: el más reciente vale el doble
    # [0.25, 0.33, 0.42] para 3 folds → énfasis en el período más reciente
    n = len(wf_splits)
    fold_weights = np.array([i + 1 for i in range(n)], dtype=float)
    fold_weights /= fold_weights.sum()

    def objective(trial):
        # --- Hiperparámetros XGBoost ---
        # Espacio de búsqueda ajustado: excluye zonas extremas que Optuna
        # tiende a explorar y que producen overfitting al CV
        params = {
            'n_estimators'     : trial.suggest_int  ('n_estimators',      200,  800),
            'max_depth'        : trial.suggest_int  ('max_depth',           4,    9),
            'learning_rate'    : trial.suggest_float('learning_rate',    0.01, 0.15, log=True),
            'subsample'        : trial.suggest_float('subsample',        0.60,  1.0),
            'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.55,  1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel',0.55,  1.0),
            'reg_alpha'        : trial.suggest_float('reg_alpha',        0.0,   3.0),
            'reg_lambda'       : trial.suggest_float('reg_lambda',       0.1,   4.0),
            'min_child_weight' : trial.suggest_int  ('min_child_weight',   1,    8),
            'gamma'            : trial.suggest_float('gamma',            0.0,   1.0),
            'random_state': SEED,
            'n_jobs': -1,
            'eval_metric': 'mlogloss',
        }

        # --- Pesos de clase ---
        w_local     = trial.suggest_float('peso_local',     0.8, 2.5)
        w_empate    = trial.suggest_float('peso_empate',    1.5, 5.0)
        w_visitante = trial.suggest_float('peso_visitante', 0.8, 2.8)
        pesos = {0: w_local, 1: w_empate, 2: w_visitante}

        # --- Walk-forward validation ---
        fold_scores = []
        for tr_idx, val_idx in wf_splits:
            X_f_tr  = X_tr.iloc[tr_idx]
            X_f_val = X_tr.iloc[val_idx]
            y_f_tr  = y_tr.iloc[tr_idx]
            y_f_val = y_tr.iloc[val_idx]

            sw = compute_sample_weight(class_weight=pesos, y=y_f_tr)
            m  = XGBClassifier(**params)
            m.fit(X_f_tr, y_f_tr, sample_weight=sw)

            score = f1_score(y_f_val, m.predict(X_f_val), average='weighted')
            fold_scores.append(score)

        # Promedio ponderado: más peso al fold más reciente
        return float(np.average(fold_scores, weights=fold_weights))

    return objective


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("  OPTIMIZACIÓN XGBoost CON OPTUNA — WALK-FORWARD")
    print(f"  {N_TRIALS} trials  |  WF {WF_N_FOLDS} folds  |  seed={SEED}")
    print("=" * 70 + "\n")

    # 1. Datos
    X_tr, X_te, y_tr, y_te, features, df = preparar_datos()

    # 2. Generar splits walk-forward
    wf_splits = make_wf_splits(len(X_tr), WF_N_FOLDS, WF_VAL_SIZE)
    print(f"\nWalk-Forward splits generados:")
    for i, (tr_idx, val_idx) in enumerate(wf_splits):
        peso = (i + 1) / sum(range(1, len(wf_splits) + 1))
        print(f"   Fold {i+1}: train={len(tr_idx)} | val={len(val_idx)} | peso={peso:.2f}"
              f"{'  ← más reciente' if i == len(wf_splits)-1 else ''}")

    # 3. Estudio Optuna
    print(f"\n{'=' * 70}")
    print(f"OPTUNA — {N_TRIALS} TRIALS")
    print(f"{'=' * 70}")
    print("Progreso:\n")

    sampler = TPESampler(seed=SEED)
    study   = optuna.create_study(direction='maximize', sampler=sampler)

    t_start = time.time()
    study.optimize(
        make_objective(X_tr, y_tr, wf_splits),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )
    elapsed = time.time() - t_start

    best       = study.best_params
    best_wf    = study.best_value
    best_trial = study.best_trial.number

    print(f"\n✅ Optuna finalizado en {elapsed/60:.1f} min")
    print(f"   Mejor trial       : #{best_trial}")
    print(f"   Mejor WF F1       : {best_wf:.4f}")

    # 4. Evaluar en test set
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

    pred     = best_model.predict(X_te)
    f1_test  = f1_score(y_te, pred, average='weighted')
    acc_test = (pred == y_te).mean()

    REFERENCIA = 0.5726
    print(f"\n   F1 test (15%)  : {f1_test:.4f}")
    print(f"   Accuracy       : {acc_test:.2%}")
    print(f"   Objetivo       : 0.5460  {'✅ SUPERADO' if f1_test > 0.5460 else '❌ no superado'}")
    print(f"   Referencia     : {REFERENCIA}  {'✅ MEJORADO' if f1_test > REFERENCIA else '(no mejorado)'}")

    print(f"\n   Reporte por clase:")
    print(classification_report(y_te, pred,
                                 target_names=['Local', 'Empate', 'Visitante'],
                                 digits=4))

    # Top 10 features
    imp = sorted(zip(features, best_model.feature_importances_), key=lambda x: -x[1])
    print("   Top 10 features:")
    for fname, fval in imp[:10]:
        print(f"     {fname:<35} {fval:.4f}")

    # 5. Bloque para copiar a config.py
    print(f"\n{'=' * 70}")
    print("COPIAR A config.py SI F1 test MEJORA LA REFERENCIA (0.5726):")
    print(f"{'=' * 70}")
    print(f"""
# Optimizados por Optuna ({N_TRIALS} trials, WF{WF_N_FOLDS}, F1_test={f1_test:.4f})
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
    print(f"   WF F1 optimo : {best_wf:.4f}")
    print(f"   F1 test      : {f1_test:.4f}  "
          f"({'MEJOR que referencia' if f1_test > REFERENCIA else 'NO mejor — no copiar'})")
    print(f"   Trials       : {N_TRIALS}")
    print(f"   Tiempo       : {elapsed/60:.1f} min\n")


if __name__ == '__main__':
    main()
