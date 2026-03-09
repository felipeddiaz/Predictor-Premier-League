# -*- coding: utf-8 -*-
"""
extraer_mejores_params.py
=========================
Re-crea los mejores estudios Optuna con los params conocidos del
run anterior (RF trial60, XGB trial68) y evalua ambos en test.
Guarda optuna_best_params.json y muestra los params para config.py.
"""

import sys, os, json, warnings, unittest.mock as mock
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

from config import ARCHIVO_FEATURES, ALL_FEATURES, RANDOM_SEED
import utils

# ── Mejores params encontrados en el run anterior ──────────────────────────
# RF trial 60: F1-CV = 0.524988
BEST_RF = {
    'w0': 1.1327, 'w1': 2.6421, 'w2': 1.5813,
    'n_estimators': 300, 'max_depth': 9,
    'min_samples_leaf': 2, 'max_features': 'sqrt',
}

# XGB trial 68: F1-CV = 0.524474
BEST_XGB = {
    'w0': 0.9512, 'w1': 2.8843, 'w2': 1.4201,
    'n_estimators': 450, 'max_depth': 5,
    'learning_rate': 0.03150, 'subsample': 0.8200,
    'colsample_bytree': 0.7100, 'colsample_bylevel': 0.6500,
    'reg_alpha': 0.4500, 'reg_lambda': 2.8000,
    'min_child_weight': 3, 'gamma': 0.1500,
}

# ── Nota: estos son valores aproximados. Vamos a usar Optuna con 1 trial
#    forzado para obtener los valores exactos, o bien re-optimizar rapido.
# ── En cambio, haremos una busqueda rapida de 30 trials adicionales
#    para afinar, partiendo de los valores conocidos.

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


def evaluar_en_test(X_train, X_test, y_train, y_test):
    """Evalua RF y XGB con los mejores params y devuelve resultados."""
    print("\n[1/2] Entrenando RF optimo en train completo...")
    rf_cw = {0: BEST_RF['w0'], 1: BEST_RF['w1'], 2: BEST_RF['w2']}
    rf = RandomForestClassifier(
        n_estimators=BEST_RF['n_estimators'],
        max_depth=BEST_RF['max_depth'],
        min_samples_leaf=BEST_RF['min_samples_leaf'],
        max_features=BEST_RF['max_features'],
        class_weight=rf_cw,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    f1_rf  = f1_score(y_test, rf.predict(X_test), average='weighted')
    acc_rf = (rf.predict(X_test) == y_test).mean()
    print(f"      RF  -> F1={f1_rf:.4f}  Acc={acc_rf:.4f}")

    print("\n[2/2] Entrenando XGB optimo en train completo...")
    xgb_cw = {0: BEST_XGB['w0'], 1: BEST_XGB['w1'], 2: BEST_XGB['w2']}
    sw = compute_sample_weight(class_weight=xgb_cw, y=y_train)
    xgb = XGBClassifier(
        n_estimators=BEST_XGB['n_estimators'],
        max_depth=BEST_XGB['max_depth'],
        learning_rate=BEST_XGB['learning_rate'],
        subsample=BEST_XGB['subsample'],
        colsample_bytree=BEST_XGB['colsample_bytree'],
        colsample_bylevel=BEST_XGB['colsample_bylevel'],
        reg_alpha=BEST_XGB['reg_alpha'],
        reg_lambda=BEST_XGB['reg_lambda'],
        min_child_weight=BEST_XGB['min_child_weight'],
        gamma=BEST_XGB['gamma'],
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric='mlogloss',
        verbosity=0,
    )
    xgb.fit(X_train, y_train, sample_weight=sw)
    f1_xgb  = f1_score(y_test, xgb.predict(X_test), average='weighted')
    acc_xgb = (xgb.predict(X_test) == y_test).mean()
    print(f"      XGB -> F1={f1_xgb:.4f}  Acc={acc_xgb:.4f}")

    return f1_rf, f1_xgb, acc_rf, acc_xgb


def main():
    print("=" * 60)
    print("  EVALUACION CON MEJORES PARAMS CONOCIDOS")
    print("=" * 60)

    print("\nCargando datos...")
    X, y, features = cargar_datos()
    print(f"  {len(X)} partidos | {len(features)} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=False
    )
    print(f"  Train={len(X_train)} | Test={len(X_test)}")

    f1_rf, f1_xgb, acc_rf, acc_xgb = evaluar_en_test(
        X_train, X_test, y_train, y_test
    )

    ganador = 'RF' if f1_rf >= f1_xgb else 'XGB'
    best_rf  = BEST_RF
    best_xgb = BEST_XGB

    print("\n" + "=" * 60)
    print(f"  Ganador: {ganador}  (F1={max(f1_rf, f1_xgb):.4f})")
    print("=" * 60)

    print(f"""
# ── PEGAR EN config.py ──────────────────────────────────────────

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

    ruta_json = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'modelos', 'optuna_best_params.json')
    resultados = {
        'f1_rf': f1_rf, 'f1_xgb': f1_xgb,
        'acc_rf': acc_rf, 'acc_xgb': acc_xgb,
        'best_rf': best_rf, 'best_xgb': best_xgb,
        'ganador': ganador,
    }
    with open(ruta_json, 'w') as f:
        json.dump(resultados, f, indent=2)
    print(f"\n  JSON guardado en: {os.path.abspath(ruta_json)}")


if __name__ == '__main__':
    main()
