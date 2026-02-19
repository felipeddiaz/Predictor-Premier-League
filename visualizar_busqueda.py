# -*- coding: utf-8 -*-
"""
Optimización de pesos e hiperparámetros con Optuna.

Modos de ejecución (flags principales abajo):
  MODO_SIN_CUOTAS = False, MODO_XGB = False  →  RF  PARAMS_OPTIMOS    (modelo 02, RF con cuotas)
  MODO_SIN_CUOTAS = True,  MODO_XGB = False  →  RF  PARAMS_OPTIMOS_VB (modelo 03, RF sin cuotas)
  MODO_SIN_CUOTAS = False, MODO_XGB = True   →  XGB PARAMS_XGB        (modelo 02, XGBoost con cuotas)

Al terminar imprime el bloque listo para copiar a config.py.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from config import (
    ARCHIVO_FEATURES,
    FEATURES_BASE,
    FEATURES_CUOTAS,
    FEATURES_CUOTAS_DERIVADAS,
    FEATURES_XG,
    FEATURES_H2H,
    FEATURES_H2H_DERIVADAS,
    FEATURES_TABLA,
    ALL_FEATURES,
)
from utils import agregar_xg_rolling, agregar_features_tabla, agregar_features_cuotas_derivadas

# ============================================================================
# FLAGS PRINCIPALES — cambia aquí antes de ejecutar
# ============================================================================

MODO_SIN_CUOTAS = False   # False = datos CON cuotas (modelo 02)
                            # True  = datos SIN cuotas (modelo 03) — solo aplica a RF

MODO_XGB =  True           # False = optimiza Random Forest
                            # True  = optimiza XGBoost (solo con cuotas, modelo 02)
                            # NOTA: MODO_XGB=True ignora MODO_SIN_CUOTAS

N_TRIALS = 150             # Número de trials Optuna (5-8 min con 150)

# ============================================================================
# CARGAR Y PREPARAR DATOS
# ============================================================================

if MODO_XGB:
    modo_label = "XGBOOST CON CUOTAS (modelo 02)"
elif MODO_SIN_CUOTAS:
    modo_label = "RF SIN CUOTAS (modelo 03)"
else:
    modo_label = "RF CON CUOTAS (modelo 02)"

print("=" * 70)
print(f"OPTIMIZACIÓN DE HIPERPARÁMETROS CON OPTUNA")
print(f"Modo: {modo_label}")
print("=" * 70)

df = pd.read_csv(ARCHIVO_FEATURES)
print(f"\n✅ Cargados: {len(df)} partidos")

# Features calculadas en memoria
df = agregar_xg_rolling(df)
df = agregar_features_tabla(df)

if MODO_SIN_CUOTAS and not MODO_XGB:
    # Modo sin cuotas (RF modelo 03): filtrar solo partidos con H2H disponible
    if 'H2H_Available' in df.columns:
        antes = len(df)
        df = df[df['H2H_Available'] == 1].copy()
        print(f"✅ Filtro H2H: {antes} → {len(df)} partidos con historial")
    df = df.reset_index(drop=True)

    # Features sin cuotas ni cuotas derivadas
    all_sin_cuotas = FEATURES_BASE + FEATURES_H2H + FEATURES_H2H_DERIVADAS + FEATURES_XG + FEATURES_TABLA
    features = [f for f in all_sin_cuotas if f in df.columns]

    print(f"\n✅ Features totales: {len(features)} (SIN cuotas)")
    print(f"   • Base:          {len([f for f in FEATURES_BASE if f in features])}")
    print(f"   • H2H:           {len([f for f in FEATURES_H2H if f in features])}")
    print(f"   • H2H derivadas: {len([f for f in FEATURES_H2H_DERIVADAS if f in features])}")
    print(f"   • xG rolling:    {len([f for f in FEATURES_XG if f in features])}")
    print(f"   • Tabla:         {len([f for f in FEATURES_TABLA if f in features])}")

else:
    # Modo con cuotas (RF modelo 02 o XGBoost modelo 02)
    df = agregar_features_cuotas_derivadas(df)
    features = [f for f in ALL_FEATURES if f in df.columns]

    print(f"\n✅ Features totales: {len(features)} (CON cuotas)")
    print(f"   • Base:             {len([f for f in FEATURES_BASE if f in features])}")
    print(f"   • Cuotas raw:       {len([f for f in FEATURES_CUOTAS if f in features])}")
    print(f"   • Cuotas derivadas: {len([f for f in FEATURES_CUOTAS_DERIVADAS if f in features])}")
    print(f"   • xG rolling:       {len([f for f in FEATURES_XG if f in features])}")
    print(f"   • H2H:              {len([f for f in FEATURES_H2H if f in features])}")
    print(f"   • H2H derivadas:    {len([f for f in FEATURES_H2H_DERIVADAS if f in features])}")
    print(f"   • Tabla:            {len([f for f in FEATURES_TABLA if f in features])}")

X = df[features].fillna(0)
y = df['FTR_numeric']

# Split temporal
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"\n✅ Train: {len(X_train)} | Test: {len(X_test)}")

# ============================================================================
# OBJETIVO OPTUNA
# ============================================================================

def objective(trial):
    if MODO_XGB:
        # ── XGBoost con cuotas (modelo 02) ──────────────────────────────────
        # Pesos de clase como sample_weight (igual que en 02_entrenar_modelo.py)
        w_local     = trial.suggest_float('peso_local',     0.5, 2.5)
        w_empate    = trial.suggest_float('peso_empate',    1.0, 5.0)
        w_visitante = trial.suggest_float('peso_visitante', 0.5, 2.5)

        n_estimators      = trial.suggest_int  ('n_estimators',    100, 500)
        max_depth         = trial.suggest_int  ('max_depth',         3,  10)
        learning_rate     = trial.suggest_float('learning_rate',  0.01, 0.30, log=True)
        subsample         = trial.suggest_float('subsample',       0.5,  1.0)
        colsample_bytree  = trial.suggest_float('colsample_bytree',0.5,  1.0)
        reg_alpha         = trial.suggest_float('reg_alpha',       0.0,  2.0)
        reg_lambda        = trial.suggest_float('reg_lambda',      0.5,  3.0)

        sample_weights = compute_sample_weight(
            class_weight={0: w_local, 1: w_empate, 2: w_visitante},
            y=y_train
        )

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss',
        )

        # Cross-val manual para pasar sample_weight por fold
        tscv = TimeSeriesSplit(n_splits=3)
        fold_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            Xf_tr, Xf_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            yf_tr, yf_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            sw_tr = sample_weights[train_idx]
            model.fit(Xf_tr, yf_tr, sample_weight=sw_tr)
            pred = model.predict(Xf_val)
            fold_scores.append(f1_score(yf_val, pred, average='weighted'))
        return np.mean(fold_scores)

    elif MODO_SIN_CUOTAS:
        # ── RF sin cuotas (modelo 03) ────────────────────────────────────────
        # Rangos ajustados: dataset más pequeño (~1346 train)
        w_local     = trial.suggest_float('peso_local',     0.5, 2.0)
        w_empate    = trial.suggest_float('peso_empate',    1.0, 3.5)
        w_visitante = trial.suggest_float('peso_visitante', 0.5, 2.0)

        n_estimators     = trial.suggest_int('n_estimators',     100, 400)
        max_depth        = trial.suggest_int('max_depth',          4,  12)
        min_samples_leaf = trial.suggest_int('min_samples_leaf',   5,  20)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight={0: w_local, 1: w_empate, 2: w_visitante},
            random_state=42,
            n_jobs=-1
        )
        tscv = TimeSeriesSplit(n_splits=3)
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='f1_weighted', n_jobs=-1)
        return scores.mean()

    else:
        # ── RF con cuotas (modelo 02) ────────────────────────────────────────
        w_local     = trial.suggest_float('peso_local',     0.5, 2.5)
        w_empate    = trial.suggest_float('peso_empate',    1.0, 5.0)
        w_visitante = trial.suggest_float('peso_visitante', 0.5, 2.5)

        n_estimators     = trial.suggest_int('n_estimators',     100, 400)
        max_depth        = trial.suggest_int('max_depth',          4,  20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf',   3,  15)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight={0: w_local, 1: w_empate, 2: w_visitante},
            random_state=42,
            n_jobs=-1
        )
        tscv = TimeSeriesSplit(n_splits=3)
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='f1_weighted', n_jobs=-1)
        return scores.mean()

# ============================================================================
# EJECUTAR OPTIMIZACIÓN
# ============================================================================

print("\n" + "=" * 70)
print(f"EJECUTANDO OPTIMIZACIÓN ({N_TRIALS} trials) — modo: {modo_label}")
print("=" * 70)
print("Esto puede tardar 5-8 minutos...\n")

study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# ============================================================================
# RESULTADOS
# ============================================================================

print("\n" + "=" * 70)
print("RESULTADOS DE OPTIMIZACIÓN")
print("=" * 70)

best = study.best_params
print(f"\n📊 MEJORES PARÁMETROS ({modo_label}):")
if MODO_XGB:
    print(f"   Peso Local:        {best['peso_local']:.4f}")
    print(f"   Peso Empate:       {best['peso_empate']:.4f}")
    print(f"   Peso Visitante:    {best['peso_visitante']:.4f}")
    print(f"   n_estimators:      {best['n_estimators']}")
    print(f"   max_depth:         {best['max_depth']}")
    print(f"   learning_rate:     {best['learning_rate']:.4f}")
    print(f"   subsample:         {best['subsample']:.4f}")
    print(f"   colsample_bytree:  {best['colsample_bytree']:.4f}")
    print(f"   reg_alpha:         {best['reg_alpha']:.4f}")
    print(f"   reg_lambda:        {best['reg_lambda']:.4f}")
else:
    print(f"   Peso Local:        {best['peso_local']:.4f}")
    print(f"   Peso Empate:       {best['peso_empate']:.4f}")
    print(f"   Peso Visitante:    {best['peso_visitante']:.4f}")
    print(f"   n_estimators:      {best['n_estimators']}")
    print(f"   max_depth:         {best['max_depth']}")
    print(f"   min_samples_leaf:  {best['min_samples_leaf']}")
print(f"\n   Mejor F1 en CV:    {study.best_value:.4f}")

# ============================================================================
# EVALUAR EN TEST
# ============================================================================

print("\n" + "=" * 70)
print("EVALUACIÓN EN TEST SET")
print("=" * 70)

best_weights = {0: best['peso_local'], 1: best['peso_empate'], 2: best['peso_visitante']}

if MODO_XGB:
    sample_weights_test_fit = compute_sample_weight(class_weight=best_weights, y=y_train)
    modelo_opt = XGBClassifier(
        n_estimators=best['n_estimators'],
        max_depth=best['max_depth'],
        learning_rate=best['learning_rate'],
        subsample=best['subsample'],
        colsample_bytree=best['colsample_bytree'],
        reg_alpha=best['reg_alpha'],
        reg_lambda=best['reg_lambda'],
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
    )
    modelo_opt.fit(X_train, y_train, sample_weight=sample_weights_test_fit)
else:
    modelo_opt = RandomForestClassifier(
        n_estimators=best['n_estimators'],
        max_depth=best['max_depth'],
        min_samples_leaf=best['min_samples_leaf'],
        class_weight=best_weights,
        random_state=42,
        n_jobs=-1
    )
    modelo_opt.fit(X_train, y_train)

pred_opt = modelo_opt.predict(X_test)
acc = accuracy_score(y_test, pred_opt)
f1  = f1_score(y_test, pred_opt, average='weighted')

print(f"\n📊 MODELO OPTIMIZADO ({modo_label}):")
print(f"   Accuracy: {acc:.4f} ({acc:.2%})")
print(f"   F1-Score: {f1:.4f}")

cm = confusion_matrix(y_test, pred_opt)
print(f"\n   Matriz de Confusión:")
print(f"                    Predicción")
print(f"                 Local  Empate  Visit")
print(f"   Real Local    {cm[0,0]:>5}  {cm[0,1]:>6}  {cm[0,2]:>5}")
print(f"   Real Empate   {cm[1,0]:>5}  {cm[1,1]:>6}  {cm[1,2]:>5}")
print(f"   Real Visit    {cm[2,0]:>5}  {cm[2,1]:>6}  {cm[2,2]:>5}")

recall_l = cm[0,0] / cm[0].sum() if cm[0].sum() > 0 else 0
recall_e = cm[1,1] / cm[1].sum() if cm[1].sum() > 0 else 0
recall_v = cm[2,2] / cm[2].sum() if cm[2].sum() > 0 else 0

print(f"\n   Recall por clase:")
print(f"   Local:     {recall_l:.2%}")
print(f"   Empate:    {recall_e:.2%}")
print(f"   Visitante: {recall_v:.2%}")

# ============================================================================
# COMPARACIÓN CON BASELINES
# ============================================================================

print("\n" + "=" * 70)
print("COMPARACIÓN CON BASELINES")
print("=" * 70)

rf_base = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf_base.fit(X_train, y_train)
pred_base = rf_base.predict(X_test)

rf_bal = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=-1)
rf_bal.fit(X_train, y_train)
pred_bal = rf_bal.predict(X_test)

etiqueta_opt = "XGBoost Optuna ⚡" if MODO_XGB else "Optuna nuevos pesos ⭐"

print(f"\n{'Modelo':<40} {'Accuracy':<12} {'F1-Score':<12}")
print("-" * 65)
print(f"{'Básico (sin pesos)':<40} {accuracy_score(y_test, pred_base):>10.4f}  {f1_score(y_test, pred_base, average='weighted'):>10.4f}")
print(f"{'Balanceado (auto)':<40} {accuracy_score(y_test, pred_bal):>10.4f}  {f1_score(y_test, pred_bal, average='weighted'):>10.4f}")
print(f"{etiqueta_opt:<40} {acc:>10.4f}  {f1:>10.4f}")

# ============================================================================
# CÓDIGO LISTO PARA COPIAR A config.py
# ============================================================================

print("\n" + "=" * 70)
if MODO_XGB:
    print("COPIA ESTO EN config.py  →  PARAMS_XGB  y  PESOS_OPTIMOS")
elif MODO_SIN_CUOTAS:
    print("COPIA ESTO EN config.py  →  PARAMS_OPTIMOS_VB")
else:
    print("COPIA ESTO EN config.py  →  PARAMS_OPTIMOS y PESOS_OPTIMOS")
print("=" * 70)

if MODO_XGB:
    print(f"""
PESOS_OPTIMOS = {{
    0: {best['peso_local']:.4f},   # Local
    1: {best['peso_empate']:.4f},  # Empate
    2: {best['peso_visitante']:.4f}   # Visitante
}}

PARAMS_XGB = {{
    'n_estimators': {best['n_estimators']},
    'max_depth': {best['max_depth']},
    'learning_rate': {best['learning_rate']:.4f},
    'subsample': {best['subsample']:.4f},
    'colsample_bytree': {best['colsample_bytree']:.4f},
    'reg_alpha': {best['reg_alpha']:.4f},
    'reg_lambda': {best['reg_lambda']:.4f},
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'mlogloss',
}}
""")
else:
    nombre_params = "PARAMS_OPTIMOS_VB" if MODO_SIN_CUOTAS else "PARAMS_OPTIMOS"
    nombre_pesos  = "PESOS_OPTIMOS"

    print(f"""
{nombre_pesos} = {{
    0: {best['peso_local']:.4f},   # Local
    1: {best['peso_empate']:.4f},  # Empate
    2: {best['peso_visitante']:.4f}   # Visitante
}}

{nombre_params} = {{
    'n_estimators': {best['n_estimators']},
    'max_depth': {best['max_depth']},
    'min_samples_leaf': {best['min_samples_leaf']},
    'class_weight': {nombre_pesos},
    'random_state': 42,
    'n_jobs': -1
}}
""")
