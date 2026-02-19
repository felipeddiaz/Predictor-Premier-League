# -*- coding: utf-8 -*-
"""
Optimización de pesos e hiperparámetros con Optuna.

Modos de ejecución:
  MODO_SIN_CUOTAS = False  →  busca PARAMS_OPTIMOS      (modelo 02, con cuotas)
  MODO_SIN_CUOTAS = True   →  busca PARAMS_OPTIMOS_VB   (modelo 03, sin cuotas)

Al terminar imprime el bloque listo para copiar a config.py.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
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
# FLAG PRINCIPAL — cambia aquí antes de ejecutar
# ============================================================================

MODO_SIN_CUOTAS = False   # False = optimiza PARAMS_OPTIMOS (02, con cuotas)
                           # True  = optimiza PARAMS_OPTIMOS_VB (03, sin cuotas)

N_TRIALS = 150             # Número de trials Optuna (5-8 min con 150)

# ============================================================================
# CARGAR Y PREPARAR DATOS
# ============================================================================

modo_label = "SIN CUOTAS (modelo 03)" if MODO_SIN_CUOTAS else "CON CUOTAS (modelo 02)"

print("=" * 70)
print(f"OPTIMIZACIÓN DE PESOS CON OPTUNA")
print(f"Modo: {modo_label}")
print("=" * 70)

df = pd.read_csv(ARCHIVO_FEATURES)
print(f"\n✅ Cargados: {len(df)} partidos")

# Features calculadas en memoria
df = agregar_xg_rolling(df)
df = agregar_features_tabla(df)

if MODO_SIN_CUOTAS:
    # Modo sin cuotas: filtrar solo partidos con H2H disponible (igual que el 03)
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
    # Modo con cuotas: incluir cuotas derivadas (igual que el 02)
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
print(f"\n📊 MEJORES PARÁMETROS:")
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

rf_opt = RandomForestClassifier(
    n_estimators=best['n_estimators'],
    max_depth=best['max_depth'],
    min_samples_leaf=best['min_samples_leaf'],
    class_weight=best_weights,
    random_state=42,
    n_jobs=-1
)
rf_opt.fit(X_train, y_train)
pred_opt = rf_opt.predict(X_test)

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

print(f"\n{'Modelo':<40} {'Accuracy':<12} {'F1-Score':<12}")
print("-" * 65)
print(f"{'Básico (sin pesos)':<40} {accuracy_score(y_test, pred_base):>10.4f}  {f1_score(y_test, pred_base, average='weighted'):>10.4f}")
print(f"{'Balanceado (auto)':<40} {accuracy_score(y_test, pred_bal):>10.4f}  {f1_score(y_test, pred_bal, average='weighted'):>10.4f}")
print(f"{'Optuna nuevos pesos ⭐':<40} {acc:>10.4f}  {f1:>10.4f}")

# ============================================================================
# CÓDIGO LISTO PARA COPIAR A config.py
# ============================================================================

print("\n" + "=" * 70)
if MODO_SIN_CUOTAS:
    print("COPIA ESTO EN config.py  →  PARAMS_OPTIMOS_VB")
else:
    print("COPIA ESTO EN config.py  →  PARAMS_OPTIMOS y PESOS_OPTIMOS")
print("=" * 70)

nombre_params = "PARAMS_OPTIMOS_VB" if MODO_SIN_CUOTAS else "PARAMS_OPTIMOS"
nombre_pesos  = "PESOS_OPTIMOS"     # mismo nombre en ambos modos

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
