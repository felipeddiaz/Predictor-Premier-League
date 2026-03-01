# -*- coding: utf-8 -*-
"""
Optimización de pesos de clase e hiperparámetros con Optuna
Compatible con la estructura config.py + utils.py

USO:
    1. Modifica las listas de features abajo si quieres probar combinaciones
    2. Ejecuta: python optimizar_pesos_optuna.py
    3. Copia los pesos/params que imprime al final a tu config.py
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

# Importar desde tu proyecto
from config import (
    ARCHIVO_FEATURES,
    FEATURES_BASE,
    FEATURES_CUOTAS,
    FEATURES_CUOTAS_DERIVADAS,
    FEATURES_XG,
    FEATURES_H2H,
    FEATURES_H2H_DERIVADAS,
    FEATURES_TABLA,
    FEATURES_ASIAN_HANDICAP,
    RANDOM_SEED,
    TEST_SIZE,
    PESOS_OPTIMOS,  # Para comparar con baseline
)
from utils import agregar_xg_rolling, agregar_features_tabla, agregar_features_cuotas_derivadas, agregar_features_asian_handicap

# ============================================================================
# CONFIGURACIÓN DE FEATURES A USAR
# ============================================================================
# Modifica estas listas para probar diferentes combinaciones de features
# Comenta/descomenta grupos enteros o features individuales

USAR_FEATURES_BASE = True
USAR_FEATURES_CUOTAS = True
USAR_FEATURES_CUOTAS_DERIVADAS = True
USAR_FEATURES_XG = True
USAR_FEATURES_H2H = True
USAR_FEATURES_H2H_DERIVADAS = True
USAR_FEATURES_TABLA = True
USAR_FEATURES_ASIAN_HANDICAP = True

# ============================================================================
# CARGAR Y PREPARAR DATOS
# ============================================================================

print("="*70)
print("OPTIMIZACIÓN DE PESOS CON OPTUNA")
print("="*70)

df = pd.read_csv(ARCHIVO_FEATURES)
print(f"\n✅ Cargados: {len(df)} partidos")

# Aplicar funciones de feature engineering
df = agregar_xg_rolling(df)
df = agregar_features_tabla(df)
df = agregar_features_cuotas_derivadas(df)
df = agregar_features_asian_handicap(df)

# Construir lista de features según configuración
all_features = []

if USAR_FEATURES_BASE:
    all_features += FEATURES_BASE
if USAR_FEATURES_CUOTAS:
    all_features += FEATURES_CUOTAS
if USAR_FEATURES_CUOTAS_DERIVADAS:
    all_features += FEATURES_CUOTAS_DERIVADAS
if USAR_FEATURES_XG:
    all_features += FEATURES_XG
if USAR_FEATURES_H2H:
    all_features += FEATURES_H2H
if USAR_FEATURES_H2H_DERIVADAS:
    all_features += FEATURES_H2H_DERIVADAS
if USAR_FEATURES_TABLA:
    all_features += FEATURES_TABLA
if USAR_FEATURES_ASIAN_HANDICAP:
    all_features += FEATURES_ASIAN_HANDICAP

# Filtrar solo las que existen en el DataFrame
features = [f for f in all_features if f in df.columns]

X = df[features].fillna(0)
y = df['FTR_numeric']

# Split temporal
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, shuffle=False, random_state=RANDOM_SEED
)

print(f"\n📊 Features seleccionadas: {len(features)}")
print(f"   • Base: {len([f for f in FEATURES_BASE if f in features])}")
print(f"   • Cuotas: {len([f for f in FEATURES_CUOTAS if f in features])}")
print(f"   • Cuotas derivadas: {len([f for f in FEATURES_CUOTAS_DERIVADAS if f in features])}")
print(f"   • xG: {len([f for f in FEATURES_XG if f in features])}")
print(f"   • H2H: {len([f for f in FEATURES_H2H if f in features])}")
print(f"   • H2H derivadas: {len([f for f in FEATURES_H2H_DERIVADAS if f in features])}")
print(f"   • Tabla: {len([f for f in FEATURES_TABLA if f in features])}")
print(f"   • Asian Handicap: {len([f for f in FEATURES_ASIAN_HANDICAP if f in features])}")
print(f"\n✅ Train: {len(X_train)} | Test: {len(X_test)}")

# ============================================================================
# OBJETIVO DE OPTUNA
# ============================================================================

def objective(trial):
    # Pesos de clase
    w_local = trial.suggest_float('peso_local', 0.5, 2.5)
    w_empate = trial.suggest_float('peso_empate', 1.0, 5.0)
    w_visitante = trial.suggest_float('peso_visitante', 0.5, 2.5)
    
    # Hiperparámetros RF
    n_estimators = trial.suggest_int('n_estimators', 100, 400)
    max_depth = trial.suggest_int('max_depth', 4, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 3, 15)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight={0: w_local, 1: w_empate, 2: w_visitante},
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    tscv = TimeSeriesSplit(n_splits=3)
    scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='f1_weighted', n_jobs=-1)
    
    return scores.mean()

# ============================================================================
# EJECUTAR OPTIMIZACIÓN
# ============================================================================

N_TRIALS = 150  # Aumentar para mejor búsqueda (más lento)

print("\n" + "="*70)
print(f"EJECUTANDO OPTIMIZACIÓN ({N_TRIALS} trials)")
print("="*70)
print("Esto puede tardar 3-8 minutos...\n")

study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# ============================================================================
# RESULTADOS
# ============================================================================

print("\n" + "="*70)
print("RESULTADOS DE OPTIMIZACIÓN")
print("="*70)

best = study.best_params
print(f"\n📊 MEJORES PARÁMETROS ENCONTRADOS:")
print(f"   Peso Local: {best['peso_local']:.4f}")
print(f"   Peso Empate: {best['peso_empate']:.4f}")
print(f"   Peso Visitante: {best['peso_visitante']:.4f}")
print(f"   n_estimators: {best['n_estimators']}")
print(f"   max_depth: {best['max_depth']}")
print(f"   min_samples_leaf: {best['min_samples_leaf']}")
print(f"\n   Mejor F1 en CV: {study.best_value:.4f}")

# ============================================================================
# EVALUAR EN TEST SET
# ============================================================================

print("\n" + "="*70)
print("EVALUACIÓN EN TEST SET")
print("="*70)

best_weights = {0: best['peso_local'], 1: best['peso_empate'], 2: best['peso_visitante']}

rf_opt = RandomForestClassifier(
    n_estimators=best['n_estimators'],
    max_depth=best['max_depth'],
    min_samples_leaf=best['min_samples_leaf'],
    class_weight=best_weights,
    random_state=RANDOM_SEED,
    n_jobs=-1
)
rf_opt.fit(X_train, y_train)
pred_opt = rf_opt.predict(X_test)

acc = accuracy_score(y_test, pred_opt)
f1 = f1_score(y_test, pred_opt, average='weighted')

print(f"\n📊 MODELO OPTIMIZADO:")
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
print(f"   Local: {recall_l:.2%}")
print(f"   Empate: {recall_e:.2%}")
print(f"   Visitante: {recall_v:.2%}")

# ============================================================================
# COMPARACIÓN CON BASELINES
# ============================================================================

print("\n" + "="*70)
print("COMPARACIÓN CON BASELINES")
print("="*70)

# Básico
rf_base = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=RANDOM_SEED, n_jobs=-1)
rf_base.fit(X_train, y_train)
pred_base = rf_base.predict(X_test)

# Balanceado
rf_bal = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)
rf_bal.fit(X_train, y_train)
pred_bal = rf_bal.predict(X_test)

# Pesos anteriores (de config.py)
rf_prev = RandomForestClassifier(
    n_estimators=229, max_depth=8, min_samples_leaf=3,
    class_weight=PESOS_OPTIMOS, random_state=RANDOM_SEED, n_jobs=-1
)
rf_prev.fit(X_train, y_train)
pred_prev = rf_prev.predict(X_test)

f1_base = f1_score(y_test, pred_base, average='weighted')
f1_bal = f1_score(y_test, pred_bal, average='weighted')
f1_prev = f1_score(y_test, pred_prev, average='weighted')

print(f"\n{'Modelo':<35} {'Accuracy':<12} {'F1-Score':<12}")
print("-" * 60)
print(f"{'Básico (sin pesos)':<35} {accuracy_score(y_test, pred_base):>10.4f}  {f1_base:>10.4f}")
print(f"{'Balanceado (auto)':<35} {accuracy_score(y_test, pred_bal):>10.4f}  {f1_bal:>10.4f}")
print(f"{'Config.py actual':<35} {accuracy_score(y_test, pred_prev):>10.4f}  {f1_prev:>10.4f}")
print(f"{'Optuna nuevo ⭐':<35} {acc:>10.4f}  {f1:>10.4f}")

if f1 > f1_prev:
    mejora = (f1 - f1_prev) / f1_prev * 100
    print(f"\n📈 Mejora vs config.py actual: +{mejora:.2f}%")
else:
    diferencia = (f1_prev - f1) / f1_prev * 100
    print(f"\n📉 Config.py actual es mejor por: {diferencia:.2f}%")
    print("   (Los pesos actuales ya son óptimos para estas features)")

# ============================================================================
# CÓDIGO PARA COPIAR A CONFIG.PY
# ============================================================================

print("\n" + "="*70)
print("CÓDIGO PARA COPIAR A CONFIG.PY")
print("="*70)

print(f"""
# Pesos óptimos (encontrados con {N_TRIALS} trials de Optuna)
# Features usadas: {len(features)}
PESOS_OPTIMOS = {{
    0: {best['peso_local']:.4f},  # Local
    1: {best['peso_empate']:.4f},  # Empate
    2: {best['peso_visitante']:.4f}   # Visitante
}}

PARAMS_OPTIMOS = {{
    'n_estimators': {best['n_estimators']},
    'max_depth': {best['max_depth']},
    'min_samples_leaf': {best['min_samples_leaf']},
    'class_weight': PESOS_OPTIMOS,
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}}
""")

print("="*70)
print("✅ OPTIMIZACIÓN COMPLETADA")
print("="*70)