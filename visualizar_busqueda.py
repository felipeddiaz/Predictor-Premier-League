# -*- coding: utf-8 -*-
"""
Optimización de pesos de clase con Optuna
VERSIÓN 2: Con features derivadas de cuotas
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

from config import ARCHIVO_FEATURES
from utils import agregar_xg_rolling, agregar_features_tabla, agregar_features_cuotas_derivadas

# ============================================================================
# CARGAR Y PREPARAR DATOS
# ============================================================================

print("="*70)
print("OPTIMIZACIÓN DE PESOS CON OPTUNA v2")
print("(Con features derivadas de cuotas)")
print("="*70)

df = pd.read_csv(ARCHIVO_FEATURES)
print(f"\n✅ Cargados: {len(df)} partidos")

# Aplicar todas las funciones de features
df = agregar_xg_rolling(df)
df = agregar_features_tabla(df)
df = agregar_features_cuotas_derivadas(df)

# Definir features
features_base = ['HT_AvgGoals', 'AT_AvgGoals', 'HT_AvgShotsTarget', 'AT_AvgShotsTarget',
                 'HT_Form_W', 'HT_Form_D', 'HT_Form_L', 'AT_Form_W', 'AT_Form_D', 'AT_Form_L']

features_cuotas = ['B365H', 'B365D', 'B365A', 'B365CH', 'B365CD', 'B365CA']

features_cuotas_derivadas = ['Prob_H', 'Prob_D', 'Prob_A', 'Prob_Move_H', 'Prob_Move_D', 'Prob_Move_A',
                              'Market_Move_Strength', 'Prob_Spread', 'Market_Confidence', 'Home_Advantage_Prob']

features_xg = ['HT_xG_Avg', 'AT_xG_Avg', 'HT_xGA_Avg', 'AT_xGA_Avg', 'xG_Diff', 'xG_Total']

features_h2h = ['H2H_Available', 'H2H_Matches', 'H2H_Home_Wins', 'H2H_Draws', 'H2H_Away_Wins',
                'H2H_Home_Goals_Avg', 'H2H_Away_Goals_Avg', 'H2H_Home_Win_Rate', 'H2H_BTTS_Rate']

features_h2h_derivadas = ['H2H_Goal_Diff', 'H2H_Win_Advantage', 'H2H_Total_Goals_Avg', 'H2H_Home_Consistent']

features_tabla = ['HT_Position', 'AT_Position', 'Position_Diff', 'Season_Progress', 'Position_Reliability']

all_features = (features_base + features_cuotas + features_cuotas_derivadas + 
                features_xg + features_h2h + features_h2h_derivadas + features_tabla)
features = [f for f in all_features if f in df.columns]

X = df[features].fillna(0)
y = df['FTR_numeric']

# Split temporal
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"✅ Features totales: {len(features)}")
print(f"   • Base: {len([f for f in features_base if f in features])}")
print(f"   • Cuotas: {len([f for f in features_cuotas if f in features])}")
print(f"   • Cuotas derivadas: {len([f for f in features_cuotas_derivadas if f in features])}")
print(f"   • xG: {len([f for f in features_xg if f in features])}")
print(f"   • H2H: {len([f for f in features_h2h if f in features])}")
print(f"   • Tabla: {len([f for f in features_tabla if f in features])}")
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
        random_state=42,
        n_jobs=-1
    )
    
    tscv = TimeSeriesSplit(n_splits=3)
    scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='f1_weighted', n_jobs=-1)
    
    return scores.mean()

# ============================================================================
# EJECUTAR OPTIMIZACIÓN
# ============================================================================

print("\n" + "="*70)
print("EJECUTANDO OPTIMIZACIÓN (150 trials)")
print("="*70)
print("Esto puede tardar 5-8 minutos...\n")

study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=150, show_progress_bar=True)

# ============================================================================
# RESULTADOS
# ============================================================================

print("\n" + "="*70)
print("RESULTADOS DE OPTIMIZACIÓN")
print("="*70)

best = study.best_params
print(f"\n📊 MEJORES PARÁMETROS:")
print(f"   Peso Local: {best['peso_local']:.4f}")
print(f"   Peso Empate: {best['peso_empate']:.4f}")
print(f"   Peso Visitante: {best['peso_visitante']:.4f}")
print(f"   n_estimators: {best['n_estimators']}")
print(f"   max_depth: {best['max_depth']}")
print(f"   min_samples_leaf: {best['min_samples_leaf']}")
print(f"\n   Mejor F1 en CV: {study.best_value:.4f}")

# ============================================================================
# EVALUAR EN TEST
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
    random_state=42,
    n_jobs=-1
)
rf_opt.fit(X_train, y_train)
pred_opt = rf_opt.predict(X_test)

acc = accuracy_score(y_test, pred_opt)
f1 = f1_score(y_test, pred_opt, average='weighted')

print(f"\n📊 MODELO OPTIMIZADO v2:")
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
rf_base = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf_base.fit(X_train, y_train)
pred_base = rf_base.predict(X_test)

# Balanceado
rf_bal = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=-1)
rf_bal.fit(X_train, y_train)
pred_bal = rf_bal.predict(X_test)

# Pesos anteriores (v1)
pesos_v1 = {0: 1.1957, 1: 2.8155, 2: 1.8358}
rf_v1 = RandomForestClassifier(n_estimators=219, max_depth=6, min_samples_leaf=8, class_weight=pesos_v1, random_state=42, n_jobs=-1)
rf_v1.fit(X_train, y_train)
pred_v1 = rf_v1.predict(X_test)

print(f"\n{'Modelo':<35} {'Accuracy':<12} {'F1-Score':<12}")
print("-" * 60)
print(f"{'Básico (sin pesos)':<35} {accuracy_score(y_test, pred_base):>10.4f}  {f1_score(y_test, pred_base, average='weighted'):>10.4f}")
print(f"{'Balanceado (auto)':<35} {accuracy_score(y_test, pred_bal):>10.4f}  {f1_score(y_test, pred_bal, average='weighted'):>10.4f}")
print(f"{'Optuna v1 (pesos anteriores)':<35} {accuracy_score(y_test, pred_v1):>10.4f}  {f1_score(y_test, pred_v1, average='weighted'):>10.4f}")
print(f"{'Optuna v2 (nuevos pesos) ⭐':<35} {acc:>10.4f}  {f1:>10.4f}")

mejora = (f1 - f1_score(y_test, pred_v1, average='weighted')) / f1_score(y_test, pred_v1, average='weighted') * 100
print(f"\n📈 Mejora vs Optuna v1: {mejora:+.2f}%")

# ============================================================================
# CÓDIGO PARA COPIAR
# ============================================================================

print("\n" + "="*70)
print("CÓDIGO PARA TU MODELO")
print("="*70)

print(f"""
# Pesos óptimos v2 (con features derivadas de cuotas)
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
    'random_state': 42,
    'n_jobs': -1
}}
""")