# -*- coding: utf-8 -*-
"""
Premier League Match Predictor
02 - ENTRENAMIENTO DE MODELOS (VERSION OPTIMIZADA CON PESOS OPTUNA + XGBOOST)

Entrena cuatro variantes (RF basico, RF balanceado, RF Optuna, XGBoost),
selecciona la mejor por F1-Score ponderado y aplica calibracion Platt
Scaling para obtener probabilidades realistas.

Pipeline:
    datos/procesados/premier_league_con_features.csv
    → features en memoria (xG rolling, tabla, cuotas derivadas)
    → 80/20 split temporal
    → RF basico + RF balanced + RF Optuna + XGBoost (PARAMS_XGB)
    → CalibratedClassifierCV (sigmoid)
    → modelos/modelo_final_optimizado.pkl
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

from config import (
    ARCHIVO_FEATURES,
    RUTA_MODELOS,
    ARCHIVO_MODELO,
    ARCHIVO_FEATURES_PKL,
    ARCHIVO_METADATA,
    PESOS_OPTIMOS,
    PESOS_XGB,
    PARAMS_OPTIMOS,
    PARAMS_XGB,
    ALL_FEATURES,
    FEATURES_BASE,
    FEATURES_CUOTAS,
    FEATURES_CUOTAS_DERIVADAS,
    FEATURES_XG,
    FEATURES_H2H,
    FEATURES_H2H_DERIVADAS,
    FEATURES_TABLA,
    FEATURES_ASIAN_HANDICAP,
    FEATURES_ROLLING_EXTRA,
    TEST_SIZE,
    RANDOM_SEED,
)
from utils import (
    agregar_xg_rolling,
    agregar_features_tabla,
    agregar_features_cuotas_derivadas,
    agregar_features_asian_handicap,
    agregar_features_rolling_extra,
)

warnings.filterwarnings('ignore')

os.makedirs(RUTA_MODELOS, exist_ok=True)

RUTA_DATOS = ARCHIVO_FEATURES

# ============================================================================
# CARGA DE DATOS
# ============================================================================

def cargar_datos():
    """Carga datos procesados y prepara features."""
    print("="*70)
    print("FASE 1: CARGANDO DATOS")
    print("="*70)
    
    if not os.path.exists(RUTA_DATOS):
        print(f"❌ ERROR: No se encontró '{RUTA_DATOS}'")
        print("   Ejecuta primero: python 01_preparar_datos.py")
        return None, None, None
    
    df = pd.read_csv(RUTA_DATOS)
    print(f"✅ Cargados: {len(df)} partidos")
    
    # Agregar features calculadas en memoria (funciones canonicas de utils.py)
    df = agregar_xg_rolling(df)
    df = agregar_features_tabla(df)
    df = agregar_features_cuotas_derivadas(df)
    df = agregar_features_asian_handicap(df)
    df = agregar_features_rolling_extra(df)

    # Filtrar solo las que existen en el DataFrame
    # Usa ALL_FEATURES de config.py como lista canonica unica
    features = [f for f in ALL_FEATURES if f in df.columns]
    
    print(f"\n📊 Features totales: {len(features)}")
    print(f"   • Base: {len([f for f in FEATURES_BASE if f in features])}")
    print(f"   • Cuotas: {len([f for f in FEATURES_CUOTAS if f in features])}")
    print(f"   • xG rolling: {len([f for f in FEATURES_XG if f in features])}")
    print(f"   • H2H: {len([f for f in FEATURES_H2H if f in features])}")
    print(f"   • H2H derivadas: {len([f for f in FEATURES_H2H_DERIVADAS if f in features])}")
    print(f"   • Tabla: {len([f for f in FEATURES_TABLA if f in features])}")
    print(f"   • Cuotas derivadas: {len([f for f in FEATURES_CUOTAS_DERIVADAS if f in features])}")
    print(f"   • Asian Handicap: {len([f for f in FEATURES_ASIAN_HANDICAP if f in features])}")
    print(f"   • Rolling extra: {len([f for f in FEATURES_ROLLING_EXTRA if f in features])}")
    
    # Info de H2H
    if 'H2H_Available' in features:
        con_h2h = df['H2H_Available'].sum()
        sin_h2h = len(df) - con_h2h
        print(f"\n✅ H2H disponible:")
        print(f"   Partidos CON H2H: {con_h2h} ({con_h2h/len(df)*100:.1f}%)")
        print(f"   Partidos SIN H2H: {sin_h2h} ({sin_h2h/len(df)*100:.1f}%)")
    
    X = df[features].fillna(0)
    y = df['FTR_numeric']
    
    # Distribución
    print(f"\n📊 Distribución de resultados:")
    for clase in [0, 1, 2]:
        nombre = ['Local', 'Empate', 'Visitante'][clase]
        count = (y == clase).sum()
        pct = count / len(y) * 100
        print(f"   {nombre}: {count} ({pct:.1f}%)")
    
    return X, y, features

# ============================================================================
# ENTRENAMIENTO DE MODELOS
# ============================================================================

def entrenar_modelos(X_train, y_train, X_test, y_test):
    """Entrena los modelos RF y devuelve resultados."""
    
    resultados = {}
    
    # -------------------------------------------------------------------------
    # MODELO 1: Random Forest Básico (baseline)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("MODELO 1: RANDOM FOREST BÁSICO (Baseline)")
    print("="*70)
    
    rf_basico = RandomForestClassifier(
        n_estimators=100, 
        min_samples_leaf=5, 
        random_state=42, 
        n_jobs=-1
    )
    rf_basico.fit(X_train, y_train)
    pred_basico = rf_basico.predict(X_test)
    
    acc = accuracy_score(y_test, pred_basico)
    f1 = f1_score(y_test, pred_basico, average='weighted')
    
    resultados['RF_Basico'] = {
        'modelo': rf_basico,
        'predicciones': pred_basico,
        'accuracy': acc,
        'f1_score': f1,
        'nombre': 'Random Forest Básico'
    }
    
    print(f"✅ Accuracy: {acc:.2%} | F1-Score: {f1:.4f}")
    
    # -------------------------------------------------------------------------
    # MODELO 2: Random Forest Balanceado (auto)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("MODELO 2: RANDOM FOREST BALANCEADO (Auto)")
    print("="*70)
    
    rf_balanceado = RandomForestClassifier(
        n_estimators=100, 
        min_samples_leaf=5, 
        class_weight='balanced',
        random_state=42, 
        n_jobs=-1
    )
    rf_balanceado.fit(X_train, y_train)
    pred_balanceado = rf_balanceado.predict(X_test)
    
    acc = accuracy_score(y_test, pred_balanceado)
    f1 = f1_score(y_test, pred_balanceado, average='weighted')
    
    resultados['RF_Balanceado'] = {
        'modelo': rf_balanceado,
        'predicciones': pred_balanceado,
        'accuracy': acc,
        'f1_score': f1,
        'nombre': 'Random Forest Balanceado'
    }
    
    print(f"✅ Accuracy: {acc:.2%} | F1-Score: {f1:.4f}")
    
    # -------------------------------------------------------------------------
    # MODELO 3: Random Forest con PESOS OPTUNA (el mejor)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("MODELO 3: RANDOM FOREST CON PESOS OPTUNA ⭐")
    print("="*70)
    print(f"   Pesos optimizados:")
    print(f"   • Local (0): {PESOS_OPTIMOS[0]:.4f}")
    print(f"   • Empate (1): {PESOS_OPTIMOS[1]:.4f}")
    print(f"   • Visitante (2): {PESOS_OPTIMOS[2]:.4f}")
    print(f"   Hiperparámetros:")
    print(f"   • n_estimators: {PARAMS_OPTIMOS['n_estimators']}")
    print(f"   • max_depth: {PARAMS_OPTIMOS['max_depth']}")
    print(f"   • min_samples_leaf: {PARAMS_OPTIMOS['min_samples_leaf']}")
    
    rf_optuna = RandomForestClassifier(**PARAMS_OPTIMOS)
    rf_optuna.fit(X_train, y_train)
    pred_optuna = rf_optuna.predict(X_test)
    
    acc = accuracy_score(y_test, pred_optuna)
    f1 = f1_score(y_test, pred_optuna, average='weighted')
    
    resultados['RF_Optuna'] = {
        'modelo': rf_optuna,
        'predicciones': pred_optuna,
        'accuracy': acc,
        'f1_score': f1,
        'nombre': 'Random Forest Optuna'
    }
    
    print(f"\n✅ Accuracy: {acc:.2%} | F1-Score: {f1:.4f}")
    
    # Mostrar recall por clase
    cm = confusion_matrix(y_test, pred_optuna)
    recall_local = cm[0,0] / cm[0].sum() if cm[0].sum() > 0 else 0
    recall_empate = cm[1,1] / cm[1].sum() if cm[1].sum() > 0 else 0
    recall_visitante = cm[2,2] / cm[2].sum() if cm[2].sum() > 0 else 0
    
    print(f"\n📊 Recall por clase:")
    print(f"   Local: {recall_local:.2%}")
    print(f"   Empate: {recall_empate:.2%}")
    print(f"   Visitante: {recall_visitante:.2%}")

    # -------------------------------------------------------------------------
    # MODELO 4: XGBoost con pesos de clase (PARAMS_XGB de config.py)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("MODELO 4: XGBOOST ⚡")
    print("="*70)
    print(f"   Pesos XGB optimizados:")
    print(f"   • Local (0): {PESOS_XGB[0]:.4f}")
    print(f"   • Empate (1): {PESOS_XGB[1]:.4f}")
    print(f"   • Visitante (2): {PESOS_XGB[2]:.4f}")
    print(f"   Hiperparámetros:")
    print(f"   • n_estimators: {PARAMS_XGB['n_estimators']}")
    print(f"   • max_depth: {PARAMS_XGB['max_depth']}")
    print(f"   • learning_rate: {PARAMS_XGB['learning_rate']}")
    print(f"   • subsample: {PARAMS_XGB['subsample']}")
    print(f"   • colsample_bytree: {PARAMS_XGB['colsample_bytree']}")
    print(f"   • colsample_bylevel: {PARAMS_XGB.get('colsample_bylevel', 'N/A')}")

    # XGBoost multiclase no acepta class_weight dict — se usan sample_weight
    # basados en PESOS_XGB (optimizados especificamente para XGB)
    sample_weights_train = compute_sample_weight(
        class_weight=PESOS_XGB, y=y_train
    )

    xgb_model = XGBClassifier(**PARAMS_XGB)
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights_train)
    pred_xgb = xgb_model.predict(X_test)

    acc = accuracy_score(y_test, pred_xgb)
    f1 = f1_score(y_test, pred_xgb, average='weighted')

    resultados['XGBoost'] = {
        'modelo': xgb_model,
        'predicciones': pred_xgb,
        'accuracy': acc,
        'f1_score': f1,
        'nombre': 'XGBoost'
    }

    print(f"\n✅ Accuracy: {acc:.2%} | F1-Score: {f1:.4f}")

    # Recall por clase
    cm_xgb = confusion_matrix(y_test, pred_xgb)
    recall_local = cm_xgb[0,0] / cm_xgb[0].sum() if cm_xgb[0].sum() > 0 else 0
    recall_empate = cm_xgb[1,1] / cm_xgb[1].sum() if cm_xgb[1].sum() > 0 else 0
    recall_visitante = cm_xgb[2,2] / cm_xgb[2].sum() if cm_xgb[2].sum() > 0 else 0

    print(f"\n📊 Recall por clase:")
    print(f"   Local: {recall_local:.2%}")
    print(f"   Empate: {recall_empate:.2%}")
    print(f"   Visitante: {recall_visitante:.2%}")

    return resultados

# ============================================================================
# CALIBRACIÓN DE PROBABILIDADES
# ============================================================================

def calibrar_modelo(modelo, X_train, y_train, X_test, y_test):
    """
    Calibra el modelo para obtener probabilidades más realistas.

    Usa TimeSeriesSplit (no KFold estándar) para respetar la causalidad temporal.
    Evalúa calibración con Brier Score y Log Loss — NO con F1.
    F1 mide clasificación; Brier/LogLoss miden calidad probabilística,
    que es lo que importa para value betting y Kelly Criterion.

    Retorna (modelo_a_guardar, probs, es_calibrado):
      - Si la calibración mejora Brier Score → retorna modelo calibrado
      - Si no mejora → retorna modelo original (probabilidades ya eran buenas)
    """
    print("\n" + "="*70)
    print("CALIBRACIÓN DE PROBABILIDADES")
    print("="*70)

    # --- Métricas del modelo SIN calibrar ---
    probs_original = modelo.predict_proba(X_test)
    logloss_original = log_loss(y_test, probs_original)

    # Brier Score multiclase: promedio del Brier por clase (one-vs-rest)
    brier_original = 0.0
    for clase in range(3):
        y_bin = (y_test == clase).astype(int)
        brier_original += brier_score_loss(y_bin, probs_original[:, clase])
    brier_original /= 3

    print(f"\n📊 Modelo ORIGINAL (sin calibrar):")
    print(f"   Log Loss:    {logloss_original:.4f}")
    print(f"   Brier Score: {brier_original:.4f}")

    # --- Calibrar con TimeSeriesSplit (respeta orden temporal) ---
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)

    modelo_calibrado = CalibratedClassifierCV(
        estimator=modelo,
        method='sigmoid',   # Platt Scaling
        cv=tscv             # Temporal, no shuffle
    )
    modelo_calibrado.fit(X_train, y_train)

    # --- Métricas del modelo CALIBRADO ---
    probs_calibradas = modelo_calibrado.predict_proba(X_test)
    logloss_calibrado = log_loss(y_test, probs_calibradas)

    brier_calibrado = 0.0
    for clase in range(3):
        y_bin = (y_test == clase).astype(int)
        brier_calibrado += brier_score_loss(y_bin, probs_calibradas[:, clase])
    brier_calibrado /= 3

    print(f"\n📊 Modelo CALIBRADO (Platt Scaling + TimeSeriesSplit):")
    print(f"   Log Loss:    {logloss_calibrado:.4f}")
    print(f"   Brier Score: {brier_calibrado:.4f}")

    # --- Comparar: ¿la calibración mejoró las probabilidades? ---
    mejora_logloss = logloss_original - logloss_calibrado
    mejora_brier = brier_original - brier_calibrado

    print(f"\n📊 COMPARACIÓN:")
    print(f"   {'Métrica':<15} {'Original':<12} {'Calibrado':<12} {'Cambio':<12}")
    print(f"   {'-'*50}")
    print(f"   {'Log Loss':<15} {logloss_original:<12.4f} {logloss_calibrado:<12.4f} {mejora_logloss:>+10.4f}")
    print(f"   {'Brier Score':<15} {brier_original:<12.4f} {brier_calibrado:<12.4f} {mejora_brier:>+10.4f}")

    # F1 informativo (pero NO es el criterio de decisión)
    pred_cal = modelo_calibrado.predict(X_test)
    f1_cal = f1_score(y_test, pred_cal, average='weighted')
    pred_orig = modelo.predict(X_test)
    f1_orig = f1_score(y_test, pred_orig, average='weighted')
    print(f"   {'F1 (info)':<15} {f1_orig:<12.4f} {f1_cal:<12.4f} {f1_cal - f1_orig:>+10.4f}")

    # Decisión basada en Brier Score (más importante para betting)
    if brier_calibrado < brier_original:
        print(f"\n✅ Calibración MEJORA probabilidades (Brier -{mejora_brier:.4f})")
        print(f"   → Guardando modelo CALIBRADO (mejor para value betting)")
        return modelo_calibrado, probs_calibradas, True
    else:
        print(f"\n⚠️  Calibración NO mejora probabilidades (Brier +{abs(mejora_brier):.4f})")
        print(f"   → Guardando modelo ORIGINAL (ya produce buenas probabilidades)")
        return modelo, probs_original, False

# ============================================================================
# COMPARACIÓN Y SELECCIÓN DEL MEJOR
# ============================================================================

def seleccionar_mejor_modelo(resultados, y_test):
    """Compara modelos y selecciona el mejor según F1-Score."""
    
    print("\n" + "="*70)
    print("FASE 2: COMPARACIÓN DE MODELOS")
    print("="*70)
    
    print(f"\n{'Modelo':<40} {'Accuracy':<12} {'F1-Score':<12}")
    print("-" * 65)
    
    for key, datos in resultados.items():
        print(f"{datos['nombre']:<40} {datos['accuracy']:>10.2%}  {datos['f1_score']:>10.4f}")
    
    # Elegir mejor por F1-Score
    mejor_key = max(resultados.items(), key=lambda x: x[1]['f1_score'])[0]
    mejor = resultados[mejor_key]
    
    print("\n" + "🏆" * 30)
    print(f"GANADOR: {mejor['nombre']}")
    print(f"   Accuracy: {mejor['accuracy']:.2%}")
    print(f"   F1-Score: {mejor['f1_score']:.4f}")
    print("🏆" * 30)
    
    # Reporte detallado
    print(f"\n📊 REPORTE DETALLADO:")
    print("-" * 65)
    target_names = ['Local', 'Empate', 'Visitante']
    print(classification_report(y_test, mejor['predicciones'], target_names=target_names))
    
    return mejor_key, mejor


# ============================================================================
# OPTIMIZACIÓN ADICIONAL (opcional)
# ============================================================================

def optimizar_modelo_adicional(mejor_key, mejor_modelo, X_train, y_train, X_test, y_test):
    """
    Optimización adicional con RandomizedSearchCV.
    Solo se ejecuta si el modelo ganador NO es el de Optuna (ya está optimizado).
    """
    
    # Si el modelo Optuna o XGBoost ganó, no necesita más optimización
    if mejor_key in ('RF_Optuna', 'XGBoost'):
        print("\n" + "="*70)
        print("FASE 3: OPTIMIZACIÓN ADICIONAL")
        print("="*70)
        print(f"\n✅ {mejor_modelo['nombre']} ya está optimizado. Saltando esta fase.")
        return mejor_modelo['modelo'], mejor_modelo['predicciones'], False
    
    print("\n" + "="*70)
    print("FASE 3: OPTIMIZACIÓN DE HIPERPARÁMETROS")
    print("="*70)
    
    print(f"🔧 Optimizando: {mejor_modelo['nombre']}")
    print("   Usando TimeSeriesSplit para validación temporal")
    print("   (Esto puede tardar 2-5 minutos)\n")
    
    # Configurar class_weight según el modelo ganador
    if mejor_key == 'RF_Balanceado':
        class_weight = 'balanced'
    else:
        class_weight = None
    
    # Grid de búsqueda
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [6, 10, 15, 20, None],
        'min_samples_leaf': [4, 5, 8, 10],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None]
    }
    
    base_model = RandomForestClassifier(
        class_weight=class_weight,
        random_state=42, 
        n_jobs=-1
    )
    
    # TimeSeriesSplit para datos temporales
    tscv = TimeSeriesSplit(n_splits=3)
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=30,
        cv=tscv,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        scoring='f1_weighted'
    )
    
    random_search.fit(X_train, y_train)
    
    # Evaluar
    modelo_optimizado = random_search.best_estimator_
    pred_optimizado = modelo_optimizado.predict(X_test)
    
    acc_opt = accuracy_score(y_test, pred_optimizado)
    f1_opt = f1_score(y_test, pred_optimizado, average='weighted')
    
    print("\n✅ OPTIMIZACIÓN COMPLETADA")
    print("="*70)
    print("\n📋 MEJORES PARÁMETROS:")
    for param, value in random_search.best_params_.items():
        print(f"   {param}: {value}")
    
    print("\n📊 COMPARACIÓN ANTES/DESPUÉS:")
    print(f"{'Métrica':<15} {'Antes':<15} {'Después':<15} {'Cambio':<10}")
    print("-" * 55)
    
    cambio_acc = acc_opt - mejor_modelo['accuracy']
    cambio_f1 = f1_opt - mejor_modelo['f1_score']
    
    print(f"{'Accuracy':<15} {mejor_modelo['accuracy']:>13.2%} {acc_opt:>13.2%} {cambio_acc:>+9.2%}")
    print(f"{'F1-Score':<15} {mejor_modelo['f1_score']:>13.4f} {f1_opt:>13.4f} {cambio_f1:>+9.4f}")
    
    if f1_opt > mejor_modelo['f1_score']:
        print("\n🎉 ¡El modelo mejoró con la optimización!")
        return modelo_optimizado, pred_optimizado, True
    else:
        print("\n⚠️  Sin mejora significativa. Usando el original.")
        return mejor_modelo['modelo'], mejor_modelo['predicciones'], False


# ============================================================================
# VISUALIZACIONES
# ============================================================================

def visualizar_resultados(y_test, predictions, nombre_modelo, features, modelo):
    """Genera visualizaciones del modelo final."""
    
    print("\n" + "="*70)
    print("FASE 4: GENERANDO VISUALIZACIONES")
    print("="*70)
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, predictions)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Local', 'Empate', 'Visitante'],
                yticklabels=['Local', 'Empate', 'Visitante'])
    plt.title(f'Matriz de Confusión - {nombre_modelo}', fontsize=16, fontweight='bold')
    plt.ylabel('Resultado Real', fontsize=12)
    plt.xlabel('Predicción del Modelo', fontsize=12)
    plt.tight_layout()
    
    archivo_cm = os.path.join(RUTA_MODELOS, 'confusion_matrix_final.png')
    plt.savefig(archivo_cm, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {archivo_cm}")
    plt.close()
    
    # Importancia de características (RF y XGBoost ambos tienen feature_importances_)
    modelo_base = modelo.estimator if hasattr(modelo, 'estimator') else modelo
    if hasattr(modelo_base, 'feature_importances_'):
        importances = modelo_base.feature_importances_
        
        df_imp = pd.DataFrame({
            'Feature': features,
            'Importancia': importances
        }).sort_values('Importancia', ascending=False)
        
        plt.figure(figsize=(12, 8))
        top_n = min(15, len(df_imp))
        sns.barplot(x='Importancia', y='Feature', data=df_imp.head(top_n), palette='viridis')
        plt.title(f'Top {top_n} Features - {nombre_modelo}', fontsize=16, fontweight='bold')
        plt.xlabel('Importancia', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        archivo_imp = os.path.join(RUTA_MODELOS, 'feature_importance_final.png')
        plt.savefig(archivo_imp, dpi=150, bbox_inches='tight')
        print(f"✅ Guardado: {archivo_imp}")
        plt.close()
        
        print("\n📋 TOP 10 FEATURES:")
        print("-" * 40)
        for i, (_, row) in enumerate(df_imp.head(10).iterrows(), 1):
            print(f"   {i:2}. {row['Feature']:<25} {row['Importancia']:.4f}")


# ============================================================================
# GUARDAR MODELO FINAL
# ============================================================================

def guardar_modelo_final(modelo, features, nombre_modelo):
    """Guarda el modelo final."""
    
    print("\n" + "="*70)
    print("FASE 5: GUARDANDO MODELO")
    print("="*70)
    
    # Usar rutas de config.py (unica fuente de verdad)
    joblib.dump(modelo, ARCHIVO_MODELO)
    print(f"✅ Modelo: {ARCHIVO_MODELO}")

    joblib.dump(features, ARCHIVO_FEATURES_PKL)
    print(f"✅ Features: {ARCHIVO_FEATURES_PKL}")

    metadata = {
        'nombre_modelo': nombre_modelo,
        'n_features': len(features),
        'features': features,
        'pesos_optuna': PESOS_OPTIMOS,
        'params_optuna': PARAMS_OPTIMOS,
        'params_xgb': PARAMS_XGB,
    }
    joblib.dump(metadata, ARCHIVO_METADATA)
    print(f"✅ Metadata: {ARCHIVO_METADATA}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Pipeline completo."""
    
    print("\n" + "⚽" * 35)
    print("   PREMIER LEAGUE - ENTRENAMIENTO DE MODELOS")
    print("   (Versión con Pesos Optimizados por Optuna)")
    print("⚽" * 35 + "\n")
    
    # Cargar datos
    resultado = cargar_datos()
    if resultado[0] is None:
        return None, None
    
    X, y, features = resultado
    
    # Split temporal (shuffle=False mantiene orden)
    pct_label = f"{int(TEST_SIZE*100)}"
    print(f"\n🔪 División de datos ({100-int(TEST_SIZE*100)}/{pct_label} temporal)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False
    )
    print(f"   Entrenamiento: {len(X_train)} partidos")
    print(f"   Prueba: {len(X_test)} partidos")
    
    # Entrenar modelos (incluyendo el de Optuna)
    resultados = entrenar_modelos(X_train, y_train, X_test, y_test)
    
    # Seleccionar mejor
    mejor_key, mejor_modelo = seleccionar_mejor_modelo(resultados, y_test)
    
    # Optimización adicional (solo si no ganó Optuna)
    modelo_final, pred_final, mejorado = optimizar_modelo_adicional(
        mejor_key, mejor_modelo, X_train, y_train, X_test, y_test
    )
    
    # Calibrar probabilidades — decide automáticamente si calibrar o no
    modelo_a_guardar, probs_finales, fue_calibrado = calibrar_modelo(
        modelo_final, X_train, y_train, X_test, y_test
    )

    tag_cal = "(Calibrado)" if fue_calibrado else "(Sin Calibrar)"
    nombre_final = f"{mejor_modelo['nombre']} {tag_cal}"

    # Visualizaciones (usa modelo base para feature_importances_)
    visualizar_resultados(y_test, pred_final, nombre_final, features, modelo_final)

    # Guardar el modelo que ganó la comparación de calibración
    guardar_modelo_final(modelo_a_guardar, features, nombre_final)

    # Métricas reales del modelo guardado
    pred_guardado = modelo_a_guardar.predict(X_test)
    acc_final = accuracy_score(y_test, pred_guardado)
    f1_final = f1_score(y_test, pred_guardado, average='weighted')

    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"\n🏆 Modelo guardado: {nombre_final}")
    print(f"📊 Accuracy: {acc_final:.2%}")
    print(f"📊 F1-Score: {f1_final:.4f}")

    if mejor_key == 'RF_Optuna':
        print(f"\n⭐ Pesos Optuna utilizados:")
        print(f"   Local: {PESOS_OPTIMOS[0]:.4f}")
        print(f"   Empate: {PESOS_OPTIMOS[1]:.4f}")
        print(f"   Visitante: {PESOS_OPTIMOS[2]:.4f}")
    elif mejor_key == 'XGBoost':
        print(f"\n⚡ Parámetros XGBoost utilizados:")
        print(f"   n_estimators: {PARAMS_XGB['n_estimators']}")
        print(f"   max_depth: {PARAMS_XGB['max_depth']}")
        print(f"   learning_rate: {PARAMS_XGB['learning_rate']}")
        print(f"\n⚖️  Pesos XGB:")
        print(f"   Local: {PESOS_XGB[0]:.4f}")
        print(f"   Empate: {PESOS_XGB[1]:.4f}")
        print(f"   Visitante: {PESOS_XGB[2]:.4f}")

    print(f"\n📁 Archivos guardados en {RUTA_MODELOS}")
    print(f"\n➡️  Siguiente: python 03_entrenar_sin_cuotas.py  o  python predecir_jornada_completa.py\n")

    return modelo_a_guardar, features


if __name__ == "__main__":
    modelo_final, features = main()