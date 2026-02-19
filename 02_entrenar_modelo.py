# -*- coding: utf-8 -*-
"""
Premier League Match Predictor
02 - ENTRENAMIENTO DE MODELOS (VERSION OPTIMIZADA CON PESOS OPTUNA)

Entrena tres variantes de Random Forest (basico, balanceado, Optuna),
selecciona la mejor por F1-Score ponderado y aplica calibracion Platt
Scaling para obtener probabilidades realistas.

Pipeline:
    datos/procesados/premier_league_con_features.csv
    → features en memoria (xG rolling, tabla, cuotas derivadas)
    → 80/20 split temporal
    → RF basico + RF balanced + RF Optuna
    → CalibratedClassifierCV (sigmoid)
    → modelos/modelo_final_optimizado.pkl
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.calibration import CalibratedClassifierCV
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
    PARAMS_OPTIMOS,
    ALL_FEATURES,
    FEATURES_BASE,
    FEATURES_CUOTAS,
    FEATURES_CUOTAS_DERIVADAS,
    FEATURES_XG,
    FEATURES_H2H,
    FEATURES_H2H_DERIVADAS,
    FEATURES_TABLA,
    TEST_SIZE,
    RANDOM_SEED,
)
from utils import agregar_xg_rolling, agregar_features_tabla, agregar_features_cuotas_derivadas

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
    
    return resultados

# ============================================================================
# CALIBRACIÓN DE PROBABILIDADES
# ============================================================================

def calibrar_modelo(modelo, X_train, y_train, X_test, y_test):
    """
    Calibra el modelo para obtener probabilidades más realistas.
    Usa validación cruzada isotónica (Platt scaling o isotónico).
    """
    print("\n" + "="*70)
    print("CALIBRACIÓN DE PROBABILIDADES")
    print("="*70)
    
    # Crear modelo calibrado usando validación cruzada
    modelo_calibrado = CalibratedClassifierCV(
        estimator=modelo,
        method='sigmoid',  # Platt scaling (menos propenso al overfitting)
        cv=3  # 3-fold cross-validation
    )
    
    # Entrenar con datos de calibración (usamos X_train)
    modelo_calibrado.fit(X_train, y_train)
    
    # Obtener probabilidades calibradas
    probs_calibradas = modelo_calibrado.predict_proba(X_test)
    
    print(f"✅ Modelo calibrado usando Platt Scaling (sigmoid)")
    print(f"   Probabilidades ahora más realistas para predicciones")
    
    return modelo_calibrado, probs_calibradas

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
    
    # Si el modelo Optuna ganó, no necesita más optimización
    if mejor_key == 'RF_Optuna':
        print("\n" + "="*70)
        print("FASE 3: OPTIMIZACIÓN ADICIONAL")
        print("="*70)
        print("\n✅ El modelo Optuna ya está optimizado. Saltando esta fase.")
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
    
    # Importancia de características
    if hasattr(modelo, 'feature_importances_'):
        importances = modelo.feature_importances_
        
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
    print("\n🔪 División de datos (80/20 temporal)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
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
    
    # Calibrar probabilidades
    modelo_calibrado, probs_calibradas = calibrar_modelo(
        modelo_final, X_train, y_train, X_test, y_test
    )

    nombre_final = f"{mejor_modelo['nombre']} {'(Calibrado)' if mejor_key == 'RF_Optuna' else '(Opt+Cal)'}"

    # Visualizaciones (usa modelo base para feature_importances_)
    visualizar_resultados(y_test, pred_final, nombre_final, features, modelo_final)

    # Guardar modelo calibrado (el que realmente se usa en prediccion)
    guardar_modelo_final(modelo_calibrado, features, nombre_final)

    # Resumen — se reportan las metricas del modelo calibrado (consistente con lo guardado)
    pred_calibrado = modelo_calibrado.predict(X_test)
    acc_final = accuracy_score(y_test, pred_calibrado)
    f1_final = f1_score(y_test, pred_calibrado, average='weighted')

    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"\n🏆 Modelo guardado: {nombre_final}")
    print(f"📊 Accuracy (calibrado): {acc_final:.2%}")
    print(f"📊 F1-Score (calibrado): {f1_final:.4f}")

    if mejor_key == 'RF_Optuna':
        print(f"\n⭐ Pesos Optuna utilizados:")
        print(f"   Local: {PESOS_OPTIMOS[0]:.4f}")
        print(f"   Empate: {PESOS_OPTIMOS[1]:.4f}")
        print(f"   Visitante: {PESOS_OPTIMOS[2]:.4f}")

    print(f"\n📁 Archivos guardados en {RUTA_MODELOS}")
    print(f"\n➡️  Siguiente: python 03_entrenar_con_cuotas.py  o  python predecir_jornada_completa.py\n")

    return modelo_calibrado, features


if __name__ == "__main__":
    modelo_final, features = main()