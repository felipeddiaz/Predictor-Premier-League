# -*- coding: utf-8 -*-
"""
Premier League Match Predictor
03 - ENTRENAMIENTO SIN CUOTAS (modelo para value betting puro)

Mismo pipeline que 02_entrenar_modelo.py pero SIN features de cuotas.
Las cuotas no entran en el entrenamiento: el modelo aprende solo de
rendimiento (forma, goles, xG, H2H, tabla). Luego se evalua el ROI
usando las cuotas como filtro externo post-prediccion.

Filosofia: si el modelo encuentra edge sobre el mercado SIN haberlo
"visto", la ventaja es estructural y no un artefacto de las cuotas.

Pipeline:
    datos/procesados/premier_league_con_features.csv
    → features sin cuotas (base + H2H + xG + tabla)
    → solo partidos con H2H disponible
    → feature selection top-15 (sobre train, sin leakage)
    → 80/20 split temporal
    → RF basico + RF balanceado + RF Optuna (PARAMS_OPTIMOS_VB)
    → CalibratedClassifierCV (sigmoid)
    → Evaluacion ROI por umbral de edge
    → modelos/modelo_value_betting.pkl

Para afinar PARAMS_OPTIMOS_VB: ejecutar visualizar_busqueda.py
con MODO_SIN_CUOTAS = True y copiar los valores al config.py.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
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
    ARCHIVO_MODELO_VB,
    ARCHIVO_FEATURES_VB,
    PARAMS_OPTIMOS_VB,
    PESOS_OPTIMOS,
    FEATURES_BASE,
    FEATURES_H2H,
    FEATURES_H2H_DERIVADAS,
    FEATURES_XG,
    FEATURES_TABLA,
    RANDOM_SEED,
)
from utils import agregar_xg_rolling, agregar_features_tabla

warnings.filterwarnings('ignore')

RUTA_DATOS = ARCHIVO_FEATURES
os.makedirs(RUTA_MODELOS, exist_ok=True)

# ============================================================================
# CARGA Y PREPARACIÓN
# ============================================================================

def cargar_datos():
    """Carga datos y prepara features SIN cuotas."""
    print("=" * 70)
    print("FASE 1: CARGANDO DATOS - MODELO SIN CUOTAS")
    print("=" * 70)

    if not os.path.exists(RUTA_DATOS):
        print(f"❌ ERROR: No se encontró '{RUTA_DATOS}'")
        print("   Ejecuta primero: python 01_preparar_datos.py")
        return None, None, None, None

    df = pd.read_csv(RUTA_DATOS)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    print(f"✅ Cargados: {len(df)} partidos")

    # Features calculadas en memoria (funciones canónicas de utils.py)
    df = agregar_xg_rolling(df)
    df = agregar_features_tabla(df)

    # Solo partidos con H2H disponible
    if 'H2H_Available' in df.columns:
        antes = len(df)
        df = df[df['H2H_Available'] == 1].copy()
        print(f"✅ Filtro H2H: {antes} → {len(df)} partidos con historial disponible")

    df = df.reset_index(drop=True)

    # Features disponibles (SIN cuotas ni cuotas derivadas)
    all_sin_cuotas = FEATURES_BASE + FEATURES_H2H + FEATURES_H2H_DERIVADAS + FEATURES_XG + FEATURES_TABLA
    features = [f for f in all_sin_cuotas if f in df.columns]

    print(f"\n📊 Features totales: {len(features)} (SIN cuotas)")
    print(f"   • Base:          {len([f for f in FEATURES_BASE if f in features])}")
    print(f"   • H2H:           {len([f for f in FEATURES_H2H if f in features])}")
    print(f"   • H2H derivadas: {len([f for f in FEATURES_H2H_DERIVADAS if f in features])}")
    print(f"   • xG rolling:    {len([f for f in FEATURES_XG if f in features])}")
    print(f"   • Tabla:         {len([f for f in FEATURES_TABLA if f in features])}")

    X = df[features].fillna(0)
    y = df['FTR_numeric']

    print(f"\n📊 Distribución de resultados:")
    for clase in [0, 1, 2]:
        nombre = ['Local', 'Empate', 'Visitante'][clase]
        count = (y == clase).sum()
        pct = count / len(y) * 100
        print(f"   {nombre}: {count} ({pct:.1f}%)")

    return X, y, features, df


# ============================================================================
# FEATURE SELECTION (top-15, solo sobre train para evitar leakage)
# ============================================================================

def seleccionar_top_features(X_train, y_train, X_test, features, top_n=15):
    """Selecciona las top_n features por importancia, entrenando solo sobre train."""
    print(f"\n🔝 Seleccionando top {top_n} features (sobre train únicamente)...")

    rf_temp = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf_temp.fit(X_train, y_train)

    importancias = rf_temp.feature_importances_
    top_idx = np.argsort(importancias)[::-1][:top_n]
    top_features = [features[i] for i in top_idx]

    print(f"\n   Top {top_n} features seleccionadas:")
    for i, (feat, idx) in enumerate(zip(top_features, top_idx), 1):
        print(f"   {i:2}. {feat:<30} {importancias[idx]:.4f}")

    return X_train[top_features], X_test[top_features], top_features


# ============================================================================
# ENTRENAMIENTO DE MODELOS
# ============================================================================

def entrenar_modelos(X_train, y_train, X_test, y_test):
    """Entrena RF básico, RF balanceado y RF Optuna. Devuelve resultados."""

    resultados = {}

    # -------------------------------------------------------------------------
    # MODELO 1: Random Forest Básico (baseline)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MODELO 1: RANDOM FOREST BÁSICO (Baseline)")
    print("=" * 70)

    rf_basico = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
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
        'nombre': 'Random Forest Básico',
    }
    print(f"✅ Accuracy: {acc:.2%} | F1-Score: {f1:.4f}")

    # -------------------------------------------------------------------------
    # MODELO 2: Random Forest Balanceado
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MODELO 2: RANDOM FOREST BALANCEADO (Auto)")
    print("=" * 70)

    rf_balanceado = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=RANDOM_SEED,
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
        'nombre': 'Random Forest Balanceado',
    }
    print(f"✅ Accuracy: {acc:.2%} | F1-Score: {f1:.4f}")

    # -------------------------------------------------------------------------
    # MODELO 3: Random Forest con PESOS OPTUNA
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MODELO 3: RANDOM FOREST CON PESOS OPTUNA ⭐")
    print("=" * 70)
    print(f"   Pesos optimizados (PARAMS_OPTIMOS_VB):")
    print(f"   • Local (0):       {PESOS_OPTIMOS[0]:.4f}")
    print(f"   • Empate (1):      {PESOS_OPTIMOS[1]:.4f}")
    print(f"   • Visitante (2):   {PESOS_OPTIMOS[2]:.4f}")
    print(f"   Hiperparámetros:")
    print(f"   • n_estimators:    {PARAMS_OPTIMOS_VB['n_estimators']}")
    print(f"   • max_depth:       {PARAMS_OPTIMOS_VB['max_depth']}")
    print(f"   • min_samples_leaf:{PARAMS_OPTIMOS_VB['min_samples_leaf']}")

    rf_optuna = RandomForestClassifier(**PARAMS_OPTIMOS_VB)
    rf_optuna.fit(X_train, y_train)
    pred_optuna = rf_optuna.predict(X_test)

    acc = accuracy_score(y_test, pred_optuna)
    f1 = f1_score(y_test, pred_optuna, average='weighted')

    resultados['RF_Optuna'] = {
        'modelo': rf_optuna,
        'predicciones': pred_optuna,
        'accuracy': acc,
        'f1_score': f1,
        'nombre': 'Random Forest Optuna (sin cuotas)',
    }
    print(f"\n✅ Accuracy: {acc:.2%} | F1-Score: {f1:.4f}")

    cm = confusion_matrix(y_test, pred_optuna)
    recall_local     = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
    recall_empate    = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0
    recall_visitante = cm[2, 2] / cm[2].sum() if cm[2].sum() > 0 else 0

    print(f"\n📊 Recall por clase:")
    print(f"   Local:     {recall_local:.2%}")
    print(f"   Empate:    {recall_empate:.2%}")
    print(f"   Visitante: {recall_visitante:.2%}")

    return resultados


# ============================================================================
# SELECCIÓN DEL MEJOR MODELO
# ============================================================================

def seleccionar_mejor_modelo(resultados, y_test):
    """Compara modelos y selecciona el mejor por F1-Score."""

    print("\n" + "=" * 70)
    print("FASE 2: COMPARACIÓN DE MODELOS")
    print("=" * 70)

    print(f"\n{'Modelo':<45} {'Accuracy':<12} {'F1-Score':<12}")
    print("-" * 70)

    for datos in resultados.values():
        print(f"{datos['nombre']:<45} {datos['accuracy']:>10.2%}  {datos['f1_score']:>10.4f}")

    mejor_key = max(resultados, key=lambda k: resultados[k]['f1_score'])
    mejor = resultados[mejor_key]

    print("\n" + "🏆" * 30)
    print(f"GANADOR: {mejor['nombre']}")
    print(f"   Accuracy: {mejor['accuracy']:.2%}")
    print(f"   F1-Score: {mejor['f1_score']:.4f}")
    print("🏆" * 30)

    print(f"\n📊 REPORTE DETALLADO:")
    print("-" * 65)
    print(classification_report(y_test, mejor['predicciones'],
                                 target_names=['Local', 'Empate', 'Visitante']))

    return mejor_key, mejor


# ============================================================================
# CALIBRACIÓN DE PROBABILIDADES
# ============================================================================

def calibrar_modelo(modelo, X_train, y_train, X_test, y_test):
    """Calibra el modelo con Platt Scaling para probabilidades realistas."""

    print("\n" + "=" * 70)
    print("FASE 3: CALIBRACIÓN DE PROBABILIDADES (Platt Scaling)")
    print("=" * 70)

    modelo_calibrado = CalibratedClassifierCV(
        estimator=modelo,
        method='sigmoid',
        cv=3
    )
    modelo_calibrado.fit(X_train, y_train)

    print("✅ Modelo calibrado con Platt Scaling (sigmoid, cv=3)")
    print("   Probabilidades más realistas para value betting")

    return modelo_calibrado


# ============================================================================
# EVALUACIÓN CON VALUE BETTING (diferencial del 03 vs el 02)
# ============================================================================

def evaluar_value_betting(modelo, X_test, y_test, df_test):
    """
    Evalúa el modelo usando cuotas como filtro EXTERNO post-predicción.
    Las cuotas NO entraron en el entrenamiento: si hay edge, es estructural.
    """

    print("\n" + "=" * 70)
    print("💰 EVALUACIÓN CON VALUE BETTING (cuotas como filtro externo)")
    print("=" * 70)

    if not all(col in df_test.columns for col in ['B365H', 'B365D', 'B365A']):
        print("⚠️  No hay cuotas disponibles para la evaluación de value betting")
        return

    y_proba = modelo.predict_proba(X_test)

    # Probabilidades del mercado (normalizadas para eliminar el margen)
    prob_h = 1 / df_test['B365H'].values
    prob_d = 1 / df_test['B365D'].values
    prob_a = 1 / df_test['B365A'].values
    total = prob_h + prob_d + prob_a
    prob_h /= total
    prob_d /= total
    prob_a /= total

    edge_home = y_proba[:, 0] - prob_h
    edge_draw = y_proba[:, 1] - prob_d
    edge_away = y_proba[:, 2] - prob_a

    umbrales = [0.00, 0.03, 0.05, 0.08, 0.10]

    print(f"\n📊 ROI SIMULADO POR UMBRAL DE EDGE (apuesta fija de 1 unidad):")
    print(f"{'Umbral':<10} {'Apuestas':<12} {'Accuracy':<12} {'ROI':<10}")
    print("-" * 50)

    for umbral in umbrales:
        max_edge = np.maximum(np.maximum(edge_home, edge_draw), edge_away)
        mask = max_edge > umbral
        num_apuestas = mask.sum()

        if num_apuestas == 0:
            print(f"{umbral:>6.1%}    {'0':>10}  {'N/A':>10}  {'N/A':>8}")
            continue

        predicciones_vb = []
        y_true_vb = []

        for i in range(len(y_proba)):
            if not mask[i]:
                continue
            mejor_apuesta = np.argmax([edge_home[i], edge_draw[i], edge_away[i]])
            predicciones_vb.append(mejor_apuesta)
            y_true_vb.append(y_test.iloc[i])

        acc_vb = accuracy_score(y_true_vb, predicciones_vb)

        roi = 0
        for i, idx in enumerate(np.where(mask)[0]):
            apuesta = predicciones_vb[i]
            real = y_true_vb[i]
            if apuesta == real:
                cuota = [df_test.iloc[idx]['B365H'],
                         df_test.iloc[idx]['B365D'],
                         df_test.iloc[idx]['B365A']][apuesta]
                roi += cuota - 1
            else:
                roi -= 1

        roi_pct = (roi / num_apuestas) * 100
        print(f"{umbral:>6.1%}    {num_apuestas:>10}  {acc_vb:>10.1%}  {roi_pct:>8.1f}%")

    print("\n💡 INTERPRETACIÓN:")
    print("   • Edge = modelo ve valor que el mercado NO vio (sin haberlo consultado)")
    print("   • Edge >5%  → apuestas con buena probabilidad")
    print("   • Edge >8%  → apuestas con alta confianza")
    print("   • ROI >0%   → estrategia rentable a largo plazo")
    print("\n🎯 RECOMENDACIÓN: usar edge >5% como filtro mínimo para apostar")


# ============================================================================
# VISUALIZACIONES
# ============================================================================

def visualizar_resultados(y_test, predictions, nombre_modelo, features, modelo):
    """Genera visualizaciones del modelo final."""

    print("\n" + "=" * 70)
    print("FASE 4: GENERANDO VISUALIZACIONES")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Matriz de confusión
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[0],
                xticklabels=['Local', 'Empate', 'Visitante'],
                yticklabels=['Local', 'Empate', 'Visitante'])
    axes[0].set_title(f'Matriz de Confusión\n{nombre_modelo}', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Real')
    axes[0].set_xlabel('Predicción')

    # Importancia de features (modelo base antes de calibrar)
    if hasattr(modelo, 'feature_importances_'):
        importances = modelo.feature_importances_
        df_imp = pd.DataFrame({
            'Feature': features,
            'Importancia': importances
        }).sort_values('Importancia', ascending=True).tail(15)

        df_imp.plot(x='Feature', y='Importancia', kind='barh', ax=axes[1],
                   legend=False, color='coral')
        axes[1].set_title('Top Features (modelo sin cuotas)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Importancia')

    plt.tight_layout()

    archivo = os.path.join(RUTA_MODELOS, 'modelo_value_betting.png')
    plt.savefig(archivo, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {archivo}")
    plt.close()


# ============================================================================
# GUARDAR MODELO
# ============================================================================

def guardar_modelo(modelo, features):
    """Guarda el modelo calibrado y la lista de features."""

    print("\n" + "=" * 70)
    print("FASE 5: GUARDANDO MODELO")
    print("=" * 70)

    joblib.dump(modelo, ARCHIVO_MODELO_VB)
    print(f"✅ Modelo: {ARCHIVO_MODELO_VB}")

    joblib.dump(features, ARCHIVO_FEATURES_VB)
    print(f"✅ Features: {ARCHIVO_FEATURES_VB}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Pipeline completo — modelo sin cuotas."""

    print("\n" + "⚽" * 35)
    print("   PREMIER LEAGUE - MODELO SIN CUOTAS")
    print("   (Value Betting puro — cuotas solo como filtro externo)")
    print("⚽" * 35 + "\n")

    # Cargar datos
    resultado = cargar_datos()
    if resultado[0] is None:
        return None, None
    X, y, features, df = resultado

    # Split temporal 80/20
    print("\n🔪 División de datos (80/20 temporal)")
    split_idx = int(len(X) * 0.8)
    X_train_full = X.iloc[:split_idx]
    X_test_full  = X.iloc[split_idx:]
    y_train      = y.iloc[:split_idx]
    y_test       = y.iloc[split_idx:]
    df_test      = df.iloc[split_idx:].reset_index(drop=True)

    print(f"   Entrenamiento: {len(X_train_full)} partidos")
    print(f"   Prueba:        {len(X_test_full)} partidos")

    # Feature selection top-15 (solo sobre train, sin leakage)
    X_train, X_test, features = seleccionar_top_features(
        X_train_full, y_train, X_test_full, features, top_n=15
    )

    # Entrenar los 3 modelos RF
    resultados = entrenar_modelos(X_train, y_train, X_test, y_test)

    # Seleccionar el mejor por F1
    mejor_key, mejor = seleccionar_mejor_modelo(resultados, y_test)

    # Calibrar probabilidades del ganador
    modelo_calibrado = calibrar_modelo(
        mejor['modelo'], X_train, y_train, X_test, y_test
    )

    nombre_final = f"{mejor['nombre']} (Calibrado)"

    # Evaluación value betting (diferencial exclusivo del 03)
    evaluar_value_betting(modelo_calibrado, X_test, y_test, df_test)

    # Visualizaciones (usa modelo base para feature_importances_)
    visualizar_resultados(y_test, mejor['predicciones'], nombre_final, features, mejor['modelo'])

    # Guardar modelo calibrado
    guardar_modelo(modelo_calibrado, features)

    # Métricas finales del modelo calibrado
    pred_calibrado = modelo_calibrado.predict(X_test)
    acc_final = accuracy_score(y_test, pred_calibrado)
    f1_final  = f1_score(y_test, pred_calibrado, average='weighted')

    print("\n" + "=" * 70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    print(f"\n🏆 Modelo guardado: {nombre_final}")
    print(f"📊 Accuracy (calibrado): {acc_final:.2%}")
    print(f"📊 F1-Score (calibrado): {f1_final:.4f}")
    print(f"\n📁 Archivos en {RUTA_MODELOS}")
    print(f"\n💡 Para afinar hiperparámetros:")
    print(f"   1. Abre visualizar_busqueda.py")
    print(f"   2. Cambia MODO_SIN_CUOTAS = True")
    print(f"   3. Ejecuta el script y copia los valores a PARAMS_OPTIMOS_VB en config.py")
    print(f"\n➡️  Siguiente: python predecir_jornada_completa.py\n")

    return modelo_calibrado, features


if __name__ == "__main__":
    modelo_final, features = main()
