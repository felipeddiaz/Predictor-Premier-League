# -*- coding: utf-8 -*-
"""
Premier League Match Predictor
02 - ENTRENAMIENTO DE MODELOS (VERSION OPTIMIZADA CON PESOS OPTUNA + XGBOOST)

Entrena cuatro variantes (RF basico, RF balanceado, RF Optuna, XGBoost),
selecciona la mejor por Log Loss (calidad probabilistica) y aplica
calibracion Platt Scaling para obtener probabilidades realistas.

Metricas de optimizacion:
    Primaria:    Log Loss (calidad probabilistica)
    Secundaria:  Brier Score (error cuadratico de probs)
    Terciaria:   ROI simulado (value betting)
    Referencia:  F1-weighted, Accuracy

Pipeline:
    datos/procesados/premier_league_con_features.csv
    → features en memoria (tabla, cuotas, AH, rolling, forma, pinnacle, arbitro)
    → walk-forward season-by-season (diagnostico de consistencia)
    → 80/20 split temporal → entrenar 4 modelos → seleccionar por Log Loss
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
    ARCHIVO_MODELO_VB,
    ARCHIVO_FEATURES_VB,
    PESOS_OPTIMOS,
    PESOS_XGB,
    PARAMS_OPTIMOS,
    PARAMS_XGB,
    PARAMS_XGB_VB,
    ALL_FEATURES,
    FEATURES_ESTRUCTURALES,
    FEATURES_BASE,
    FEATURES_CUOTAS,
    FEATURES_CUOTAS_DERIVADAS,
    FEATURES_XG,
    FEATURES_XG_GLOBAL,
    FEATURES_H2H,
    FEATURES_H2H_DERIVADAS,
    FEATURES_TABLA,
    FEATURES_ASIAN_HANDICAP,
    FEATURES_ROLLING_EXTRA,
    FEATURES_PINNACLE,
    FEATURES_REFEREE,
    FEATURES_FORMA_MOMENTUM,
    FEATURES_DESCANSO,
    FEATURES_MULTI_ESCALA,
    FEATURES_ELO,
    TEST_SIZE,
    RANDOM_SEED,
    N_FEATURES_SELECCION,
)
from utils import (
    agregar_features_tabla,
    agregar_features_cuotas_derivadas,
    agregar_features_asian_handicap,
    agregar_features_rolling_extra,
    agregar_features_multi_escala,
    agregar_features_forma_momentum,
    agregar_features_pinnacle_move,
    agregar_features_arbitro,
    agregar_features_elo,
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
    # Nota: xG rolling ya se calcula en 01_preparar_datos.py y se guarda al CSV
    df = agregar_features_tabla(df)
    df = agregar_features_cuotas_derivadas(df)
    df = agregar_features_asian_handicap(df)
    df = agregar_features_rolling_extra(df)
    df = agregar_features_multi_escala(df)
    df = agregar_features_forma_momentum(df)
    df = agregar_features_pinnacle_move(df)
    df = agregar_features_arbitro(df)
    df = agregar_features_elo(df)

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
    print(f"   • Pinnacle: {len([f for f in FEATURES_PINNACLE if f in features])}")
    print(f"   • Referee: {len([f for f in FEATURES_REFEREE if f in features])}")
    print(f"   • Forma/Momentum: {len([f for f in FEATURES_FORMA_MOMENTUM if f in features])}")
    print(f"   • xG Global: {len([f for f in FEATURES_XG_GLOBAL if f in features])}")
    print(f"   • Multi-escala (w=10): {len([f for f in FEATURES_MULTI_ESCALA if f in features])}")
    print(f"   • Descanso/Fatiga: {len([f for f in FEATURES_DESCANSO if f in features])}")
    print(f"   • Elo ratings: {len([f for f in FEATURES_ELO if f in features])}")

    # Info de H2H
    if 'H2H_Available' in features:
        con_h2h = df['H2H_Available'].sum()
        sin_h2h = len(df) - con_h2h
        print(f"\n✅ H2H disponible:")
        print(f"   Partidos CON H2H: {con_h2h} ({con_h2h/len(df)*100:.1f}%)")
        print(f"   Partidos SIN H2H: {sin_h2h} ({sin_h2h/len(df)*100:.1f}%)")
    
    # X con NaN para XGBoost (maneja NaN nativamente)
    # X_filled para Random Forest (no soporta NaN)
    X = df[features]
    X_filled = X.fillna(0)
    y = df['FTR_numeric']
    
    # Distribución
    print(f"\n📊 Distribución de resultados:")
    for clase in [0, 1, 2]:
        nombre = ['Local', 'Empate', 'Visitante'][clase]
        count = (y == clase).sum()
        pct = count / len(y) * 100
        print(f"   {nombre}: {count} ({pct:.1f}%)")
    
    return X, X_filled, y, features, df

# ============================================================================
# MÉTRICAS
# ============================================================================

def _brier_multiclase(y_true, probs):
    """Brier Score multiclase: promedio del Brier por clase (one-vs-rest)."""
    bs = 0.0
    for clase in range(3):
        y_bin = (y_true == clase).astype(int)
        bs += brier_score_loss(y_bin, probs[:, clase])
    return bs / 3


def _roi_simulado(y_true, probs, df_cuotas=None, edge_minimo=0.10):
    """
    ROI simulado con apuestas planas sobre value bets.

    Para cada partido donde max(prob_modelo) - prob_mercado > edge_minimo,
    apuesta 1 unidad al resultado con mayor edge. Calcula el ROI total.

    Si df_cuotas es None (no hay cuotas disponibles en el split),
    retorna None.
    """
    if df_cuotas is None:
        return None

    cuota_cols = ['B365H', 'B365D', 'B365A']
    if not all(c in df_cuotas.columns for c in cuota_cols):
        return None

    cuotas = df_cuotas[cuota_cols].values  # shape (n, 3)
    apostado = 0.0
    ganancia = 0.0

    for i in range(len(y_true)):
        # Probabilidades del mercado (implícitas, sin normalizar)
        prob_mercado = 1.0 / cuotas[i]
        if np.any(np.isnan(prob_mercado)) or np.any(cuotas[i] <= 1.0):
            continue

        # Edge por resultado
        edges = probs[i] - prob_mercado
        mejor_resultado = int(np.argmax(edges))
        mejor_edge = edges[mejor_resultado]

        if mejor_edge >= edge_minimo:
            apostado += 1.0
            if y_true.iloc[i] == mejor_resultado:
                ganancia += cuotas[i][mejor_resultado] - 1.0
            else:
                ganancia -= 1.0

    if apostado == 0:
        return None

    return ganancia / apostado


def _evaluar_modelo(modelo, X_test, y_test, nombre, df_cuotas=None):
    """Evalúa un modelo con todas las métricas (probabilísticas + clasificación)."""
    pred = modelo.predict(X_test)
    probs = modelo.predict_proba(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    ll = log_loss(y_test, probs)
    bs = _brier_multiclase(y_test, probs)
    roi = _roi_simulado(y_test, probs, df_cuotas)

    return {
        'modelo': modelo,
        'predicciones': pred,
        'probs': probs,
        'accuracy': acc,
        'f1_score': f1,
        'log_loss': ll,
        'brier_score': bs,
        'roi': roi,
        'nombre': nombre,
    }


# ============================================================================
# ENTRENAMIENTO DE MODELOS
# ============================================================================

def entrenar_modelos(X_train, y_train, X_test, y_test,
                     X_train_filled=None, X_test_filled=None,
                     df_cuotas_test=None):
    """Entrena los modelos RF (con fillna) y XGBoost (con NaN nativo).

    Métricas primarias: Log Loss, Brier Score (calidad probabilística).
    Métricas de referencia: F1-weighted, Accuracy (clasificación).
    Métrica terciaria: ROI simulado (si hay cuotas disponibles).
    """
    if X_train_filled is None:
        X_train_filled = X_train.fillna(0)
    if X_test_filled is None:
        X_test_filled = X_test.fillna(0)

    resultados = {}

    def _print_metricas(r):
        roi_str = f" | ROI: {r['roi']:+.2%}" if r['roi'] is not None else ""
        print(f"✅ Log Loss: {r['log_loss']:.4f} | Brier: {r['brier_score']:.4f} "
              f"| F1: {r['f1_score']:.4f} | Acc: {r['accuracy']:.2%}{roi_str}")

    # --- MODELO 1: Random Forest Básico ---
    print("\n" + "="*70)
    print("MODELO 1: RANDOM FOREST BÁSICO (Baseline)")
    print("="*70)
    rf_basico = RandomForestClassifier(
        n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf_basico.fit(X_train_filled, y_train)
    resultados['RF_Basico'] = _evaluar_modelo(
        rf_basico, X_test_filled, y_test, 'Random Forest Básico', df_cuotas_test)
    _print_metricas(resultados['RF_Basico'])

    # --- MODELO 2: Random Forest Balanceado ---
    print("\n" + "="*70)
    print("MODELO 2: RANDOM FOREST BALANCEADO (Auto)")
    print("="*70)
    rf_balanceado = RandomForestClassifier(
        n_estimators=100, min_samples_leaf=5, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    rf_balanceado.fit(X_train_filled, y_train)
    resultados['RF_Balanceado'] = _evaluar_modelo(
        rf_balanceado, X_test_filled, y_test, 'Random Forest Balanceado', df_cuotas_test)
    _print_metricas(resultados['RF_Balanceado'])

    # --- MODELO 3: Random Forest Optuna ---
    print("\n" + "="*70)
    print("MODELO 3: RANDOM FOREST CON PESOS OPTUNA")
    print("="*70)
    rf_optuna = RandomForestClassifier(**PARAMS_OPTIMOS)
    rf_optuna.fit(X_train_filled, y_train)
    resultados['RF_Optuna'] = _evaluar_modelo(
        rf_optuna, X_test_filled, y_test, 'Random Forest Optuna', df_cuotas_test)
    _print_metricas(resultados['RF_Optuna'])

    # --- MODELO 4: XGBoost ---
    print("\n" + "="*70)
    print("MODELO 4: XGBOOST")
    print("="*70)
    sample_weights_train = compute_sample_weight(
        class_weight=PESOS_XGB, y=y_train
    )
    xgb_model = XGBClassifier(**PARAMS_XGB)
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights_train)
    resultados['XGBoost'] = _evaluar_modelo(
        xgb_model, X_test, y_test, 'XGBoost', df_cuotas_test)
    _print_metricas(resultados['XGBoost'])

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
    """Compara modelos y selecciona el mejor según ROI (value betting)."""

    print("\n" + "="*70)
    print("FASE 2: COMPARACIÓN DE MODELOS (Criterio: ROI)")
    print("="*70)

    header = f"{'Modelo':<30} {'ROI':>8} {'Log Loss':>9} {'Brier':>7} {'F1':>7} {'Acc':>7}"
    print(f"\n{header}")
    print("-" * len(header))

    for key, datos in resultados.items():
        roi_str = f"{datos['roi']:>+7.2%}" if datos['roi'] is not None else f"{'N/A':>8}"
        linea = (f"{datos['nombre']:<30} {roi_str} {datos['log_loss']:>9.4f} "
                 f"{datos['brier_score']:>7.4f} {datos['f1_score']:>7.4f} "
                 f"{datos['accuracy']:>6.2%}")
        print(linea)

    # Elegir mejor por ROI (mayor = mejor para value betting)
    # Filtrar modelos con ROI disponible
    con_roi = {k: v for k, v in resultados.items() if v['roi'] is not None}
    if con_roi:
        mejor_key = max(con_roi.items(), key=lambda x: x[1]['roi'])[0]
    else:
        # Fallback a Log Loss si no hay ROI disponible
        mejor_key = min(resultados.items(), key=lambda x: x[1]['log_loss'])[0]
    mejor = resultados[mejor_key]

    print(f"\n{'='*60}")
    print(f"GANADOR: {mejor['nombre']}  (ROI: {mejor['roi']:+.2%})")
    print(f"   Log Loss:    {mejor['log_loss']:.4f}")
    print(f"   Brier Score: {mejor['brier_score']:.4f}")
    print(f"   F1-Score:    {mejor['f1_score']:.4f} (referencia)")
    print(f"   Accuracy:    {mejor['accuracy']:.2%} (referencia)")
    print(f"{'='*60}")

    # Reporte detallado
    print(f"\n📊 REPORTE DETALLADO:")
    print("-" * 65)
    target_names = ['Local', 'Empate', 'Visitante']
    print(classification_report(y_test, mejor['predicciones'], target_names=target_names))

    return mejor_key, mejor


# ============================================================================
# OPTIMIZACIÓN ADICIONAL (opcional)
# ============================================================================

def optimizar_modelo_adicional(mejor_key, mejor_modelo, X_train, y_train, X_test, y_test,
                               X_train_filled=None, X_test_filled=None):
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
        scoring='neg_log_loss'   # optimizar calidad probabilística, no F1
    )

    # RF no soporta NaN — usar versión filled
    _X_tr = X_train_filled if X_train_filled is not None else X_train.fillna(0)
    _X_te = X_test_filled if X_test_filled is not None else X_test.fillna(0)
    random_search.fit(_X_tr, y_train)

    # Evaluar con métricas probabilísticas
    modelo_optimizado = random_search.best_estimator_
    probs_opt = modelo_optimizado.predict_proba(_X_te)
    pred_optimizado = modelo_optimizado.predict(_X_te)
    ll_opt = log_loss(y_test, probs_opt)
    bs_opt = _brier_multiclase(y_test, probs_opt)
    f1_opt = f1_score(y_test, pred_optimizado, average='weighted')

    print("\n✅ OPTIMIZACIÓN COMPLETADA")
    print("="*70)
    print("\n📋 MEJORES PARÁMETROS:")
    for param, value in random_search.best_params_.items():
        print(f"   {param}: {value}")

    ll_antes = mejor_modelo['log_loss']
    bs_antes = mejor_modelo['brier_score']

    print("\n📊 COMPARACIÓN ANTES/DESPUÉS:")
    print(f"{'Métrica':<15} {'Antes':>13} {'Después':>13} {'Cambio':>10}")
    print("-" * 55)
    print(f"{'Log Loss':<15} {ll_antes:>13.4f} {ll_opt:>13.4f} {ll_opt - ll_antes:>+10.4f}")
    print(f"{'Brier Score':<15} {bs_antes:>13.4f} {bs_opt:>13.4f} {bs_opt - bs_antes:>+10.4f}")
    print(f"{'F1 (ref.)':<15} {mejor_modelo['f1_score']:>13.4f} {f1_opt:>13.4f} {f1_opt - mejor_modelo['f1_score']:>+10.4f}")

    # Decisión por Log Loss (menor = mejor)
    if ll_opt < ll_antes:
        print(f"\n   Optimización MEJORA Log Loss ({ll_antes:.4f} → {ll_opt:.4f})")
        return modelo_optimizado, pred_optimizado, True
    else:
        print("\n   Sin mejora en Log Loss. Usando el original.")
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
# CALIBRAR SHRINKAGE (FACTOR_CONSERVADOR)
# ============================================================================

def calibrar_shrinkage(modelo, X_val, y_val):
    """
    Encuentra el alpha óptimo para el ajuste conservador de probabilidades.

    probs_adj = alpha * probs_modelo + (1-alpha) * [1/3, 1/3, 1/3]

    Grid search sobre alpha in [0.50, 0.55, ..., 1.00] minimizando Brier Score.
    También prueba Temperature Scaling (un solo parámetro T) como alternativa.

    Retorna (alpha_optimo, mejor_brier, metodo) donde metodo es 'shrinkage' o 'temperature'.
    """
    print("\n" + "="*70)
    print("CALIBRACION DE SHRINKAGE (FACTOR_CONSERVADOR)")
    print("="*70)

    probs_raw = modelo.predict_proba(X_val)
    uniforme = np.array([1/3, 1/3, 1/3])

    # Grid search: shrinkage
    mejor_alpha = 1.0
    mejor_brier_shrink = float('inf')
    resultados_alpha = []

    for alpha_int in range(50, 101, 5):
        alpha = alpha_int / 100.0
        probs_adj = alpha * probs_raw + (1 - alpha) * uniforme
        probs_adj = probs_adj / probs_adj.sum(axis=1, keepdims=True)
        bs = _brier_multiclase(y_val, probs_adj)
        resultados_alpha.append((alpha, bs))
        if bs < mejor_brier_shrink:
            mejor_brier_shrink = bs
            mejor_alpha = alpha

    # Temperature Scaling: divide logits por T, luego softmax
    # Buscar T in [0.5, 0.6, ..., 2.0]
    mejor_T = 1.0
    mejor_brier_temp = float('inf')

    eps = 1e-15
    logits = np.log(np.clip(probs_raw, eps, 1.0))

    for T_int in range(50, 201, 10):
        T = T_int / 100.0
        scaled = np.exp(logits / T)
        probs_temp = scaled / scaled.sum(axis=1, keepdims=True)
        bs = _brier_multiclase(y_val, probs_temp)
        if bs < mejor_brier_temp:
            mejor_brier_temp = bs
            mejor_T = T

    # Reportar
    brier_raw = _brier_multiclase(y_val, probs_raw)

    print(f"\n   Brier Score RAW (sin ajuste):       {brier_raw:.4f}")
    print(f"\n   Shrinkage grid search:")
    for alpha, bs in resultados_alpha:
        marker = " <-- mejor" if abs(alpha - mejor_alpha) < 0.01 else ""
        print(f"     alpha={alpha:.2f}  Brier={bs:.4f}{marker}")

    print(f"\n   Mejor shrinkage: alpha={mejor_alpha:.2f} -> Brier={mejor_brier_shrink:.4f}")
    print(f"   Temperature Scaling: T={mejor_T:.2f} -> Brier={mejor_brier_temp:.4f}")

    if mejor_brier_temp < mejor_brier_shrink:
        print(f"\n   -> Temperature Scaling es mejor (T={mejor_T:.2f})")
        return mejor_alpha, mejor_brier_shrink, mejor_T, mejor_brier_temp, 'temperature'
    else:
        print(f"\n   -> Shrinkage es mejor (alpha={mejor_alpha:.2f})")
        return mejor_alpha, mejor_brier_shrink, mejor_T, mejor_brier_temp, 'shrinkage'


# ============================================================================
# GUARDAR MODELO FINAL
# ============================================================================

def guardar_modelo_final(modelo, features, nombre_modelo, alpha_shrinkage=0.60):
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
        'alpha_shrinkage': alpha_shrinkage,
    }
    joblib.dump(metadata, ARCHIVO_METADATA)
    print(f"✅ Metadata: {ARCHIVO_METADATA}")
    print(f"   alpha_shrinkage: {alpha_shrinkage:.2f}")


# ============================================================================
# WALK-FORWARD SEASON-BY-SEASON
# ============================================================================

def _asignar_temporada(dates: pd.Series) -> pd.Series:
    """Asigna temporada tipo '2020-21' basándose en la fecha."""
    return dates.apply(
        lambda d: f"{d.year}-{str(d.year+1)[-2:]}" if d.month >= 8
        else f"{d.year-1}-{str(d.year)[-2:]}"
    )


def walk_forward_temporal(df, features):
    """
    Validación walk-forward expandida season-by-season.

    Train: 2016-2020  →  Test: 2020-21
    Train: 2016-2021  →  Test: 2021-22
    Train: 2016-2022  →  Test: 2022-23
    Train: 2016-2023  →  Test: 2023-24
    Train: 2016-2024  →  Test: 2024-25

    Entrena XGBoost (mejor modelo actual) en cada fold y reporta
    métricas probabilísticas por temporada. Revela si el rendimiento
    es consistente o depende de una temporada anómala.
    """
    print("\n" + "="*70)
    print("WALK-FORWARD TEMPORAL (season-by-season)")
    print("="*70)

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['_Season'] = _asignar_temporada(df['Date'])

    temporadas = sorted(df['_Season'].unique())
    print(f"   Temporadas disponibles: {', '.join(temporadas)}")

    # Definir folds: test = cada temporada desde 2020-21
    test_seasons = [s for s in temporadas if s >= '2020-21']
    if not test_seasons:
        print("   No hay temporadas >= 2020-21 para walk-forward")
        return []

    resultados_wf = []

    for test_season in test_seasons:
        # Train = todas las temporadas anteriores
        train_mask = df['_Season'] < test_season
        test_mask = df['_Season'] == test_season

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_train = df.loc[train_mask, features]
        y_train = df.loc[train_mask, 'FTR_numeric']
        X_test = df.loc[test_mask, features]
        y_test = df.loc[test_mask, 'FTR_numeric']

        # Cuotas para ROI
        df_cuotas_test = df.loc[test_mask] if 'B365H' in df.columns else None

        # Entrenar XGBoost
        sample_weights = compute_sample_weight(class_weight=PESOS_XGB, y=y_train)
        modelo = XGBClassifier(**PARAMS_XGB)
        modelo.fit(X_train, y_train, sample_weight=sample_weights)

        probs = modelo.predict_proba(X_test)
        pred = modelo.predict(X_test)

        ll = log_loss(y_test, probs)
        bs = _brier_multiclase(y_test, probs)
        f1 = f1_score(y_test, pred, average='weighted')
        acc = accuracy_score(y_test, pred)
        roi = _roi_simulado(y_test, probs, df_cuotas_test)

        fold = {
            'season': test_season,
            'train_size': train_mask.sum(),
            'test_size': test_mask.sum(),
            'log_loss': ll,
            'brier_score': bs,
            'f1_score': f1,
            'accuracy': acc,
            'roi': roi,
        }
        resultados_wf.append(fold)

    # Imprimir tabla de resultados
    print(f"\n{'Test Season':<12} {'Train':>6} {'Test':>5} "
          f"{'Log Loss':>9} {'Brier':>7} {'F1':>7} {'Acc':>7} {'ROI':>8}")
    print("-" * 75)

    for r in resultados_wf:
        roi_str = f"{r['roi']:>+7.2%}" if r['roi'] is not None else f"{'N/A':>8}"
        print(f"{r['season']:<12} {r['train_size']:>6} {r['test_size']:>5} "
              f"{r['log_loss']:>9.4f} {r['brier_score']:>7.4f} "
              f"{r['f1_score']:>7.4f} {r['accuracy']:>6.2%} {roi_str}")

    # Promedios
    avg_ll = np.mean([r['log_loss'] for r in resultados_wf])
    avg_bs = np.mean([r['brier_score'] for r in resultados_wf])
    avg_f1 = np.mean([r['f1_score'] for r in resultados_wf])
    avg_acc = np.mean([r['accuracy'] for r in resultados_wf])
    rois = [r['roi'] for r in resultados_wf if r['roi'] is not None]
    avg_roi = np.mean(rois) if rois else None

    roi_avg_str = f"{avg_roi:>+7.2%}" if avg_roi is not None else f"{'N/A':>8}"
    print("-" * 75)
    print(f"{'PROMEDIO':<12} {'':>6} {'':>5} "
          f"{avg_ll:>9.4f} {avg_bs:>7.4f} {avg_f1:>7.4f} {avg_acc:>6.2%} {roi_avg_str}")

    # Desviación estándar para ver consistencia
    std_ll = np.std([r['log_loss'] for r in resultados_wf])
    std_bs = np.std([r['brier_score'] for r in resultados_wf])
    print(f"{'STD':<12} {'':>6} {'':>5} "
          f"{std_ll:>9.4f} {std_bs:>7.4f}")

    # Diagnóstico
    if std_ll > 0.05:
        print("\n   ALERTA: Log Loss varía significativamente entre temporadas.")
        print("   El modelo puede no ser estable.")
    else:
        print("\n   Log Loss consistente entre temporadas.")

    df.drop(columns=['_Season'], inplace=True, errors='ignore')
    return resultados_wf


# ============================================================================
# FEATURE SELECTION (Top-N por importancia XGBoost)
# ============================================================================

def seleccionar_features(X_train, y_train, features, n_top=N_FEATURES_SELECCION):
    """
    Entrena un XGBoost rápido y selecciona las top-N features por importancia.

    Reduce overfitting al eliminar features ruidosas. Retorna la lista
    ordenada de features seleccionadas.
    """
    print("\n" + "="*70)
    print(f"SELECCION DE FEATURES (Top {n_top} de {len(features)})")
    print("="*70)

    sample_weights = compute_sample_weight(class_weight=PESOS_XGB, y=y_train)
    selector = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric='mlogloss',
    )
    selector.fit(X_train, y_train, sample_weight=sample_weights)

    importances = selector.feature_importances_
    df_imp = pd.DataFrame({
        'Feature': features,
        'Importancia': importances,
    }).sort_values('Importancia', ascending=False)

    top_features = df_imp.head(n_top)['Feature'].tolist()

    print(f"\n   Top {n_top} features seleccionadas:")
    for i, (_, row) in enumerate(df_imp.head(n_top).iterrows(), 1):
        print(f"   {i:2d}. {row['Feature']:<30} {row['Importancia']:.4f}")

    # Features eliminadas
    eliminadas = df_imp.tail(len(features) - n_top)
    print(f"\n   Eliminadas ({len(eliminadas)}):")
    for _, row in eliminadas.iterrows():
        print(f"       {row['Feature']:<30} {row['Importancia']:.4f}")

    return top_features


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Pipeline completo con feature selection y Elo ratings. Selecciona por ROI."""

    print("\n" + "="*70)
    print("   PREMIER LEAGUE - ENTRENAMIENTO DE MODELOS")
    print("   (Feature Selection + Elo Ratings — Optimizado para ROI)")
    print("="*70 + "\n")

    # Cargar datos
    resultado = cargar_datos()
    if resultado[0] is None:
        return None, None

    X, X_filled, y, features_all, df = resultado

    # ── Split 80/20 para entrenamiento final ──────────────────────────────
    pct_label = f"{int(TEST_SIZE*100)}"
    print(f"\n   Division de datos ({100-int(TEST_SIZE*100)}/{pct_label} temporal)")
    X_train_all, X_test_all, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False
    )
    X_train_all_filled = X_filled.loc[X_train_all.index]
    X_test_all_filled = X_filled.loc[X_test_all.index]
    df_cuotas_test = df.loc[X_test_all.index] if 'B365H' in df.columns else None
    print(f"   Entrenamiento: {len(X_train_all)} partidos")
    print(f"   Prueba: {len(X_test_all)} partidos")

    # ── P1: Seleccion de features (top N por importancia) ─────────────────
    top_features = seleccionar_features(X_train_all, y_train, features_all)

    # Subseleccionar features
    X_train = X_train_all[top_features]
    X_test = X_test_all[top_features]
    X_train_filled = X_train.fillna(0)
    X_test_filled = X_test.fillna(0)
    features = top_features

    # ── Entrenar modelos individuales (RF, XGBoost) ───────────────────────
    resultados = entrenar_modelos(X_train, y_train, X_test, y_test,
                                  X_train_filled, X_test_filled,
                                  df_cuotas_test)

    # Seleccionar mejor (por ROI — objetivo: value betting)
    mejor_key, mejor_modelo = seleccionar_mejor_modelo(resultados, y_test)

    modelo_final = mejor_modelo['modelo']
    pred_final = mejor_modelo['predicciones']

    # ── Calibracion de probabilidades ─────────────────────────────────────
    es_xgb = mejor_key == 'XGBoost'
    _X_tr_cal = X_train if es_xgb else X_train_filled
    _X_te_cal = X_test if es_xgb else X_test_filled

    cal_split = int(len(_X_tr_cal) * 0.80)
    X_cal_train = _X_tr_cal.iloc[:cal_split]
    y_cal_train = y_train.iloc[:cal_split]

    modelo_a_guardar, probs_finales, fue_calibrado = calibrar_modelo(
        modelo_final, X_cal_train, y_cal_train, _X_te_cal, y_test
    )

    tag_cal = "(Calibrado)" if fue_calibrado else "(Sin Calibrar)"
    nombre_final = f"{mejor_modelo['nombre']} {tag_cal}"

    _X_te_eval = _X_te_cal

    # Calibrar shrinkage (FACTOR_CONSERVADOR)
    alpha_opt, _, _, _, metodo = calibrar_shrinkage(modelo_a_guardar, _X_te_eval, y_test)

    # Visualizaciones
    visualizar_resultados(y_test, pred_final, nombre_final, features, modelo_final)

    # Guardar modelo
    guardar_modelo_final(modelo_a_guardar, features, nombre_final, alpha_shrinkage=alpha_opt)

    # Métricas finales
    probs_final = modelo_a_guardar.predict_proba(_X_te_eval)
    pred_guardado = modelo_a_guardar.predict(_X_te_eval)
    ll_final = log_loss(y_test, probs_final)
    bs_final = _brier_multiclase(y_test, probs_final)
    f1_final = f1_score(y_test, pred_guardado, average='weighted')
    acc_final = accuracy_score(y_test, pred_guardado)
    roi_final = _roi_simulado(y_test, probs_final, df_cuotas_test)

    print("\n" + "="*70)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"\n   Modelo guardado: {nombre_final}")
    print(f"   Features: {len(features)} (de {len(features_all)} originales)")
    print(f"\n   Metricas primarias (calidad probabilistica):")
    print(f"   Log Loss:    {ll_final:.4f}")
    print(f"   Brier Score: {bs_final:.4f}")
    if roi_final is not None:
        print(f"   ROI sim.:    {roi_final:+.2%}")
    print(f"\n   Metricas de referencia (clasificacion):")
    print(f"   Accuracy:    {acc_final:.2%}")
    print(f"   F1-Score:    {f1_final:.4f}")

    print(f"\n   Archivos guardados en {RUTA_MODELOS}\n")

    # ── Walk-forward temporal con features seleccionadas ──────────────────
    walk_forward_temporal(df, features)

    # ── Modelo estructural (sin cuotas) ───────────────────────────────────
    print("\n" + "=" * 70)
    print("MODELO ESTRUCTURAL (SIN CUOTAS)")
    print("=" * 70)

    features_estr = [f for f in FEATURES_ESTRUCTURALES if f in df.columns]
    print(f"   Features estructurales: {len(features_estr)} (sin cuotas)")

    X_estr = df[features_estr]
    X_estr_train = X_estr.iloc[X_train_all.index]
    X_estr_test = X_estr.iloc[X_test_all.index]

    sample_weights_estr = compute_sample_weight(class_weight=PESOS_XGB, y=y_train)
    xgb_estr = XGBClassifier(**PARAMS_XGB_VB)
    xgb_estr.fit(X_estr_train, y_train, sample_weight=sample_weights_estr,
                 eval_set=[(X_estr_test, y_test)], verbose=False)

    probs_estr = xgb_estr.predict_proba(X_estr_test)
    pred_estr = xgb_estr.predict(X_estr_test)
    ll_estr = log_loss(y_test, probs_estr)
    bs_estr = _brier_multiclase(y_test, probs_estr)
    f1_estr = f1_score(y_test, pred_estr, average='weighted')
    acc_estr = accuracy_score(y_test, pred_estr)

    print(f"   Log Loss:    {ll_estr:.4f}")
    print(f"   Brier Score: {bs_estr:.4f}")
    print(f"   F1-Score:    {f1_estr:.4f}")
    print(f"   Accuracy:    {acc_estr:.2%}")

    # Calibrar modelo estructural
    params_no_es = {k: v for k, v in PARAMS_XGB_VB.items() if k != 'early_stopping_rounds'}
    xgb_estr_for_cal = XGBClassifier(**params_no_es)
    xgb_estr_for_cal.fit(X_estr_train, y_train, sample_weight=sample_weights_estr)
    tscv_estr = TimeSeriesSplit(n_splits=3)
    cal_estr = CalibratedClassifierCV(estimator=xgb_estr_for_cal, method='sigmoid', cv=tscv_estr)
    cal_estr.fit(X_estr_train, y_train)
    probs_cal_estr = cal_estr.predict_proba(X_estr_test)
    bs_cal_estr = _brier_multiclase(y_test, probs_cal_estr)

    if bs_cal_estr < bs_estr:
        modelo_estr_final = cal_estr
        print(f"   Calibracion mejora Brier ({bs_estr:.4f} -> {bs_cal_estr:.4f})")
    else:
        modelo_estr_final = xgb_estr
        print(f"   Calibracion no mejora, guardando original")

    joblib.dump(modelo_estr_final, ARCHIVO_MODELO_VB)
    joblib.dump(features_estr, ARCHIVO_FEATURES_VB)
    print(f"\n   Modelo estructural: {ARCHIVO_MODELO_VB}")
    print(f"   Features:           {ARCHIVO_FEATURES_VB}")

    # Comparación final
    print(f"\n   COMPARACION FINAL:")
    print(f"   {'Modelo':<35} {'Log Loss':>9} {'Brier':>7} {'F1':>7} {'Acc':>7}")
    print(f"   {'-'*65}")
    print(f"   {'Principal (' + str(len(features)) + ' feat)':<35} {ll_final:>9.4f} {bs_final:>7.4f} {f1_final:>7.4f} {acc_final:>6.2%}")
    print(f"   {'Estructural (' + str(len(features_estr)) + ' feat)':<35} {ll_estr:>9.4f} {bs_estr:>7.4f} {f1_estr:>7.4f} {acc_estr:>6.2%}")
    diff_ll = ll_estr - ll_final
    print(f"\n   Delta Log Loss: {diff_ll:+.4f}")

    return modelo_a_guardar, features


if __name__ == "__main__":
    modelo_final, features = main()