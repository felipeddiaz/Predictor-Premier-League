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

Metricas de optimizacion:
    Primaria:    Log Loss (calidad probabilistica)
    Secundaria:  Brier Score (error cuadratico de probs)
    Terciaria:   ROI simulado (value betting)
    Referencia:  F1-weighted, Accuracy

Pipeline:
    datos/procesados/premier_league_con_features.csv
    → FEATURES_ESTRUCTURALES (base + H2H + xG + tabla + forma + arbitro + descanso)
    → XGBoost maneja NaN nativo; RF usa fillna(0)
    → solo partidos con H2H disponible
    → 80/20 split temporal
    → RF basico + RF balanceado + RF Optuna + XGBoost (early_stopping)
    → CalibratedClassifierCV (sigmoid, TimeSeriesSplit)
    → Evaluacion ROI por umbral de edge
    → modelos/modelo_value_betting.pkl

Para afinar PARAMS_OPTIMOS_VB: ejecutar visualizar_busqueda.py
con MODO_SIN_CUOTAS = True y copiar los valores al config.py.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, log_loss, brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV
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
    ARCHIVO_MODELO_VB,
    ARCHIVO_FEATURES_VB,
    PARAMS_OPTIMOS_VB,
    PARAMS_XGB_VB,
    PESOS_OPTIMOS,
    PESOS_XGB,
    FEATURES_ESTRUCTURALES,
    FEATURES_BASE,
    FEATURES_H2H,
    FEATURES_H2H_DERIVADAS,
    FEATURES_XG,
    FEATURES_XG_GLOBAL,
    FEATURES_MULTI_ESCALA,
    FEATURES_TABLA,
    FEATURES_FORMA_MOMENTUM,
    FEATURES_REFEREE,
    FEATURES_DESCANSO,
    RANDOM_SEED,
    UMBRAL_EDGE_MINIMO,
)
from utils import agregar_xg_rolling, agregar_features_tabla

warnings.filterwarnings('ignore')

RUTA_DATOS = ARCHIVO_FEATURES
os.makedirs(RUTA_MODELOS, exist_ok=True)

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


def _roi_simulado(y_true, probs, df_cuotas=None, edge_minimo=None):
    """
    ROI simulado con apuestas planas sobre value bets.

    Para cada partido donde max(prob_modelo) - prob_mercado > edge_minimo,
    apuesta 1 unidad al resultado con mayor edge. Calcula el ROI total.
    """
    if edge_minimo is None:
        edge_minimo = UMBRAL_EDGE_MINIMO

    if df_cuotas is None:
        return None

    cuota_cols = ['B365H', 'B365D', 'B365A']
    if not all(c in df_cuotas.columns for c in cuota_cols):
        return None

    cuotas = df_cuotas[cuota_cols].values
    apostado = 0.0
    ganancia = 0.0

    for i in range(len(y_true)):
        prob_mercado = 1.0 / cuotas[i]
        if np.any(np.isnan(prob_mercado)) or np.any(cuotas[i] <= 1.0):
            continue

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
# CARGA Y PREPARACIÓN
# ============================================================================

def cargar_datos():
    """Carga datos y prepara features SIN cuotas."""
    print("=" * 70)
    print("FASE 1: CARGANDO DATOS - MODELO SIN CUOTAS")
    print("=" * 70)

    if not os.path.exists(RUTA_DATOS):
        print(f"ERROR: No se encontró '{RUTA_DATOS}'")
        print("   Ejecuta primero: python 01_preparar_datos.py")
        return None, None, None, None, None

    df = pd.read_csv(RUTA_DATOS)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    print(f"   Cargados: {len(df)} partidos")

    # Features calculadas en memoria (funciones canónicas de utils.py)
    df = agregar_xg_rolling(df)
    df = agregar_features_tabla(df)

    # Solo partidos con H2H disponible
    if 'H2H_Available' in df.columns:
        antes = len(df)
        df = df[df['H2H_Available'] == 1].copy()
        print(f"   Filtro H2H: {antes} -> {len(df)} partidos con historial disponible")

    df = df.reset_index(drop=True)

    # FEATURES_ESTRUCTURALES: lista canónica sin cuotas (definida en config.py)
    # XGBoost maneja regularización implícita — no se necesita feature selection top-N
    features = [f for f in FEATURES_ESTRUCTURALES if f in df.columns]

    print(f"\n   Features totales: {len(features)} (SIN cuotas)")
    print(f"   - Base:            {len([f for f in FEATURES_BASE if f in features])}")
    print(f"   - H2H:             {len([f for f in FEATURES_H2H if f in features])}")
    print(f"   - H2H derivadas:   {len([f for f in FEATURES_H2H_DERIVADAS if f in features])}")
    print(f"   - xG rolling:      {len([f for f in FEATURES_XG if f in features])}")
    print(f"   - xG Global:       {len([f for f in FEATURES_XG_GLOBAL if f in features])}")
    print(f"   - Multi-escala:    {len([f for f in FEATURES_MULTI_ESCALA if f in features])}")
    print(f"   - Tabla:           {len([f for f in FEATURES_TABLA if f in features])}")
    print(f"   - Forma/Momentum:  {len([f for f in FEATURES_FORMA_MOMENTUM if f in features])}")
    print(f"   - Referee:         {len([f for f in FEATURES_REFEREE if f in features])}")
    print(f"   - Descanso/Fatiga: {len([f for f in FEATURES_DESCANSO if f in features])}")

    # X con NaN para XGBoost (maneja NaN nativamente)
    # X_filled para Random Forest (no soporta NaN)
    X = df[features]
    X_filled = X.fillna(0)
    y = df['FTR_numeric']

    print(f"\n   Distribución de resultados:")
    for clase in [0, 1, 2]:
        nombre = ['Local', 'Empate', 'Visitante'][clase]
        count = (y == clase).sum()
        pct = count / len(y) * 100
        print(f"   {nombre}: {count} ({pct:.1f}%)")

    return X, X_filled, y, features, df


# ============================================================================
# ENTRENAMIENTO DE MODELOS
# ============================================================================

def entrenar_modelos(X_train, y_train, X_test, y_test,
                     X_train_filled=None, X_test_filled=None,
                     df_cuotas_test=None):
    """Entrena RF (con fillna) y XGBoost (con NaN nativo + early stopping).

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
        print(f"   Log Loss: {r['log_loss']:.4f} | Brier: {r['brier_score']:.4f} "
              f"| F1: {r['f1_score']:.4f} | Acc: {r['accuracy']:.2%}{roi_str}")

    # -------------------------------------------------------------------------
    # MODELO 1: Random Forest Básico (baseline)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MODELO 1: RANDOM FOREST BASICO (Baseline)")
    print("=" * 70)

    rf_basico = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf_basico.fit(X_train_filled, y_train)
    resultados['RF_Basico'] = _evaluar_modelo(
        rf_basico, X_test_filled, y_test, 'Random Forest Básico', df_cuotas_test)
    _print_metricas(resultados['RF_Basico'])

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
    rf_balanceado.fit(X_train_filled, y_train)
    resultados['RF_Balanceado'] = _evaluar_modelo(
        rf_balanceado, X_test_filled, y_test, 'Random Forest Balanceado', df_cuotas_test)
    _print_metricas(resultados['RF_Balanceado'])

    # -------------------------------------------------------------------------
    # MODELO 3: Random Forest con PESOS OPTUNA
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MODELO 3: RANDOM FOREST CON PESOS OPTUNA")
    print("=" * 70)

    rf_optuna = RandomForestClassifier(**PARAMS_OPTIMOS_VB)
    rf_optuna.fit(X_train_filled, y_train)
    resultados['RF_Optuna'] = _evaluar_modelo(
        rf_optuna, X_test_filled, y_test, 'Random Forest Optuna (sin cuotas)', df_cuotas_test)
    _print_metricas(resultados['RF_Optuna'])

    # -------------------------------------------------------------------------
    # MODELO 4: XGBoost (NaN nativo + early stopping)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MODELO 4: XGBOOST (sin cuotas)")
    print("=" * 70)

    sample_weights_train = compute_sample_weight(
        class_weight=PESOS_XGB, y=y_train
    )

    xgb_model = XGBClassifier(**PARAMS_XGB_VB)

    # Early stopping: para en cuanto el val loss no mejore en 50 rondas
    # Evita sobreajuste sin necesidad de feature selection manual
    xgb_model.fit(
        X_train, y_train,
        sample_weight=sample_weights_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    if hasattr(xgb_model, 'best_iteration'):
        print(f"   Early stopping: mejor iteración = {xgb_model.best_iteration}")

    resultados['XGBoost'] = _evaluar_modelo(
        xgb_model, X_test, y_test, 'XGBoost (sin cuotas)', df_cuotas_test)
    _print_metricas(resultados['XGBoost'])

    return resultados


# ============================================================================
# SELECCIÓN DEL MEJOR MODELO
# ============================================================================

def seleccionar_mejor_modelo(resultados, y_test):
    """Compara modelos y selecciona el mejor según Log Loss (calidad probabilística)."""

    print("\n" + "=" * 70)
    print("FASE 2: COMPARACION DE MODELOS")
    print("=" * 70)

    header = f"{'Modelo':<35} {'Log Loss':>9} {'Brier':>7} {'F1':>7} {'Acc':>7}"
    roi_disponible = any(r['roi'] is not None for r in resultados.values())
    if roi_disponible:
        header += f" {'ROI':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    for key, datos in resultados.items():
        linea = (f"{datos['nombre']:<35} {datos['log_loss']:>9.4f} "
                 f"{datos['brier_score']:>7.4f} {datos['f1_score']:>7.4f} "
                 f"{datos['accuracy']:>6.2%}")
        if roi_disponible:
            roi_str = f"{datos['roi']:>+7.2%}" if datos['roi'] is not None else f"{'N/A':>8}"
            linea += f" {roi_str}"
        print(linea)

    # Elegir mejor por Log Loss (menor = mejor calidad probabilística)
    mejor_key = min(resultados.items(), key=lambda x: x[1]['log_loss'])[0]
    mejor = resultados[mejor_key]

    print(f"\n{'=' * 60}")
    print(f"GANADOR: {mejor['nombre']}  (Log Loss: {mejor['log_loss']:.4f})")
    print(f"   Brier Score: {mejor['brier_score']:.4f}")
    print(f"   F1-Score:    {mejor['f1_score']:.4f} (referencia)")
    print(f"   Accuracy:    {mejor['accuracy']:.2%} (referencia)")
    if mejor['roi'] is not None:
        print(f"   ROI sim.:    {mejor['roi']:+.2%}")
    print(f"{'=' * 60}")

    print(f"\n   REPORTE DETALLADO:")
    print("-" * 65)
    print(classification_report(y_test, mejor['predicciones'],
                                target_names=['Local', 'Empate', 'Visitante']))

    return mejor_key, mejor


# ============================================================================
# CALIBRACIÓN DE PROBABILIDADES
# ============================================================================

def calibrar_modelo(modelo, X_train, y_train, X_test, y_test):
    """
    Calibra el modelo para obtener probabilidades más realistas.

    Usa TimeSeriesSplit (no KFold estándar) para respetar la causalidad temporal.
    Evalúa calibración con Brier Score y Log Loss — NO con F1.

    Retorna (modelo_a_guardar, probs, es_calibrado):
      - Si la calibración mejora Brier Score -> retorna modelo calibrado
      - Si no mejora -> retorna modelo original (probabilidades ya eran buenas)
    """
    print("\n" + "=" * 70)
    print("CALIBRACION DE PROBABILIDADES (Platt Scaling)")
    print("=" * 70)

    # --- Métricas del modelo SIN calibrar ---
    probs_original = modelo.predict_proba(X_test)
    logloss_original = log_loss(y_test, probs_original)
    brier_original = _brier_multiclase(y_test, probs_original)

    print(f"\n   Modelo ORIGINAL (sin calibrar):")
    print(f"   Log Loss:    {logloss_original:.4f}")
    print(f"   Brier Score: {brier_original:.4f}")

    # --- Calibrar con TimeSeriesSplit (respeta orden temporal) ---
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
    brier_calibrado = _brier_multiclase(y_test, probs_calibradas)

    print(f"\n   Modelo CALIBRADO (Platt Scaling + TimeSeriesSplit):")
    print(f"   Log Loss:    {logloss_calibrado:.4f}")
    print(f"   Brier Score: {brier_calibrado:.4f}")

    # --- Comparar ---
    mejora_logloss = logloss_original - logloss_calibrado
    mejora_brier = brier_original - brier_calibrado

    print(f"\n   COMPARACION:")
    print(f"   {'Métrica':<15} {'Original':<12} {'Calibrado':<12} {'Cambio':<12}")
    print(f"   {'-' * 50}")
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
        print(f"\n   Calibración MEJORA probabilidades (Brier -{mejora_brier:.4f})")
        print(f"   -> Guardando modelo CALIBRADO")
        return modelo_calibrado, probs_calibradas, True
    else:
        print(f"\n   Calibración NO mejora probabilidades (Brier +{abs(mejora_brier):.4f})")
        print(f"   -> Guardando modelo ORIGINAL")
        return modelo, probs_original, False


# ============================================================================
# EVALUACIÓN CON VALUE BETTING (diferencial del 03 vs el 02)
# ============================================================================

def evaluar_value_betting(modelo, X_test, y_test, df_test):
    """
    Evalúa el modelo usando cuotas como filtro EXTERNO post-predicción.
    Las cuotas NO entraron en el entrenamiento: si hay edge, es estructural.
    """

    print("\n" + "=" * 70)
    print("EVALUACION CON VALUE BETTING (cuotas como filtro externo)")
    print("=" * 70)

    if not all(col in df_test.columns for col in ['B365H', 'B365D', 'B365A']):
        print("   No hay cuotas disponibles para la evaluación de value betting")
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

    print(f"\n   ROI SIMULADO POR UMBRAL DE EDGE (apuesta fija de 1 unidad):")
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

    print("\n   INTERPRETACION:")
    print("   - Edge = modelo ve valor que el mercado NO vio (sin haberlo consultado)")
    print("   - Edge >5%  -> apuestas con buena probabilidad")
    print("   - Edge >8%  -> apuestas con alta confianza")
    print("   - ROI >0%   -> estrategia rentable a largo plazo")
    print("\n   RECOMENDACION: usar edge >5% como filtro mínimo para apostar")


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
    modelo_base = modelo.estimator if hasattr(modelo, 'estimator') else modelo
    if hasattr(modelo_base, 'feature_importances_'):
        importances = modelo_base.feature_importances_
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
    print(f"   Guardado: {archivo}")
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
    print(f"   Modelo: {ARCHIVO_MODELO_VB}")

    joblib.dump(features, ARCHIVO_FEATURES_VB)
    print(f"   Features: {ARCHIVO_FEATURES_VB}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Pipeline completo — modelo sin cuotas."""

    print("\n" + "=" * 70)
    print("   PREMIER LEAGUE - MODELO SIN CUOTAS")
    print("   (Value Betting puro — cuotas solo como filtro externo)")
    print("=" * 70 + "\n")

    # Cargar datos
    resultado = cargar_datos()
    if resultado[0] is None:
        return None, None
    X, X_filled, y, features, df = resultado

    # Split temporal 80/20
    print("\n   División de datos (80/20 temporal)")
    split_idx = int(len(X) * 0.8)
    X_train      = X.iloc[:split_idx]
    X_test       = X.iloc[split_idx:]
    X_train_filled = X_filled.iloc[:split_idx]
    X_test_filled  = X_filled.iloc[split_idx:]
    y_train      = y.iloc[:split_idx]
    y_test       = y.iloc[split_idx:]
    df_test      = df.iloc[split_idx:].reset_index(drop=True)

    print(f"   Entrenamiento: {len(X_train)} partidos")
    print(f"   Prueba:        {len(X_test)} partidos")

    # Entrenar los 4 modelos (RF con filled, XGBoost con NaN nativo)
    resultados = entrenar_modelos(
        X_train, y_train, X_test, y_test,
        X_train_filled, X_test_filled,
        df_cuotas_test=df_test
    )

    # Seleccionar el mejor por Log Loss
    mejor_key, mejor = seleccionar_mejor_modelo(resultados, y_test)

    # Calibrar probabilidades del ganador (TimeSeriesSplit, no KFold)
    es_xgb = 'XGBoost' in mejor['nombre']
    _X_tr_cal = X_train if es_xgb else X_train_filled
    _X_te_cal = X_test if es_xgb else X_test_filled

    modelo_a_guardar, probs_finales, fue_calibrado = calibrar_modelo(
        mejor['modelo'], _X_tr_cal, y_train, _X_te_cal, y_test
    )

    tag_cal = "(Calibrado)" if fue_calibrado else "(Sin Calibrar)"
    nombre_final = f"{mejor['nombre']} {tag_cal}"

    # Evaluación value betting (diferencial exclusivo del 03)
    evaluar_value_betting(modelo_a_guardar, _X_te_cal, y_test, df_test)

    # Visualizaciones (usa modelo base para feature_importances_)
    visualizar_resultados(y_test, mejor['predicciones'], nombre_final, features, mejor['modelo'])

    # Guardar modelo
    guardar_modelo(modelo_a_guardar, features)

    # Métricas finales
    probs_final = modelo_a_guardar.predict_proba(_X_te_cal)
    pred_guardado = modelo_a_guardar.predict(_X_te_cal)
    ll_final = log_loss(y_test, probs_final)
    bs_final = _brier_multiclase(y_test, probs_final)
    f1_final = f1_score(y_test, pred_guardado, average='weighted')
    acc_final = accuracy_score(y_test, pred_guardado)
    roi_final = _roi_simulado(y_test, probs_final, df_test)

    print("\n" + "=" * 70)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    print(f"\n   Modelo guardado: {nombre_final}")
    print(f"\n   Métricas primarias (calidad probabilística):")
    print(f"   Log Loss:    {ll_final:.4f}")
    print(f"   Brier Score: {bs_final:.4f}")
    if roi_final is not None:
        print(f"   ROI sim.:    {roi_final:+.2%}")
    print(f"\n   Métricas de referencia (clasificación):")
    print(f"   Accuracy:    {acc_final:.2%}")
    print(f"   F1-Score:    {f1_final:.4f}")

    print(f"\n   Archivos en {RUTA_MODELOS}")
    print(f"\n   Para afinar hiperparámetros:")
    print(f"   1. Abre visualizar_busqueda.py")
    print(f"   2. Cambia MODO_SIN_CUOTAS = True")
    print(f"   3. Ejecuta el script y copia los valores a PARAMS_OPTIMOS_VB en config.py\n")

    return modelo_a_guardar, features


if __name__ == "__main__":
    modelo_final, features = main()
