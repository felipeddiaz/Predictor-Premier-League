# -*- coding: utf-8 -*-
"""
Premier League Match Predictor
04 - MODELO DEDICADO DE EMPATES (binario: Draw vs No-Draw)

El modelo principal (02) casi nunca predice empates porque la clase es
minoritaria (~23%) y las métricas de optimización (Log Loss, F1-weighted)
penalizan poco el error en empates.

Este script entrena un clasificador binario separado:
  - Target: 1 = Empate, 0 = No-Empate
  - Features: las mismas del modelo principal + features derivadas
    específicas para detectar paridad entre equipos
  - Se integra como segunda capa en el Predictor: si P(draw) del modelo
    de empates > umbral, se boostea la probabilidad de empate del modelo
    principal.

Features derivadas para empates:
  - xG_Abs_Diff:        |xG_local - xG_visitante| (menor = más parejo)
  - Goals_Abs_Diff:     |AvgGoals_local - AvgGoals_visitante|
  - Position_Abs_Diff:  |Position_local - Position_visitante|
  - Form_Draw_Sum:      Form_D_local + Form_D_visitante (tendencia a empatar)
  - H2H_Draw_Rate:      H2H_Draws / H2H_Matches (historial de empates)
  - Cuota_Draw_Implied:  1/B365D normalizada (señal del mercado para empate)

Pipeline:
    datos/procesados/archive/premier_league_RESTAURADO.csv
    → features canónicas + features derivadas de paridad
    → 80/20 split temporal
    → XGBoost binario (Draw vs No-Draw) con scale_pos_weight
    → Calibración Platt Scaling
    → Búsqueda de umbral óptimo (F1 de empates)
    → modelos/modelo_empates.pkl
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, log_loss, brier_score_loss, precision_recall_curve,
    roc_auc_score,
)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

from config import (
    ARCHIVO_FEATURES,
    RUTA_MODELOS,
    ARCHIVO_MODELO_EMPATES,
    ARCHIVO_FEATURES_EMPATES,
    ARCHIVO_METADATA_EMPATES,
    PARAMS_XGB_EMPATES,
    FEATURES_EMPATES,
    FEATURES_BASE,
    FEATURES_CUOTAS,
    FEATURES_CUOTAS_DERIVADAS,
    FEATURES_XG,
    FEATURES_XG_GLOBAL,
    FEATURES_H2H,
    FEATURES_H2H_DERIVADAS,
    FEATURES_TABLA,
    FEATURES_ASIAN_HANDICAP,
    FEATURES_DESCANSO,
    RANDOM_SEED,
)
from utils import agregar_xg_rolling, agregar_features_tabla

warnings.filterwarnings('ignore')
os.makedirs(RUTA_MODELOS, exist_ok=True)


# ============================================================================
# FEATURES DERIVADAS PARA EMPATES
# ============================================================================

FEATURES_DRAW_DERIVADAS = [
    'xG_Abs_Diff',
    'Goals_Abs_Diff',
    'Position_Abs_Diff',
    'Form_Draw_Sum',
    'H2H_Draw_Rate',
    'Cuota_Draw_Implied',
    'xG_Low_Total',
    'Form_Similarity',
]


def agregar_features_empates(df):
    """Agrega features derivadas específicas para detectar empates."""

    # Diferencia absoluta de xG (menor = equipos más parejos)
    if 'HT_xG_Avg' in df.columns and 'AT_xG_Avg' in df.columns:
        df['xG_Abs_Diff'] = (df['HT_xG_Avg'] - df['AT_xG_Avg']).abs()
    else:
        df['xG_Abs_Diff'] = 0.0

    # Diferencia absoluta de goles promedio
    if 'HT_AvgGoals' in df.columns and 'AT_AvgGoals' in df.columns:
        df['Goals_Abs_Diff'] = (df['HT_AvgGoals'] - df['AT_AvgGoals']).abs()
    else:
        df['Goals_Abs_Diff'] = 0.0

    # Diferencia absoluta de posición en la tabla
    if 'HT_Position' in df.columns and 'AT_Position' in df.columns:
        df['Position_Abs_Diff'] = (df['HT_Position'] - df['AT_Position']).abs()
    elif 'Position_Diff' in df.columns:
        df['Position_Abs_Diff'] = df['Position_Diff'].abs()
    else:
        df['Position_Abs_Diff'] = 0.0

    # Suma de empates recientes (Form_D) de ambos equipos
    if 'HT_Form_D' in df.columns and 'AT_Form_D' in df.columns:
        df['Form_Draw_Sum'] = df['HT_Form_D'] + df['AT_Form_D']
    else:
        df['Form_Draw_Sum'] = 0.0

    # Tasa de empates en H2H
    if 'H2H_Draws' in df.columns and 'H2H_Matches' in df.columns:
        df['H2H_Draw_Rate'] = df['H2H_Draws'] / df['H2H_Matches'].clip(lower=1)
    elif 'H2H_Home_Win_Rate' in df.columns and 'H2H_Win_Advantage' in df.columns:
        # Aproximar: Draw_Rate ≈ 1 - Home_Win_Rate - Away_Win_Rate
        # Away_Win_Rate ≈ Home_Win_Rate - Win_Advantage
        away_wr = (df['H2H_Home_Win_Rate'] - df['H2H_Win_Advantage']).clip(lower=0)
        df['H2H_Draw_Rate'] = (1 - df['H2H_Home_Win_Rate'] - away_wr).clip(lower=0, upper=1)
    else:
        df['H2H_Draw_Rate'] = 0.25

    # Probabilidad implícita del empate en el mercado (normalizada)
    if 'B365D' in df.columns and 'B365H' in df.columns and 'B365A' in df.columns:
        inv_h = 1.0 / df['B365H']
        inv_d = 1.0 / df['B365D']
        inv_a = 1.0 / df['B365A']
        total = inv_h + inv_d + inv_a
        df['Cuota_Draw_Implied'] = inv_d / total
    else:
        df['Cuota_Draw_Implied'] = 0.25

    # xG total bajo (partidos con pocos goles esperados tienden a empatar)
    if 'xG_Total' in df.columns:
        df['xG_Low_Total'] = (df['xG_Total'] < df['xG_Total'].median()).astype(float)
    else:
        df['xG_Low_Total'] = 0.0

    # Similitud de forma (cuanto más parecida la forma, más probable el empate)
    if all(c in df.columns for c in ['HT_Form_W', 'AT_Form_W', 'HT_Form_L', 'AT_Form_L']):
        df['Form_Similarity'] = 1.0 - (
            (df['HT_Form_W'] - df['AT_Form_W']).abs() +
            (df['HT_Form_L'] - df['AT_Form_L']).abs()
        ) / 10.0  # Normalizado: 0-1, mayor = más similares
        df['Form_Similarity'] = df['Form_Similarity'].clip(lower=0, upper=1)
    else:
        df['Form_Similarity'] = 0.5

    return df


# ============================================================================
# CARGA Y PREPARACIÓN
# ============================================================================

def cargar_datos():
    """Carga datos y prepara features para el modelo binario de empates."""
    print("=" * 70)
    print("FASE 1: CARGANDO DATOS - MODELO DE EMPATES")
    print("=" * 70)

    ruta = ARCHIVO_FEATURES
    if not os.path.exists(ruta):
        print(f"ERROR: No se encontró '{ruta}'")
        print("   Ejecuta primero: python 01_preparar_datos.py")
        return None, None, None, None

    df = pd.read_csv(ruta)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    print(f"   Cargados: {len(df)} partidos")

    # Agregar features calculadas
    df = agregar_xg_rolling(df)
    df = agregar_features_tabla(df)

    # Agregar features derivadas específicas para empates
    df = agregar_features_empates(df)

    # Solo partidos con H2H disponible
    if 'H2H_Available' in df.columns:
        antes = len(df)
        df = df[df['H2H_Available'] == 1].copy()
        print(f"   Filtro H2H: {antes} -> {len(df)} partidos")

    df = df.reset_index(drop=True)

    # Features: canónicas + derivadas de empate
    features_canonicas = [f for f in FEATURES_EMPATES if f in df.columns]
    features_draw = [f for f in FEATURES_DRAW_DERIVADAS if f in df.columns]
    features = features_canonicas + features_draw

    print(f"\n   Features totales: {len(features)}")
    print(f"   - Canónicas:     {len(features_canonicas)}")
    print(f"   - Draw-específ.: {len(features_draw)}")

    # Target binario: 1 = Empate, 0 = No-Empate
    if 'FTR_numeric' in df.columns:
        y = (df['FTR_numeric'] == 1).astype(int)
    else:
        y = (df['FTR'] == 'D').astype(int)

    X = df[features]

    print(f"\n   Distribución:")
    print(f"   No-Empate (0): {(y == 0).sum()} ({(y == 0).mean():.1%})")
    print(f"   Empate    (1): {(y == 1).sum()} ({(y == 1).mean():.1%})")

    return X, y, features, df


# ============================================================================
# ENTRENAMIENTO
# ============================================================================

def entrenar_modelo(X_train, y_train, X_test, y_test):
    """Entrena XGBoost binario para Draw vs No-Draw."""

    print("\n" + "=" * 70)
    print("FASE 2: ENTRENAMIENTO - XGBOOST BINARIO (Draw vs No-Draw)")
    print("=" * 70)

    # Calcular scale_pos_weight dinámicamente
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw = n_neg / n_pos
    print(f"\n   Ratio No-Empate/Empate: {spw:.2f}")
    print(f"   scale_pos_weight: {spw:.2f}")

    params = PARAMS_XGB_EMPATES.copy()
    params['scale_pos_weight'] = spw

    modelo = XGBClassifier(**params)
    modelo.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    if hasattr(modelo, 'best_iteration'):
        print(f"   Early stopping: mejor iteración = {modelo.best_iteration}")

    # Métricas
    proba = modelo.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    ll = log_loss(y_test, proba)
    bs = brier_score_loss(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, pred)
    acc = accuracy_score(y_test, pred)

    print(f"\n   Métricas en test set:")
    print(f"   Log Loss:  {ll:.4f}")
    print(f"   Brier:     {bs:.4f}")
    print(f"   AUC-ROC:   {auc:.4f}")
    print(f"   F1 (Draw): {f1:.4f}")
    print(f"   Accuracy:  {acc:.2%}")

    print(f"\n   Reporte detallado:")
    print(classification_report(y_test, pred, target_names=['No-Empate', 'Empate']))

    return modelo


# ============================================================================
# CALIBRACIÓN
# ============================================================================

def calibrar_modelo(modelo, X_train, y_train, X_test, y_test):
    """Calibra probabilidades con Platt Scaling + TimeSeriesSplit."""

    print("\n" + "=" * 70)
    print("FASE 3: CALIBRACIÓN DE PROBABILIDADES")
    print("=" * 70)

    proba_orig = modelo.predict_proba(X_test)[:, 1]
    bs_orig = brier_score_loss(y_test, proba_orig)
    ll_orig = log_loss(y_test, proba_orig)

    print(f"\n   ORIGINAL:  Brier={bs_orig:.4f}  Log Loss={ll_orig:.4f}")

    tscv = TimeSeriesSplit(n_splits=3)
    modelo_cal = CalibratedClassifierCV(
        estimator=modelo,
        method='isotonic',  # Isotonic para patrones no lineales en empates
        cv=tscv,
    )
    modelo_cal.fit(X_train, y_train)

    proba_cal = modelo_cal.predict_proba(X_test)[:, 1]
    bs_cal = brier_score_loss(y_test, proba_cal)
    ll_cal = log_loss(y_test, proba_cal)

    print(f"   CALIBRADO: Brier={bs_cal:.4f}  Log Loss={ll_cal:.4f}")
    print(f"   Cambio:    Brier={bs_orig - bs_cal:+.4f}  Log Loss={ll_orig - ll_cal:+.4f}")

    if bs_cal < bs_orig:
        print(f"\n   -> Usando modelo CALIBRADO (isotonic)")
        return modelo_cal, proba_cal, True
    else:
        print(f"\n   -> Calibración no mejora, usando modelo ORIGINAL")
        return modelo, proba_orig, False


# ============================================================================
# BÚSQUEDA DE UMBRAL ÓPTIMO
# ============================================================================

def buscar_umbral_optimo(y_test, proba):
    """Busca el umbral de decisión que maximiza F1 de empates."""

    print("\n" + "=" * 70)
    print("FASE 4: BÚSQUEDA DE UMBRAL ÓPTIMO")
    print("=" * 70)

    precision, recall, thresholds = precision_recall_curve(y_test, proba)

    # F1 para cada umbral
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    # También evaluar umbrales fijos para entender el trade-off
    print(f"\n   {'Umbral':<10} {'Precision':<12} {'Recall':<10} {'F1':<10} {'N_pred':<10}")
    print("   " + "-" * 52)

    umbrales_fijos = [0.20, 0.22, 0.25, 0.28, 0.30, 0.33, 0.35, 0.40, 0.45, 0.50]
    mejor_f1 = 0
    mejor_umbral = 0.25

    for umbral in umbrales_fijos:
        pred = (proba >= umbral).astype(int)
        if pred.sum() == 0:
            continue
        p = precision_score_safe(y_test, pred)
        r = recall_score_safe(y_test, pred)
        f1 = f1_score(y_test, pred)
        n = pred.sum()
        marker = ""
        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_umbral = umbral
            marker = " <-- mejor"
        print(f"   {umbral:<10.2f} {p:<12.4f} {r:<10.4f} {f1:<10.4f} {n:<10}{marker}")

    # Buscar el óptimo exacto en precision_recall_curve
    if len(thresholds) > 0:
        best_idx = np.argmax(f1_scores[:-1])  # último valor es edge case
        umbral_pr = thresholds[best_idx]
        f1_pr = f1_scores[best_idx]
        if f1_pr > mejor_f1:
            mejor_umbral = umbral_pr
            mejor_f1 = f1_pr

    print(f"\n   Umbral óptimo: {mejor_umbral:.4f} (F1={mejor_f1:.4f})")

    return mejor_umbral


def precision_score_safe(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score_safe(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


# ============================================================================
# EVALUACIÓN DE IMPACTO EN MODELO PRINCIPAL
# ============================================================================

def evaluar_impacto(y_test_3class, proba_draw, df_test, umbral):
    """
    Simula cómo el modelo de empates mejoraría las predicciones del modelo
    principal de 3 clases.
    """
    print("\n" + "=" * 70)
    print("FASE 5: SIMULACIÓN DE IMPACTO EN MODELO PRINCIPAL")
    print("=" * 70)

    # El modelo principal casi nunca predice empate.
    # Contar cuántos empates reales hay en el test set
    if 'FTR_numeric' in df_test.columns:
        y_3class = df_test['FTR_numeric'].values
    elif 'FTR' in df_test.columns:
        label_map = {'H': 0, 'D': 1, 'A': 2}
        y_3class = df_test['FTR'].map(label_map).values
    else:
        print("   No se puede evaluar impacto sin FTR")
        return

    n_empates_reales = (y_3class == 1).sum()
    n_draw_detectados = (proba_draw >= umbral).sum()
    n_draw_correctos = ((proba_draw >= umbral) & (y_3class == 1)).sum()

    print(f"\n   Empates reales en test:        {n_empates_reales}")
    print(f"   Empates detectados (umbral={umbral:.2f}): {n_draw_detectados}")
    print(f"   Empates correctos:             {n_draw_correctos}")

    if n_draw_detectados > 0:
        precision = n_draw_correctos / n_draw_detectados
        print(f"   Precision de detección:        {precision:.1%}")

    if n_empates_reales > 0:
        recall = n_draw_correctos / n_empates_reales
        print(f"   Recall de empates:             {recall:.1%}")

    # Distribución de P(draw) en empates reales vs no-empates
    proba_en_empates = proba_draw[y_3class == 1]
    proba_en_no_empates = proba_draw[y_3class != 1]

    print(f"\n   P(draw) promedio en empates reales:    {proba_en_empates.mean():.3f}")
    print(f"   P(draw) promedio en no-empates:        {proba_en_no_empates.mean():.3f}")
    print(f"   Separación:                            {proba_en_empates.mean() - proba_en_no_empates.mean():.3f}")


# ============================================================================
# VISUALIZACIÓN
# ============================================================================

def visualizar_resultados(y_test, proba, umbral, features, modelo):
    """Genera visualizaciones del modelo de empates."""

    print("\n" + "=" * 70)
    print("GENERANDO VISUALIZACIONES")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Distribución de P(draw) por clase real
    ax = axes[0, 0]
    proba_empate = proba[y_test == 1]
    proba_no_empate = proba[y_test == 0]
    ax.hist(proba_no_empate, bins=30, alpha=0.6, label='No-Empate', color='steelblue')
    ax.hist(proba_empate, bins=30, alpha=0.6, label='Empate', color='coral')
    ax.axvline(x=umbral, color='red', linestyle='--', label=f'Umbral={umbral:.2f}')
    ax.set_xlabel('P(Draw)')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución de P(Draw) por resultado real')
    ax.legend()

    # 2. Matriz de confusión con umbral óptimo
    ax = axes[0, 1]
    pred = (proba >= umbral).astype(int)
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                xticklabels=['No-Empate', 'Empate'],
                yticklabels=['No-Empate', 'Empate'])
    ax.set_title(f'Matriz de Confusión (umbral={umbral:.2f})')
    ax.set_ylabel('Real')
    ax.set_xlabel('Predicción')

    # 3. Feature importance top 15
    ax = axes[1, 0]
    modelo_base = modelo.estimator if hasattr(modelo, 'estimator') else modelo
    if hasattr(modelo_base, 'feature_importances_'):
        importances = modelo_base.feature_importances_
        df_imp = pd.DataFrame({
            'Feature': features,
            'Importancia': importances,
        }).sort_values('Importancia', ascending=True).tail(15)
        df_imp.plot(x='Feature', y='Importancia', kind='barh', ax=ax,
                    legend=False, color='coral')
        ax.set_title('Top 15 Features (Modelo Empates)')
        ax.set_xlabel('Importancia')

    # 4. Calibración
    ax = axes[1, 1]
    from sklearn.calibration import calibration_curve
    fraction_positives, mean_predicted = calibration_curve(y_test, proba, n_bins=10)
    ax.plot(mean_predicted, fraction_positives, 's-', label='Modelo empates')
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfectamente calibrado')
    ax.set_xlabel('P(Draw) predicha')
    ax.set_ylabel('Fracción real de empates')
    ax.set_title('Curva de Calibración')
    ax.legend()

    plt.suptitle('Modelo Dedicado de Empates — Premier League', fontsize=14, fontweight='bold')
    plt.tight_layout()

    archivo = os.path.join(RUTA_MODELOS, 'modelo_empates.png')
    plt.savefig(archivo, dpi=150, bbox_inches='tight')
    print(f"   Guardado: {archivo}")
    plt.close()


# ============================================================================
# GUARDAR MODELO
# ============================================================================

def guardar_modelo(modelo, features, umbral, fue_calibrado):
    """Guarda el modelo de empates y metadatos."""

    print("\n" + "=" * 70)
    print("GUARDANDO MODELO DE EMPATES")
    print("=" * 70)

    joblib.dump(modelo, ARCHIVO_MODELO_EMPATES)
    print(f"   Modelo:   {ARCHIVO_MODELO_EMPATES}")

    joblib.dump(features, ARCHIVO_FEATURES_EMPATES)
    print(f"   Features: {ARCHIVO_FEATURES_EMPATES}")

    metadata = {
        'tipo': 'binario_empates',
        'umbral_optimo': umbral,
        'n_features': len(features),
        'calibrado': fue_calibrado,
        'features_draw_derivadas': FEATURES_DRAW_DERIVADAS,
    }
    joblib.dump(metadata, ARCHIVO_METADATA_EMPATES)
    print(f"   Metadata: {ARCHIVO_METADATA_EMPATES}")
    print(f"   Umbral óptimo: {umbral:.4f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Pipeline completo — modelo dedicado de empates."""

    print("\n" + "=" * 70)
    print("   PREMIER LEAGUE - MODELO DEDICADO DE EMPATES")
    print("   Clasificador binario: Draw vs No-Draw")
    print("=" * 70 + "\n")

    # Cargar datos
    resultado = cargar_datos()
    if resultado[0] is None:
        return None, None
    X, y, features, df = resultado

    # Split temporal 80/20
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    df_test = df.iloc[split_idx:].reset_index(drop=True)

    print(f"\n   Train: {len(X_train)} partidos ({y_train.mean():.1%} empates)")
    print(f"   Test:  {len(X_test)} partidos ({y_test.mean():.1%} empates)")

    # Entrenar
    modelo = entrenar_modelo(X_train, y_train, X_test, y_test)

    # Calibrar
    modelo_final, proba_final, fue_calibrado = calibrar_modelo(
        modelo, X_train, y_train, X_test, y_test
    )

    # Buscar umbral óptimo
    umbral = buscar_umbral_optimo(y_test, proba_final)

    # Evaluar impacto potencial en modelo principal
    evaluar_impacto(y_test, proba_final, df_test, umbral)

    # Visualizar
    visualizar_resultados(y_test.values, proba_final, umbral, features, modelo)

    # Guardar
    guardar_modelo(modelo_final, features, umbral, fue_calibrado)

    # Resumen final
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 70)

    bs = brier_score_loss(y_test, proba_final)
    ll = log_loss(y_test, proba_final)
    pred = (proba_final >= umbral).astype(int)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, proba_final)

    print(f"\n   Métricas finales:")
    print(f"   Brier Score: {bs:.4f}")
    print(f"   Log Loss:    {ll:.4f}")
    print(f"   AUC-ROC:     {auc:.4f}")
    print(f"   F1 (Draw):   {f1:.4f}")
    print(f"   Umbral:      {umbral:.4f}")

    tag_cal = "(Calibrado)" if fue_calibrado else "(Sin Calibrar)"
    print(f"\n   Modelo guardado: XGBoost Binario {tag_cal}")
    print(f"   Archivos en {RUTA_MODELOS}\n")

    return modelo_final, features


if __name__ == "__main__":
    modelo_final, features = main()
