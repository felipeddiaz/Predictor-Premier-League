# -*- coding: utf-8 -*-
"""
Premier League Match Predictor
03 - ENTRENAMIENTO CON VALUE BETTING

Modelo SIN cuotas como features + evaluacion de value betting usando cuotas
como filtro post-prediccion. Incluye soporte opcional para XGBoost y SMOTE.

Pipeline:
    datos/procesados/premier_league_con_features.csv
    → features sin cuotas (base + H2H + xG)
    → 80/20 split temporal (sin shuffle)
    → RF basico + RF balanceado + XGBoost con SMOTE (si disponible)
    → Evaluacion ROI por umbral de edge
    → modelos/modelo_value_betting.pkl
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
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
    FEATURES_BASE,
    FEATURES_H2H,
    FEATURES_H2H_DERIVADAS,
    FEATURES_XG,
    FEATURES_TABLA,
)
from utils import agregar_xg_rolling, agregar_features_tabla

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

ADVANCED_LIBS = XGBOOST_AVAILABLE and SMOTE_AVAILABLE

RUTA_DATOS = ARCHIVO_FEATURES
os.makedirs(RUTA_MODELOS, exist_ok=True)

# ============================================================================
# CARGA Y PREPARACIÓN
# ============================================================================

def cargar_datos_hibridos():
    """Carga datos procesados y prepara features SIN cuotas."""
    print("="*70)
    print("CARGANDO DATOS - MODELO SIN CUOTAS")
    print("="*70)
    
    df = pd.read_csv(RUTA_DATOS)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    print(f"✅ Cargados: {len(df)} partidos")
    
    # Agregar xG rolling (en memoria)
    df = agregar_xg_rolling(df)
    
    # Agregar features de tabla (en memoria) - NUEVO
    df = agregar_features_tabla(df)

    # Filtrar solo partidos con H2H disponible
    if 'H2H_Available' in df.columns:
        df = df[df['H2H_Available'] == 1].copy()
        print(f"✅ Filtrados partidos con H2H disponible: {len(df)} partidos")
    
    # Resetear índices para que coincidan con X, y
    df = df.reset_index(drop=True)

    # Features base (forma reciente)
    features_base = [
        'HT_AvgGoals', 'AT_AvgGoals',
        'HT_AvgShotsTarget', 'AT_AvgShotsTarget',
        'HT_Form_W', 'HT_Form_D', 'HT_Form_L',
        'AT_Form_W', 'AT_Form_D', 'AT_Form_L'
    ]

    # Features H2H
    features_h2h = [
        'H2H_Matches', 'H2H_Home_Wins', 'H2H_Draws', 'H2H_Away_Wins',
        'H2H_Home_Goals_Avg', 'H2H_Away_Goals_Avg', 
        'H2H_Home_Win_Rate', 'H2H_BTTS_Rate'
    ]

    # Features derivadas H2H
    features_h2h_derivadas = [
        'H2H_Goal_Diff',
        'H2H_Win_Advantage',
        'H2H_Total_Goals_Avg',
        'H2H_Home_Consistent'
    ]
    
    # Features xG rolling (calculadas en memoria)
    features_xg = [
        'HT_xG_Avg', 'AT_xG_Avg',
        'HT_xGA_Avg', 'AT_xGA_Avg',
        'xG_Diff', 'xG_Total'
    ]
    
    features_tabla = [
        'HT_Position', 'AT_Position',
        'Position_Diff', 'Position_Diff_Weighted',
        'HT_Points', 'AT_Points',
        'Season_Progress', 'Position_Reliability',
        'Match_Type',
        'HT_Pressure', 'AT_Pressure'
    ]

    # Combinar features (SIN CUOTAS)
    features = features_base.copy()
    h2h_disponibles = [col for col in features_h2h if col in df.columns]
    h2h_derivadas_disponibles = [col for col in features_h2h_derivadas if col in df.columns]
    xg_disponibles = [col for col in features_xg if col in df.columns]
    #tabla_disponibles = [col for col in features_tabla if col in df.columns]
    
    #features.extend(tabla_disponibles)
    features.extend(h2h_disponibles)
    features.extend(h2h_derivadas_disponibles)
    features.extend(xg_disponibles)
    features = [f for f in features if f in df.columns]

    print(f"\n Features totales: {len(features)} (SIN CUOTAS)")
    print(f"   • Base: {len([f for f in features_base if f in features])}")
    print(f"   • H2H: {len(h2h_disponibles)}")
    print(f"   • H2H derivadas: {len(h2h_derivadas_disponibles)}")
    print(f"   • xG rolling: {len(xg_disponibles)}")
    #print(f"   • Tabla: {len(tabla_disponibles)}")

    X_full = df[features].fillna(0)
    y = df['FTR_numeric']

    print(f"\n📊 Distribución:")
    for clase in [0, 1, 2]:
        nombre = ['Local', 'Empate', 'Visitante'][clase]
        count = (y == clase).sum()
        pct = count / len(y) * 100
        print(f"   {nombre}: {count} ({pct:.1f}%)")

    return X_full, y, features, df

# ============================================================================
# ENTRENAMIENTO
# ============================================================================

def entrenar_modelos_hibridos(X_train, y_train, X_test, y_test):
    """Entrena modelos optimizados."""
    
    modelos = {}
    
    # -------------------------------------------------------------------------
    # MODELO 1: Random Forest Básico
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("MODELO 1: RANDOM FOREST BÁSICO")
    print("="*70)
    
    rf_basico = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    rf_basico.fit(X_train, y_train)
    pred_basico = rf_basico.predict(X_test)
    
    acc_basico = accuracy_score(y_test, pred_basico)
    f1_basico = f1_score(y_test, pred_basico, average='weighted')
    
    modelos['RF_Basico'] = {
        'modelo': rf_basico,
        'predicciones': pred_basico,
        'accuracy': acc_basico,
        'f1_score': f1_basico,
        'nombre': 'Random Forest Básico'
    }
    
    print(f"✅ Test Accuracy: {acc_basico:.2%} | F1: {f1_basico:.4f}")
    
    # -------------------------------------------------------------------------
    # MODELO 2: Random Forest Balanceado
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("MODELO 2: RANDOM FOREST BALANCEADO")
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
    
    acc_balanceado = accuracy_score(y_test, pred_balanceado)
    f1_balanceado = f1_score(y_test, pred_balanceado, average='weighted')
    
    modelos['RF_Balanceado'] = {
        'modelo': rf_balanceado,
        'predicciones': pred_balanceado,
        'accuracy': acc_balanceado,
        'f1_score': f1_balanceado,
        'nombre': 'Random Forest Balanceado'
    }
    
    print(f"✅ Test Accuracy: {acc_balanceado:.2%} | F1: {f1_balanceado:.4f}")
    
    # -------------------------------------------------------------------------
    # MODELO 3: XGBoost con SMOTE
    # -------------------------------------------------------------------------
    if ADVANCED_LIBS:
        print("\n" + "="*70)
        print("MODELO 3: XGBOOST CON SMOTE")
        print("="*70)
        
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"🔧 SMOTE: {len(X_train)} → {len(X_train_balanced)}")
        
        xgb = XGBClassifier(
            n_estimators=100,
            random_state=42,
            objective='multi:softprob',
            eval_metric='mlogloss',
            n_jobs=-1
        )
        
        xgb.fit(X_train_balanced, y_train_balanced)
        pred_xgb = xgb.predict(X_test)
        
        acc_xgb = accuracy_score(y_test, pred_xgb)
        f1_xgb = f1_score(y_test, pred_xgb, average='weighted')
        
        modelos['XGBoost_SMOTE'] = {
            'modelo': xgb,
            'predicciones': pred_xgb,
            'accuracy': acc_xgb,
            'f1_score': f1_xgb,
            'nombre': 'XGBoost con SMOTE'
        }
        
        print(f"✅ Test Accuracy: {acc_xgb:.2%} | F1: {f1_xgb:.4f}")
    
    return modelos


def comparar_modelos(modelos, y_test):
    """Compara todos los modelos."""
    
    print("\n" + "="*70)
    print("COMPARACIÓN DE MODELOS")
    print("="*70)
    
    print(f"{'Modelo':<30} {'Test Acc':<12} {'Test F1':<12}")
    print("-" * 70)
    
    mejor_f1 = -1
    mejor_key = None
    
    for key, datos in modelos.items():
        test_acc = datos['accuracy']
        test_f1 = datos['f1_score']
        
        print(f"{datos['nombre']:<30} {test_acc:>10.2%}  {test_f1:>10.4f}")
        
        if test_f1 > mejor_f1:
            mejor_f1 = test_f1
            mejor_key = key
    
    print("\n" + "🏆" * 35)
    mejor = modelos[mejor_key]
    print(f"GANADOR: {mejor['nombre']}")
    print(f"   Test Accuracy: {mejor['accuracy']:.2%}")
    print(f"   Test F1-Score: {mejor['f1_score']:.4f}")
    print("🏆" * 35)
    
    # Reporte detallado
    print(f"\n📊 REPORTE DETALLADO:")
    print("-" * 70)
    target_names = ['Local', 'Empate', 'Visitante']
    print(classification_report(y_test, mejor['predicciones'], target_names=target_names))
    
    return mejor_key, mejor


def visualizar_resultados(y_test, predictions, nombre, features, modelo):
    """Genera visualizaciones."""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[0],
                xticklabels=['Local', 'Empate', 'Visitante'],
                yticklabels=['Local', 'Empate', 'Visitante'])
    axes[0].set_title(f'Matriz de Confusión\n{nombre}', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Real')
    axes[0].set_xlabel('Predicción')
    
    # Importancia de features
    if hasattr(modelo, 'feature_importances_'):
        importances = modelo.feature_importances_
        df_imp = pd.DataFrame({
            'Feature': features,
            'Importancia': importances
        }).sort_values('Importancia', ascending=True).tail(15)
        
        df_imp.plot(x='Feature', y='Importancia', kind='barh', ax=axes[1], 
                   legend=False, color='coral')
        axes[1].set_title('Top 15 Features', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Importancia')
    
    plt.tight_layout()
    
    archivo = os.path.join(RUTA_MODELOS, 'modelo_value_betting.png')
    plt.savefig(archivo, dpi=150, bbox_inches='tight')
    print(f"\n✅ Guardado: {archivo}")
    plt.show()


def analizar_features_h2h(modelo, features):
    """Ver qué features H2H son importantes."""
    try:
        if hasattr(modelo, 'feature_importances_'):
            importances = pd.DataFrame({
                'Feature': features,
                'Importancia': modelo.feature_importances_
            }).sort_values('Importancia', ascending=False)
            
            print("\n📊 IMPORTANCIA DE FEATURES H2H y xG:")
            print("-" * 50)
            
            # H2H features
            h2h_features = importances[importances['Feature'].str.contains('H2H')]
            if len(h2h_features) > 0:
                print("\n   H2H:")
                for idx, row in h2h_features.iterrows():
                    print(f"      {row['Feature']:<25} {row['Importancia']:.4f}")
            
            # xG features
            xg_features = importances[importances['Feature'].str.contains('xG')]
            if len(xg_features) > 0:
                print("\n   xG:")
                for idx, row in xg_features.iterrows():
                    print(f"      {row['Feature']:<25} {row['Importancia']:.4f}")
                
    except Exception as e:
        print(f"⚠️ Error al analizar features: {e}")


def evaluar_value_betting(modelo, X_test, y_test, df_test, features):
    """
    Evalúa el modelo usando VALUE BETTING con cuotas.
    
    Las cuotas NO se usan en entrenamiento, solo para filtrar apuestas.
    """
    
    print("\n" + "="*70)
    print("💰 EVALUACIÓN CON VALUE BETTING")
    print("="*70)
    
    # Verificar que df_test tenga cuotas
    if not all(col in df_test.columns for col in ['B365H', 'B365D', 'B365A']):
        print("⚠️  No hay cuotas disponibles para value betting")
        return
    
    # Obtener probabilidades del modelo
    y_proba = modelo.predict_proba(X_test)
    
    # Probabilidades del mercado (normalizadas)
    prob_mercado_h = 1 / df_test['B365H'].values
    prob_mercado_d = 1 / df_test['B365D'].values
    prob_mercado_a = 1 / df_test['B365A'].values
    
    total = prob_mercado_h + prob_mercado_d + prob_mercado_a
    prob_mercado_h /= total
    prob_mercado_d /= total
    prob_mercado_a /= total
    
    # Calcular EDGE (ventaja del modelo sobre el mercado)
    edge_home = y_proba[:, 0] - prob_mercado_h
    edge_draw = y_proba[:, 1] - prob_mercado_d
    edge_away = y_proba[:, 2] - prob_mercado_a
    
    # Estrategia: Solo apostar cuando edge > umbral
    umbrales = [0.00, 0.03, 0.05, 0.08, 0.10]
    
    print(f"\n📊 RESULTADOS POR UMBRAL DE EDGE:")
    print(f"{'Umbral':<10} {'Apuestas':<12} {'Accuracy':<12} {'ROI':<10}")
    print("-" * 50)
    
    for umbral in umbrales:
        # Encontrar la mejor apuesta para cada partido
        max_edge = np.maximum(np.maximum(edge_home, edge_draw), edge_away)
        mask_apostar = max_edge > umbral
        
        num_apuestas = mask_apostar.sum()
        
        if num_apuestas == 0:
            print(f"{umbral:>6.1%}    {'0':>10}  {'N/A':>10}  {'N/A':>8}")
            continue
        
        # Hacer predicciones solo donde hay edge
        predicciones_value = []
        y_true_value = []
        
        for i in range(len(y_proba)):
            if not mask_apostar[i]:
                continue
            
            # Elegir la apuesta con mayor edge
            edges = [edge_home[i], edge_draw[i], edge_away[i]]
            mejor_apuesta = np.argmax(edges)
            
            predicciones_value.append(mejor_apuesta)
            y_true_value.append(y_test.iloc[i])
        
        # Calcular accuracy
        acc_value = accuracy_score(y_true_value, predicciones_value)
        
        # Calcular ROI simulado (asumiendo apuesta de 1€ cada vez)
        roi = 0
        for i, idx in enumerate(np.where(mask_apostar)[0]):
            apuesta = predicciones_value[i]
            real = y_true_value[i]
            
            if apuesta == real:
                # Ganamos
                if apuesta == 0:  # Local
                    roi += (df_test.iloc[idx]['B365H'] - 1)
                elif apuesta == 1:  # Empate
                    roi += (df_test.iloc[idx]['B365D'] - 1)
                else:  # Visitante
                    roi += (df_test.iloc[idx]['B365A'] - 1)
            else:
                # Perdemos
                roi -= 1
        
        roi_pct = (roi / num_apuestas) * 100
        
        print(f"{umbral:>6.1%}    {num_apuestas:>10}  {acc_value:>10.1%}  {roi_pct:>8.1f}%")
    
    print("\n💡 INTERPRETACIÓN:")
    print("   • Edge = Tu modelo ve valor que el mercado no ve")
    print("   • Edge >5% = Apuestas con buena probabilidad")
    print("   • Edge >8% = Apuestas con alta confianza")
    print("   • ROI >0% = Estrategia rentable")
    print("\n🎯 RECOMENDACIÓN:")
    print("   Usa edge >5% como filtro mínimo para apostar")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "⚽" * 35)
    print("   MODELO OPTIMIZADO + VALUE BETTING")
    print("⚽" * 35 + "\n")
    
    # Cargar datos (devuelve df también)
    X_full, y, features, df = cargar_datos_hibridos()
    
    # Split temporal ANTES de cualquier selección de features
    print("\n🔪 Split 80/20 (temporal)")
    split_idx = int(len(X_full) * 0.8)

    X_train_full = X_full.iloc[:split_idx]
    X_test_full = X_full.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # Usar df filtrado para obtener cuotas en test
    df_test = df.iloc[split_idx:]

    print(f"   Train: {len(X_train_full)} | Test: {len(X_test_full)}")

    # Feature selection: Top 15 — entrenado SOLO sobre train para evitar leakage
    usar_top_features = True
    top_n = 15

    if usar_top_features:
        print(f"\n🔝 Seleccionando top {top_n} features (sobre train únicamente)...")
        rf_temp = RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        rf_temp.fit(X_train_full, y_train)

        importancias = rf_temp.feature_importances_
        top_idx = np.argsort(importancias)[::-1][:top_n]
        top_features = [features[i] for i in top_idx]

        print(f"\nTop {top_n} features seleccionadas:")
        for i, feat in enumerate(top_features, 1):
            print(f"   {i:2}. {feat}")

        X_train = X_train_full[top_features]
        X_test = X_test_full[top_features]
        features = top_features
    else:
        X_train = X_train_full
        X_test = X_test_full
    
    # Entrenar múltiples modelos
    modelos = entrenar_modelos_hibridos(X_train, y_train, X_test, y_test)

    # Comparar y seleccionar el mejor
    mejor_key, mejor = comparar_modelos(modelos, y_test)
    
    # Analizar features H2H y xG
    try:
        analizar_features_h2h(mejor['modelo'], features)
    except Exception:
        pass

    # ✨ EVALUAR CON VALUE BETTING
    evaluar_value_betting(mejor['modelo'], X_test, y_test, df_test, features)

    # Visualizar resultados
    visualizar_resultados(y_test, mejor['predicciones'], mejor['nombre'], 
                         features, mejor['modelo'])

    # Guardar modelo final usando rutas de config
    joblib.dump(mejor['modelo'], ARCHIVO_MODELO_VB)
    joblib.dump(features, ARCHIVO_FEATURES_VB)

    print(f"\n💾 Modelo guardado: {ARCHIVO_MODELO_VB}")
    print(f"💾 Features guardadas: {ARCHIVO_FEATURES_VB}")
    
    # Resumen final
    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"\n🏆 Modelo final: {mejor['nombre']}")
    print(f"📊 Test Accuracy base: {mejor['accuracy']:.2%}")
    print(f"📊 Test F1-Score: {mejor['f1_score']:.4f}")
    print(f"\n📋 Features utilizadas: {len(features)} (SIN cuotas)")
    
    if mejor['accuracy'] >= 0.52:
        print(f"\n✅ Excelente! Accuracy {mejor['accuracy']:.1%} es muy bueno para fútbol")
        print(f"   → Usar con value betting (edge >5%)")
    
    print(f"\n🎯 SIGUIENTE PASO:")
    print(f"   python predecir_jornada_completa.py\n")
    
    return mejor['modelo'], features


if __name__ == "__main__":
    modelo_final, features = main()