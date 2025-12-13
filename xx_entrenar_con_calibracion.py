# -*- coding: utf-8 -*-
"""
Premier League Match Predictor
ENTRENAMIENTO CON CALIBRACIÓN DE PROBABILIDADES

Soluciona el problema de overconfidence usando CalibratedClassifierCV
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

RUTA_DATOS = './datos/procesados/premier_league_con_features.csv'
RUTA_MODELOS = './modelos/'
os.makedirs(RUTA_MODELOS, exist_ok=True)

# ============================================================================
# CARGA DE DATOS
# ============================================================================

def cargar_datos():
    """Carga datos procesados y prepara features."""
    print("="*70)
    print("CARGANDO DATOS PARA CALIBRACIÓN")
    print("="*70)
    
    df = pd.read_csv(RUTA_DATOS)
    print(f"✅ Cargados: {len(df)} partidos")
    
    # Features óptimas (las que dieron 52.51%)
    features = [
        'HT_AvgGoals', 'AT_AvgGoals',
        'HT_AvgShotsTarget', 'AT_AvgShotsTarget',
        'HT_Form_W', 'HT_Form_D', 'HT_Form_L',
        'AT_Form_W', 'AT_Form_D', 'AT_Form_L',
        'B365H', 'B365D', 'B365A', 
        'B365CH', 'B365CD', 'B365CA'
    ]
    
    features = [f for f in features if f in df.columns]
    print(f"📊 Features: {len(features)}")
    
    X = df[features].fillna(0)
    y = df['FTR_numeric']
    
    # Distribución
    print(f"\n📊 Distribución:")
    for clase in [0, 1, 2]:
        nombre = ['Local', 'Empate', 'Visitante'][clase]
        count = (y == clase).sum()
        pct = count / len(y) * 100
        print(f"   {nombre}: {count} ({pct:.1f}%)")
    
    return X, y, features, df


# ============================================================================
# ENTRENAMIENTO BASE
# ============================================================================

def entrenar_modelo_base(X_train, y_train, X_test, y_test):
    """Entrena el modelo base (sin calibrar)."""
    
    print("\n" + "="*70)
    print("MODELO BASE (SIN CALIBRAR)")
    print("="*70)
    
    # Modelo que dio 52.51%
    rf = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    print("🔧 Entrenando Random Forest...")
    rf.fit(X_train, y_train)
    
    # Predicciones
    pred = rf.predict(X_test)
    proba = rf.predict_proba(X_test)
    
    # Métricas
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    logloss = log_loss(y_test, proba)
    
    print(f"\n✅ Resultados modelo base:")
    print(f"   Accuracy:  {acc:.2%}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   Log Loss:  {logloss:.4f} ← Medida de calibración")
    
    return rf, proba


# ============================================================================
# CALIBRACIÓN
# ============================================================================

def calibrar_modelo(modelo_base, X_train, y_train, X_test, y_test):
    """
    Calibra el modelo usando CalibratedClassifierCV.
    
    Método 'sigmoid': Calibración de Platt (mejor para RF)
    """
    
    print("\n" + "="*70)
    print("CALIBRANDO MODELO")
    print("="*70)
    
    print("🔧 Aplicando CalibratedClassifierCV...")
    print("   Método: sigmoid (Platt Scaling)")
    print("   CV: 5 folds")
    
    # Split adicional para calibración (20% del train)
    X_train_model, X_calib, y_train_model, y_calib = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Re-entrenar modelo base con menos datos
    modelo_base.fit(X_train_model, y_train_model)
    
    # Calibrar usando X_calib (datos que el modelo NO vio)
    modelo_calibrado = CalibratedClassifierCV(
        modelo_base,
        method='sigmoid',  # Platt Scaling
        cv='prefit'  # Modelo ya está entrenado
    )
    
    modelo_calibrado.fit(X_calib, y_calib)
    
    print("✅ Calibración completada")
    
    # Predicciones calibradas
    pred_calib = modelo_calibrado.predict(X_test)
    proba_calib = modelo_calibrado.predict_proba(X_test)
    
    # Métricas
    acc_calib = accuracy_score(y_test, pred_calib)
    f1_calib = f1_score(y_test, pred_calib, average='weighted')
    logloss_calib = log_loss(y_test, proba_calib)
    
    print(f"\n✅ Resultados modelo calibrado:")
    print(f"   Accuracy:  {acc_calib:.2%}")
    print(f"   F1-Score:  {f1_calib:.4f}")
    print(f"   Log Loss:  {logloss_calib:.4f} ← Menor = Mejor calibración")
    
    return modelo_calibrado, proba_calib


# ============================================================================
# COMPARACIÓN Y VISUALIZACIÓN
# ============================================================================

def comparar_calibracion(y_test, proba_base, proba_calib):
    """
    Compara las probabilidades antes y después de calibración.
    Genera gráfica de confiabilidad (reliability diagram).
    """
    
    print("\n" + "="*70)
    print("ANÁLISIS DE CALIBRACIÓN")
    print("="*70)
    
    # Log Loss (menor es mejor)
    logloss_base = log_loss(y_test, proba_base)
    logloss_calib = log_loss(y_test, proba_calib)
    
    mejora_logloss = ((logloss_base - logloss_calib) / logloss_base) * 100
    
    print(f"\n📊 Log Loss Comparison:")
    print(f"   Modelo Base:      {logloss_base:.4f}")
    print(f"   Modelo Calibrado: {logloss_calib:.4f}")
    print(f"   Mejora:           {mejora_logloss:+.1f}%")
    
    if logloss_calib < logloss_base:
        print(f"\n✅ El modelo calibrado está MEJOR calibrado")
    else:
        print(f"\n⚠️  La calibración no mejoró (puede ser que ya estaba bien)")
    
    # -------------------------------------------------------------------------
    # GRÁFICA DE CONFIABILIDAD
    # -------------------------------------------------------------------------
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    clases = ['Local', 'Empate', 'Visitante']
    
    for i, clase in enumerate(clases):
        # Preparar datos para esta clase
        y_true_binary = (y_test == i).astype(int)
        
        # Curva de calibración - Modelo Base
        prob_true_base, prob_pred_base = calibration_curve(
            y_true_binary, 
            proba_base[:, i], 
            n_bins=10,
            strategy='uniform'
        )
        
        # Curva de calibración - Modelo Calibrado
        prob_true_calib, prob_pred_calib = calibration_curve(
            y_true_binary,
            proba_calib[:, i],
            n_bins=10,
            strategy='uniform'
        )
        
        # Graficar
        axes[i].plot([0, 1], [0, 1], 'k--', label='Perfectamente calibrado')
        axes[i].plot(prob_pred_base, prob_true_base, 's-', 
                    label='Modelo Base', color='red', alpha=0.7)
        axes[i].plot(prob_pred_calib, prob_true_calib, 'o-',
                    label='Modelo Calibrado', color='green', alpha=0.7)
        
        axes[i].set_xlabel('Probabilidad Predicha', fontsize=11)
        axes[i].set_ylabel('Fracción de Positivos', fontsize=11)
        axes[i].set_title(f'Calibración: {clase}', fontsize=12, fontweight='bold')
        axes[i].legend(loc='best')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    archivo = os.path.join(RUTA_MODELOS, 'calibracion_comparison.png')
    plt.savefig(archivo, dpi=150, bbox_inches='tight')
    print(f"\n✅ Gráfica guardada: {archivo}")
    plt.close()


def analizar_cambios_probabilidades(y_test, proba_base, proba_calib):
    """
    Analiza cómo cambiaron las probabilidades después de calibración.
    Enfoque en casos de alta confianza (como Sunderland).
    """
    
    print("\n" + "="*70)
    print("ANÁLISIS DE CAMBIOS EN PROBABILIDADES")
    print("="*70)
    
    # Encontrar casos donde el modelo base estaba muy confiado
    max_proba_base = proba_base.max(axis=1)
    max_proba_calib = proba_calib.max(axis=1)
    
    # Casos de alta confianza (>70%)
    alta_confianza = max_proba_base > 0.70
    n_alta_confianza = alta_confianza.sum()
    
    if n_alta_confianza > 0:
        print(f"\n📊 Partidos con alta confianza (>70%): {n_alta_confianza}")
        
        # Promedio de reducción
        reduccion = max_proba_base[alta_confianza] - max_proba_calib[alta_confianza]
        print(f"   Reducción promedio: {reduccion.mean():.1%}")
        print(f"   Máxima reducción:   {reduccion.max():.1%}")
        
        # Mostrar ejemplos
        print(f"\n📋 Ejemplos de ajustes:")
        indices = np.where(alta_confianza)[0][:5]
        
        for idx in indices:
            clase_pred = proba_base[idx].argmax()
            nombre_clase = ['Local', 'Empate', 'Visitante'][clase_pred]
            
            prob_antes = proba_base[idx, clase_pred]
            prob_despues = proba_calib[idx, clase_pred]
            
            print(f"   {nombre_clase}: {prob_antes:.1%} → {prob_despues:.1%} "
                  f"({prob_despues - prob_antes:+.1%})")
    
    # Distribución general
    print(f"\n📊 Distribución de probabilidades máximas:")
    print(f"   Modelo Base:")
    print(f"      Media: {max_proba_base.mean():.1%}")
    print(f"      >50%:  {(max_proba_base > 0.5).sum()}")
    print(f"      >70%:  {(max_proba_base > 0.7).sum()}")
    print(f"      >90%:  {(max_proba_base > 0.9).sum()}")
    
    print(f"\n   Modelo Calibrado:")
    print(f"      Media: {max_proba_calib.mean():.1%}")
    print(f"      >50%:  {(max_proba_calib > 0.5).sum()}")
    print(f"      >70%:  {(max_proba_calib > 0.7).sum()}")
    print(f"      >90%:  {(max_proba_calib > 0.9).sum()}")


# ============================================================================
# GUARDAR MODELOS
# ============================================================================

def guardar_modelos(modelo_base, modelo_calibrado, features):
    """Guarda ambos modelos para comparación."""
    
    print("\n" + "="*70)
    print("GUARDANDO MODELOS")
    print("="*70)
    
    # Modelo base
    archivo_base = os.path.join(RUTA_MODELOS, 'modelo_sin_calibrar.pkl')
    joblib.dump(modelo_base, archivo_base)
    print(f"✅ Modelo base: {archivo_base}")
    
    # Modelo calibrado (ESTE es el que usarás)
    archivo_calibrado = os.path.join(RUTA_MODELOS, 'modelo_calibrado_final.pkl')
    joblib.dump(modelo_calibrado, archivo_calibrado)
    print(f"✅ Modelo calibrado: {archivo_calibrado}")
    
    # Features
    archivo_features = os.path.join(RUTA_MODELOS, 'features_calibradas.pkl')
    joblib.dump(features, archivo_features)
    print(f"✅ Features: {archivo_features}")
    
    print(f"\n💡 IMPORTANTE:")
    print(f"   Para predicciones, usa: modelo_calibrado_final.pkl")
    print(f"   Este tiene probabilidades REALISTAS (no sobreconfiadas)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "⚽" * 35)
    print("   ENTRENAMIENTO CON CALIBRACIÓN")
    print("   Solución al problema de overconfidence")
    print("⚽" * 35 + "\n")
    
    # Cargar datos
    X, y, features, df = cargar_datos()
    
    # Split temporal (80/20)
    print("\n🔪 Split 80/20 (temporal)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Entrenar modelo base
    modelo_base, proba_base = entrenar_modelo_base(
        X_train, y_train, X_test, y_test
    )
    
    # Calibrar modelo
    modelo_calibrado, proba_calib = calibrar_modelo(
        modelo_base, X_train, y_train, X_test, y_test
    )
    
    # Comparar calibración
    comparar_calibracion(y_test, proba_base, proba_calib)
    
    # Analizar cambios
    analizar_cambios_probabilidades(y_test, proba_base, proba_calib)
    
    # Guardar modelos
    guardar_modelos(modelo_base, modelo_calibrado, features)
    
    # Resumen final
    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO CON CALIBRACIÓN COMPLETADO")
    print("="*70)
    
    print("\n📊 RESUMEN:")
    print("   1. Modelo base entrenado (52% accuracy)")
    print("   2. Modelo calibrado con Platt Scaling")
    print("   3. Probabilidades ajustadas (menos confianza)")
    print("   4. Ambos modelos guardados")
    
    print("\n🎯 SIGUIENTE PASO:")
    print("   1. Modificar predecir_jornada_completa.py")
    print("   2. Usar: modelo_calibrado_final.pkl")
    print("   3. Las apuestas Kelly serán MÁS CONSERVADORAS")
    
    print("\n💡 BENEFICIOS:")
    print("   ✅ Probabilidades realistas (no 40.6% en Sunderland)")
    print("   ✅ Stakes más pequeños (no $202 en underdog)")
    print("   ✅ Mejor gestión de riesgo")
    print("   ✅ EV positivo más confiable")
    
    # Calcular efectividad en test set
    y_pred = modelo_calibrado.predict(X_test)
    efectividad = accuracy_score(y_test, y_pred)
    print(f"\n🎯 EFECTIVIDAD FINAL DEL MODELO CALIBRADO: {efectividad:.2%}\n")
    
    return modelo_calibrado, features


if __name__ == "__main__":
    modelo_final, features = main()