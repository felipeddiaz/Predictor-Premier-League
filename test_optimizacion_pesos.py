# -*- coding: utf-8 -*-
"""
Script de prueba para la optimización de class_weights
Ejecuta SOLO la búsqueda de pesos sin entrenar todo el modelo
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import os

RUTA_DATOS = './datos/procesados/premier_league_RESTAURADO.csv'

def cargar_datos_basico():
    """Carga datos sin features complejas para prueba rápida."""
    print("📂 Cargando datos...")
    
    if not os.path.exists(RUTA_DATOS):
        print(f"❌ ERROR: No se encontró '{RUTA_DATOS}'")
        return None, None
    
    df = pd.read_csv(RUTA_DATOS)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    
    # Features base simples
    features = [
        'HT_AvgGoals', 'AT_AvgGoals',
        'HT_AvgShotsTarget', 'AT_AvgShotsTarget',
        'HT_Form_W', 'HT_Form_D', 'HT_Form_L',
        'AT_Form_W', 'AT_Form_D', 'AT_Form_L'
    ]
    
    # Verificar que existan
    features = [f for f in features if f in df.columns]
    
    X = df[features].fillna(0)
    y = df['FTR_numeric']
    
    print(f"✅ Cargados: {len(df)} partidos")
    print(f"   Features: {len(features)}")
    print(f"   Distribución: Local={sum(y==0)}, Empate={sum(y==1)}, Visitante={sum(y==2)}")
    
    return X, y

def prueba_optimizacion():
    """Prueba la optimización de pesos."""
    
    print("\n" + "="*70)
    print("TEST: OPTIMIZACIÓN LOCAL DE CLASS_WEIGHTS")
    print("="*70 + "\n")
    
    # Cargar datos
    X, y = cargar_datos_basico()
    if X is None:
        return
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    print(f"\n📊 Split:")
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}\n")
    
    # ========================================================================
    # PARÁMETROS BASE (FIJOS)
    # ========================================================================
    
    BASE_LOCAL = 1.0
    BASE_EMPATE = 2.4
    BASE_VISITANTE = 1.3
    
    print("🎯 PARÁMETROS FIJOS DEL MODELO:")
    print(f"   n_estimators:    100")
    print(f"   min_samples_leaf: 5")
    print(f"   random_state:    42")
    print(f"   Pesos base:      Local={BASE_LOCAL}, Empate={BASE_EMPATE}, Visitante={BASE_VISITANTE}\n")
    
    # ========================================================================
    # GENERAR GRID DE BÚSQUEDA LOCAL
    # ========================================================================
    
    pesos_local = np.round(np.arange(0.8, 1.3, 0.1), 1)
    pesos_empate = np.round(np.arange(2.0, 2.9, 0.2), 1)
    pesos_visitante = np.round(np.arange(1.0, 1.7, 0.2), 1)
    
    class_weights_list = []
    for w_local in pesos_local:
        for w_empate in pesos_empate:
            for w_visitante in pesos_visitante:
                class_weights_list.append({
                    0: float(w_local),
                    1: float(w_empate),
                    2: float(w_visitante)
                })
    
    print(f"🔍 GRID DE BÚSQUEDA:")
    print(f"   Local:     {sorted(set(pesos_local))}")
    print(f"   Empate:    {sorted(set(pesos_empate))}")
    print(f"   Visitante: {sorted(set(pesos_visitante))}")
    print(f"   Total combinaciones: {len(class_weights_list)}\n")
    
    # ========================================================================
    # MODELO BASE
    # ========================================================================
    
    base_model = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    # ========================================================================
    # GRID SEARCH
    # ========================================================================
    
    param_grid = {'class_weight': class_weights_list}
    
    print("⏳ Ejecutando GridSearchCV (esto tarda ~1-2 minutos)...")
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    # ========================================================================
    # RESULTADOS
    # ========================================================================
    
    print("✅ BÚSQUEDA COMPLETADA\n")
    print("="*70)
    
    # Modelo optimizado
    modelo_opt = grid_search.best_estimator_
    pred_opt = modelo_opt.predict(X_test)
    acc_opt = accuracy_score(y_test, pred_opt)
    f1_opt = f1_score(y_test, pred_opt, average='weighted')
    f1_macro_opt = f1_score(y_test, pred_opt, average='macro')
    
    # Modelo baseline (con pesos actuales)
    pesos_actuales = {0: BASE_LOCAL, 1: BASE_EMPATE, 2: BASE_VISITANTE}
    modelo_base = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        class_weight=pesos_actuales,
        random_state=42,
        n_jobs=-1
    )
    modelo_base.fit(X_train, y_train)
    pred_base = modelo_base.predict(X_test)
    acc_base = accuracy_score(y_test, pred_base)
    f1_base = f1_score(y_test, pred_base, average='weighted')
    f1_macro_base = f1_score(y_test, pred_base, average='macro')
    
    # Mostrar pesos encontrados
    best_weights = grid_search.best_params_['class_weight']
    
    print("\n📋 PESOS OPTIMIZADOS ENCONTRADOS:")
    print(f"   Local (0):     {best_weights[0]:.2f}")
    print(f"   Empate (1):    {best_weights[1]:.2f}")
    print(f"   Visitante (2): {best_weights[2]:.2f}")
    print(f"\n   F1-Weighted en CV: {grid_search.best_score_:.4f}")
    
    # Comparación
    print("\n📊 COMPARACIÓN:")
    print(f"\n{'Métrica':<20} {'Actuales':<15} {'Optimizados':<15} {'Mejora':<12}")
    print("-" * 70)
    
    mejora_acc = ((acc_opt - acc_base) / acc_base) * 100 if acc_base != 0 else 0
    mejora_f1 = ((f1_opt - f1_base) / f1_base) * 100 if f1_base != 0 else 0
    mejora_f1_macro = ((f1_macro_opt - f1_macro_base) / f1_macro_base) * 100 if f1_macro_base != 0 else 0
    
    print(f"{'Accuracy':<20} {acc_base:>13.2%} {acc_opt:>13.2%} {mejora_acc:>+10.2f}%")
    print(f"{'F1-Weighted':<20} {f1_base:>13.4f} {f1_opt:>13.4f} {mejora_f1:>+10.2f}%")
    print(f"{'F1-Macro':<20} {f1_macro_base:>13.4f} {f1_macro_opt:>13.4f} {mejora_f1_macro:>+10.2f}%")
    
    # Diferencia de pesos
    print("\n🎯 CAMBIOS EN PESOS:")
    print(f"\n{'Clase':<20} {'Actual':<12} {'Óptimo':<12} {'Cambio':<12}")
    print("-" * 70)
    
    for i, nombre in enumerate(['Local (0)', 'Empate (1)', 'Visitante (2)']):
        peso_actual = pesos_actuales[i]
        peso_optimo = best_weights[i]
        cambio = ((peso_optimo - peso_actual) / peso_actual) * 100
        print(f"{nombre:<20} {peso_actual:>10.2f} {peso_optimo:>12.2f} {cambio:>+10.1f}%")
    
    # Recomendación
    print("\n" + "="*70)
    if f1_opt >= f1_base:
        print("✅ RECOMENDACIÓN: Usar pesos optimizados")
        if f1_opt > f1_base:
            print(f"   Mejora de F1-Weighted: {mejora_f1:+.2f}%")
        print(f"\n   Nuevos pesos:\n   {best_weights}")
    else:
        print("⚠️  RECOMENDACIÓN: Mantener pesos actuales")
        print(f"   Los optimizados NO mejoran el F1-Weighted")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    prueba_optimizacion()
