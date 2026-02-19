# -*- coding: utf-8 -*-
"""
Premier League Match Predictor
02 - ENTRENAMIENTO DE MODELOS (VERSIÓN OPTIMIZADA CON PESOS OPTUNA)
Entrena modelos RF con pesos de clase optimizados por Optuna
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
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

RUTA_DATOS = './datos/procesados/premier_league_RESTAURADO.csv'
RUTA_MODELOS = './modelos/'

os.makedirs(RUTA_MODELOS, exist_ok=True)

# ============================================================================
# PESOS OPTIMIZADOS POR OPTUNA
# ============================================================================

# Estos pesos fueron encontrados mediante 100 trials de optimización
# maximizando F1-Score con TimeSeriesSplit cross-validation
PESOS_OPTIMOS = {
    0: 1.2486,  # Local
    1: 3.3228,  # Empate (subió de 2.8 a 3.3)
    2: 1.9519   # Visitante
}

# Hiperparámetros óptimos encontrados por Optuna
PARAMS_OPTIMOS = {
    'n_estimators': 229,
    'max_depth': 8,        # Subió de 6 a 8
    'min_samples_leaf': 3, # Bajó de 8 a 3
    'class_weight': PESOS_OPTIMOS,
    'random_state': 42,
    'n_jobs': -1
}

# ============================================================================
# FUNCIÓN PARA AGREGAR FEATURES DERIVADAS DE CUOTAS
# ============================================================================

def agregar_features_cuotas_derivadas(df):
    """
    Agrega features derivadas de las cuotas de apuestas.
    Incluye probabilidades implícitas, movimiento de mercado y confianza.
    
    Requiere columnas: B365H, B365D, B365A, B365CH, B365CD, B365CA
    """
    print("\n🔧 Calculando features derivadas de cuotas...")
    
    # Verificar columnas necesarias
    cols_necesarias = ['B365H', 'B365D', 'B365A', 'B365CH', 'B365CD', 'B365CA']
    cols_faltantes = [c for c in cols_necesarias if c not in df.columns]
    
    if cols_faltantes:
        print(f"   ⚠️ Faltan columnas: {cols_faltantes}")
        return df
    
    # -------------------------------------------------------------------------
    # 1. PROBABILIDADES IMPLÍCITAS BASE
    # -------------------------------------------------------------------------
    # Cuotas de apertura
    df['Prob_H'] = 1 / df['B365H']
    df['Prob_D'] = 1 / df['B365D']
    df['Prob_A'] = 1 / df['B365A']
    
    # Cuotas de cierre
    df['ProbC_H'] = 1 / df['B365CH']
    df['ProbC_D'] = 1 / df['B365CD']
    df['ProbC_A'] = 1 / df['B365CA']
    
    # -------------------------------------------------------------------------
    # 2. MOVIMIENTO DE MERCADO (muy valiosas)
    # -------------------------------------------------------------------------
    # Cambio en probabilidad entre apertura y cierre
    df['Prob_Move_H'] = df['ProbC_H'] - df['Prob_H']
    df['Prob_Move_D'] = df['ProbC_D'] - df['Prob_D']
    df['Prob_Move_A'] = df['ProbC_A'] - df['Prob_A']
    
    # Intensidad total del movimiento (detecta "algo raro pasó")
    df['Market_Move_Strength'] = (
        df['Prob_Move_H'].abs() + 
        df['Prob_Move_D'].abs() + 
        df['Prob_Move_A'].abs()
    )
    
    # -------------------------------------------------------------------------
    # 3. ESTRUCTURA DEL MERCADO
    # -------------------------------------------------------------------------
    # Probabilidad máxima (favorito) y spread
    df['Prob_Max'] = df[['Prob_H', 'Prob_D', 'Prob_A']].max(axis=1)
    df['Prob_Min'] = df[['Prob_H', 'Prob_D', 'Prob_A']].min(axis=1)
    df['Prob_Spread'] = df['Prob_Max'] - df['Prob_Min']
    
    # Confianza del mercado (qué tan seguro está)
    # Si Prob_Max ≈ 0.33 → mercado inseguro
    # Si Prob_Max > 0.50 → mercado muy seguro
    df['Market_Confidence'] = df['Prob_Max'] - (1/3)
    
    # Ventaja del local según cuotas
    df['Home_Advantage_Prob'] = df['Prob_H'] - df['Prob_A']
    
    # -------------------------------------------------------------------------
    # LIMPIEZA
    # -------------------------------------------------------------------------
    # Eliminar probabilidades de cierre (intermedias, no necesarias como features)
    df = df.drop(columns=['ProbC_H', 'ProbC_D', 'ProbC_A'], errors='ignore')
    
    # Rellenar NaN (por si hay cuotas faltantes)
    features_nuevas = [
        'Prob_H', 'Prob_D', 'Prob_A',
        'Prob_Move_H', 'Prob_Move_D', 'Prob_Move_A',
        'Market_Move_Strength',
        'Prob_Max', 'Prob_Min', 'Prob_Spread',
        'Market_Confidence', 'Home_Advantage_Prob'
    ]
    
    for col in features_nuevas:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Stats
    con_datos = (df['Prob_H'] > 0).sum()
    print(f"   ✅ Features de cuotas derivadas: {con_datos} partidos con datos")
    print(f"   ✅ Nuevas features agregadas: {len(features_nuevas)}")
    
    return df

# ============================================================================
# FUNCIÓN PARA AGREGAR xG ROLLING
# ============================================================================

def agregar_xg_rolling(df, window=5):
    """
    Agrega features xG rolling (promedios históricos).
    Se calcula en memoria, no modifica el CSV original.
    """
    print("\n🔧 Calculando xG rolling...")
    
    if 'Home_xG' not in df.columns or 'Away_xG' not in df.columns:
        print("   ⚠️ No hay datos de xG disponibles")
        return df
    
    # Ordenar de forma determinística
    df = df.sort_values(['Date', 'HomeTeam', 'AwayTeam']).reset_index(drop=True)
    
    # Inicializar columnas
    df['HT_xG_Avg'] = np.nan
    df['AT_xG_Avg'] = np.nan
    df['HT_xGA_Avg'] = np.nan
    df['AT_xGA_Avg'] = np.nan
    
    # Iterar en orden alfabético para determinismo
    for team in sorted(df['HomeTeam'].unique()):
        mask_home = df['HomeTeam'] == team
        mask_away = df['AwayTeam'] == team
        
        # xG creado por el equipo (shift para evitar leakage)
        df.loc[mask_home, 'HT_xG_Avg'] = (
            df.loc[mask_home, 'Home_xG']
            .shift(1)
            .rolling(window, min_periods=1)
            .mean()
            .values
        )
        df.loc[mask_away, 'AT_xG_Avg'] = (
            df.loc[mask_away, 'Away_xG']
            .shift(1)
            .rolling(window, min_periods=1)
            .mean()
            .values
        )
        
        # xG concedido por el equipo
        df.loc[mask_home, 'HT_xGA_Avg'] = (
            df.loc[mask_home, 'Away_xG']
            .shift(1)
            .rolling(window, min_periods=1)
            .mean()
            .values
        )
        df.loc[mask_away, 'AT_xGA_Avg'] = (
            df.loc[mask_away, 'Home_xG']
            .shift(1)
            .rolling(window, min_periods=1)
            .mean()
            .values
        )
    
    # Features derivadas
    df['xG_Diff'] = df['HT_xG_Avg'] - df['AT_xG_Avg']
    df['xG_Total'] = df['HT_xG_Avg'] + df['AT_xG_Avg']
    
    # Rellenar NaN
    xg_cols = ['HT_xG_Avg', 'AT_xG_Avg', 'HT_xGA_Avg', 'AT_xGA_Avg', 'xG_Diff', 'xG_Total']
    df[xg_cols] = df[xg_cols].fillna(0)
    
    con_xg = (df['HT_xG_Avg'] > 0).sum()
    print(f"   ✅ xG rolling agregado: {con_xg} partidos con datos históricos")
    
    return df

# ============================================================================
# FUNCIÓN PARA AGREGAR FEATURES DE POSICIÓN EN TABLA
# ============================================================================

def agregar_features_tabla(df):
    """
    Calcula features basadas en la posición en la tabla.
    Se calcula de forma temporal (solo partidos anteriores) para evitar leakage.
    """
    print("\n🔧 Calculando features de posición en tabla...")
    
    # Ordenar de forma determinística
    df = df.sort_values(['Date', 'HomeTeam', 'AwayTeam']).reset_index(drop=True)
    
    # Inicializar columnas
    df['HT_Position'] = np.nan
    df['AT_Position'] = np.nan
    df['HT_Points'] = np.nan
    df['AT_Points'] = np.nan
    df['Position_Diff'] = np.nan
    df['HT_Points_Above'] = np.nan
    df['HT_Points_Below'] = np.nan
    df['AT_Points_Above'] = np.nan
    df['AT_Points_Below'] = np.nan
    df['Matchday'] = np.nan
    df['Season_Progress'] = np.nan
    df['Position_Reliability'] = np.nan
    
    # Identificar temporadas
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df['Season'] = df['Date'].apply(lambda x: f"{x.year}-{x.year+1}" if x.month >= 8 else f"{x.year-1}-{x.year}")
        temporadas = sorted(df['Season'].unique())
    else:
        temporadas = [None]
    
    print(f"   Procesando {len(temporadas)} temporadas...")
    
    for temporada in temporadas:
        if temporada is None:
            mask_temporada = df.index.isin(df.index)
        else:
            mask_temporada = df['Season'] == temporada
        
        indices_temporada = df[mask_temporada].index.tolist()
        
        if len(indices_temporada) == 0:
            continue
        
        puntos_equipo = {}
        goles_favor = {}
        goles_contra = {}
        partidos_jugados = {}
        
        # Convertir a lista ordenada para determinismo
        equipos_temporada = sorted(
            set(df.loc[mask_temporada, 'HomeTeam'].unique()) | 
            set(df.loc[mask_temporada, 'AwayTeam'].unique())
        )
        
        for equipo in equipos_temporada:
            puntos_equipo[equipo] = 0
            goles_favor[equipo] = 0
            goles_contra[equipo] = 0
            partidos_jugados[equipo] = 0
        
        fechas_unicas = sorted(df.loc[mask_temporada, 'Date'].unique())
        fecha_to_jornada = {fecha: i+1 for i, fecha in enumerate(fechas_unicas)}
        
        for idx in indices_temporada:
            row = df.loc[idx]
            home = row['HomeTeam']
            away = row['AwayTeam']
            fecha = row['Date']
            
            jornada = fecha_to_jornada.get(fecha, 1)
            total_jornadas = len(fechas_unicas)
            
            # ANTES del partido: calcular posiciones actuales
            if partidos_jugados.get(home, 0) > 0 or partidos_jugados.get(away, 0) > 0:
                tabla = []
                for equipo in equipos_temporada:
                    if partidos_jugados.get(equipo, 0) > 0:
                        tabla.append({
                            'Equipo': equipo,
                            'Puntos': puntos_equipo.get(equipo, 0),
                            'GD': goles_favor.get(equipo, 0) - goles_contra.get(equipo, 0),
                            'GF': goles_favor.get(equipo, 0),
                            'GC': goles_contra.get(equipo, 0),
                            'PJ': partidos_jugados.get(equipo, 0)
                        })
                
                if len(tabla) > 0:
                    tabla_df = pd.DataFrame(tabla)
                    tabla_df = tabla_df.sort_values(
                            by=['Puntos', 'GD', 'GF', 'GC'], 
                            ascending=[False, False, False, True]
                    ).reset_index(drop=True)
                    tabla_df['Posicion'] = range(1, len(tabla_df) + 1)
                    
                    pos_home = tabla_df[tabla_df['Equipo'] == home]['Posicion'].values
                    pos_away = tabla_df[tabla_df['Equipo'] == away]['Posicion'].values
                    pts_home = puntos_equipo.get(home, 0)
                    pts_away = puntos_equipo.get(away, 0)
                    
                    if len(pos_home) > 0:
                        df.at[idx, 'HT_Position'] = pos_home[0]
                        df.at[idx, 'HT_Points'] = pts_home
                        
                        pos_h = int(pos_home[0])
                        if pos_h > 1:
                            equipo_arriba = tabla_df[tabla_df['Posicion'] == pos_h - 1]['Puntos'].values
                            if len(equipo_arriba) > 0:
                                df.at[idx, 'HT_Points_Below'] = equipo_arriba[0] - pts_home
                        if pos_h < len(tabla_df):
                            equipo_abajo = tabla_df[tabla_df['Posicion'] == pos_h + 1]['Puntos'].values
                            if len(equipo_abajo) > 0:
                                df.at[idx, 'HT_Points_Above'] = pts_home - equipo_abajo[0]
                    
                    if len(pos_away) > 0:
                        df.at[idx, 'AT_Position'] = pos_away[0]
                        df.at[idx, 'AT_Points'] = pts_away
                        
                        pos_a = int(pos_away[0])
                        if pos_a > 1:
                            equipo_arriba = tabla_df[tabla_df['Posicion'] == pos_a - 1]['Puntos'].values
                            if len(equipo_arriba) > 0:
                                df.at[idx, 'AT_Points_Below'] = equipo_arriba[0] - pts_away
                        if pos_a < len(tabla_df):
                            equipo_abajo = tabla_df[tabla_df['Posicion'] == pos_a + 1]['Puntos'].values
                            if len(equipo_abajo) > 0:
                                df.at[idx, 'AT_Points_Above'] = pts_away - equipo_abajo[0]
                    
                    if len(pos_home) > 0 and len(pos_away) > 0:
                        df.at[idx, 'Position_Diff'] = pos_away[0] - pos_home[0]
            
            df.at[idx, 'Matchday'] = jornada
            df.at[idx, 'Season_Progress'] = jornada / max(total_jornadas, 1)
            
            min_partidos = min(partidos_jugados.get(home, 0), partidos_jugados.get(away, 0))
            df.at[idx, 'Position_Reliability'] = min(min_partidos / 10, 1.0)
            
            # DESPUÉS del partido: actualizar puntos
            if pd.notna(row.get('FTHG')) and pd.notna(row.get('FTAG')):
                goles_h = int(row['FTHG'])
                goles_a = int(row['FTAG'])
                
                goles_favor[home] = goles_favor.get(home, 0) + goles_h
                goles_contra[home] = goles_contra.get(home, 0) + goles_a
                goles_favor[away] = goles_favor.get(away, 0) + goles_a
                goles_contra[away] = goles_contra.get(away, 0) + goles_h
                
                if goles_h > goles_a:
                    puntos_equipo[home] = puntos_equipo.get(home, 0) + 3
                elif goles_h < goles_a:
                    puntos_equipo[away] = puntos_equipo.get(away, 0) + 3
                else:
                    puntos_equipo[home] = puntos_equipo.get(home, 0) + 1
                    puntos_equipo[away] = puntos_equipo.get(away, 0) + 1
                
                partidos_jugados[home] = partidos_jugados.get(home, 0) + 1
                partidos_jugados[away] = partidos_jugados.get(away, 0) + 1
    
    # Features derivadas
    df['HT_Level'] = pd.cut(df['HT_Position'], bins=[0, 6, 14, 20], labels=[3, 2, 1])
    df['AT_Level'] = pd.cut(df['AT_Position'], bins=[0, 6, 14, 20], labels=[3, 2, 1])
    df['HT_Level'] = pd.to_numeric(df['HT_Level'], errors='coerce')
    df['AT_Level'] = pd.to_numeric(df['AT_Level'], errors='coerce')
    
    df['Match_Type'] = df['HT_Level'] - df['AT_Level']
    df['HT_Pressure'] = (df['HT_Points_Below'].fillna(0) + df['HT_Points_Above'].fillna(0)) / 2
    df['AT_Pressure'] = (df['AT_Points_Below'].fillna(0) + df['AT_Points_Above'].fillna(0)) / 2
    df['Position_Diff_Weighted'] = df['Position_Diff'] * df['Position_Reliability']
    
    # Rellenar NaN
    tabla_cols = [
        'HT_Position', 'AT_Position', 'HT_Points', 'AT_Points',
        'Position_Diff', 'HT_Points_Above', 'HT_Points_Below',
        'AT_Points_Above', 'AT_Points_Below', 'Matchday', 'Season_Progress',
        'Position_Reliability', 'HT_Level', 'AT_Level',
        'Match_Type', 'HT_Pressure', 'AT_Pressure', 'Position_Diff_Weighted'
    ]
    
    for col in tabla_cols:
        if col in df.columns:
            if col in ['HT_Position', 'AT_Position']:
                df[col] = df[col].fillna(10)
            elif col in ['HT_Level', 'AT_Level']:
                df[col] = df[col].fillna(2)
            elif col == 'Season_Progress':
                df[col] = df[col].fillna(0.5)
            elif col == 'Position_Reliability':
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(0)
    
    if 'Season' in df.columns:
        df = df.drop(columns=['Season'])
    
    con_posicion = (df['HT_Position'] != 10).sum()
    print(f"   ✅ Features de tabla calculadas: {con_posicion} partidos con posición real")
    
    return df

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
    
    # Agregar features calculadas en memoria
    df = agregar_xg_rolling(df)
    df = agregar_features_tabla(df)
    df = agregar_features_cuotas_derivadas(df)
    
    # ========================================================================
    # DEFINIR FEATURES (Base + Cuotas + H2H + xG + Tabla)
    # ========================================================================
    
    # Features base (forma reciente)
    features_base = [
        'HT_AvgGoals', 'AT_AvgGoals',
        'HT_AvgShotsTarget', 'AT_AvgShotsTarget',
        'HT_Form_W', 'HT_Form_D', 'HT_Form_L',
        'AT_Form_W', 'AT_Form_D', 'AT_Form_L'
    ]
    
    # Features de cuotas
    features_cuotas = ['B365H', 'B365D', 'B365A', 'B365CH', 'B365CD', 'B365CA']
    
    # Features xG rolling (calculadas en memoria)
    features_xg = [
        'HT_xG_Avg', 'AT_xG_Avg',
        'HT_xGA_Avg', 'AT_xGA_Avg',
        'xG_Diff', 'xG_Total'
    ]
    
    # Features H2H
    features_h2h = [
        'H2H_Available',
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
    
    # Features de tabla (calculadas en memoria)
    features_tabla = [
        'HT_Position', 'AT_Position',
        'Position_Diff', 'Position_Diff_Weighted',
        'HT_Points', 'AT_Points',
        'Season_Progress', 'Position_Reliability',
        'Match_Type',
        'HT_Pressure', 'AT_Pressure'
    ]

    features_cuotas_derivadas = [
    # Probabilidades implícitas base
    'Prob_H', 'Prob_D', 'Prob_A',
    
    # Movimiento de mercado
    'Prob_Move_H', 'Prob_Move_D', 'Prob_Move_A',
    'Market_Move_Strength',
    
    # Estructura del mercado
    'Prob_Spread',
    'Market_Confidence',
    'Home_Advantage_Prob'
]

# Nota: Prob_Max y Prob_Min se calculan pero no se usan como features
# porque son redundantes con Prob_H/D/A. Solo sirven para calcular Prob_Spread.

    # Combinar todas las features
    all_features = (features_base + features_cuotas + features_cuotas_derivadas +  features_xg + features_h2h + features_h2h_derivadas + features_tabla)

    # Filtrar solo las que existen en el DataFrame
    features = [f for f in all_features if f in df.columns]
    
    print(f"\n📊 Features totales: {len(features)}")
    print(f"   • Base: {len([f for f in features_base if f in features])}")
    print(f"   • Cuotas: {len([f for f in features_cuotas if f in features])}")
    print(f"   • xG rolling: {len([f for f in features_xg if f in features])}")
    print(f"   • H2H: {len([f for f in features_h2h if f in features])}")
    print(f"   • H2H derivadas: {len([f for f in features_h2h_derivadas if f in features])}")
    print(f"   • Tabla: {len([f for f in features_tabla if f in features])}")
    print(f"   • Cuotas derivadas: {len([f for f in features_cuotas_derivadas if f in features])}")
    
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
    
    # Modelo
    archivo_modelo = os.path.join(RUTA_MODELOS, 'modelo_final_optimizado.pkl')
    joblib.dump(modelo, archivo_modelo)
    print(f"✅ Modelo: {archivo_modelo}")
    
    # Features
    archivo_features = os.path.join(RUTA_MODELOS, 'features.pkl')
    joblib.dump(features, archivo_features)
    print(f"✅ Features: {archivo_features}")
    
    # Metadata incluyendo pesos Optuna
    metadata = {
        'nombre_modelo': nombre_modelo,
        'n_features': len(features),
        'features': features,
        'pesos_optuna': PESOS_OPTIMOS,
        'params_optuna': PARAMS_OPTIMOS
    }
    archivo_meta = os.path.join(RUTA_MODELOS, 'metadata.pkl')
    joblib.dump(metadata, archivo_meta)
    print(f"✅ Metadata: {archivo_meta}")


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
    
    # Visualizaciones
    visualizar_resultados(y_test, pred_final, nombre_final, features, modelo_final)
    
    # Guardar modelo calibrado
    guardar_modelo_final(modelo_calibrado, features, nombre_final)
    
    # Resumen
    acc_final = accuracy_score(y_test, pred_final)
    f1_final = f1_score(y_test, pred_final, average='weighted')
    
    print("\n" + "="*70)
    print("✅ ¡ENTRENAMIENTO COMPLETADO!")
    print("="*70)
    print(f"\n🏆 Modelo: {nombre_final}")
    print(f"📊 Accuracy: {acc_final:.2%}")
    print(f"📊 F1-Score: {f1_final:.4f}")
    
    if mejor_key == 'RF_Optuna':
        print(f"\n⭐ Pesos Optuna utilizados:")
        print(f"   Local: {PESOS_OPTIMOS[0]:.4f}")
        print(f"   Empate: {PESOS_OPTIMOS[1]:.4f}")
        print(f"   Visitante: {PESOS_OPTIMOS[2]:.4f}")
    
    print(f"\n📁 Archivos:")
    print(f"   • modelos/modelo_final_optimizado.pkl")
    print(f"   • modelos/features.pkl")
    print(f"   • modelos/metadata.pkl")
    print(f"   • modelos/confusion_matrix_final.png")
    print(f"   • modelos/feature_importance_final.png")
    print(f"\n➡️  Siguiente: python 03_predecir_partidos.py\n")
    
    return modelo_final, features


if __name__ == "__main__":
    modelo_final, features = main()