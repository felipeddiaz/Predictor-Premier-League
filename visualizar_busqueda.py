# -*- coding: utf-8 -*-
"""
Premier League Match Predictor
02 - ENTRENAMIENTO DE MODELOS (VERSIÓN OPTIMIZADA)
Entrena 3 modelos, elige el mejor y lo optimiza automáticamente
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
import random
warnings.filterwarnings('ignore')

# ============================================================================
# FIJAR SEEDS PARA REPRODUCIBILIDAD EXACTA
# ============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Documentación importante
print("""
⚠️  REPRODUCIBILIDAD: Para garantizar resultados idénticos en cada ejecución:
    1. Los seeds están fijados en SEED = 42
    2. n_jobs=1 (ejecución en serie, no paralela)
    3. RandomizedSearchCV usa random_state=SEED
    
Si ejecutas el script 2 veces, obtendrás EXACTAMENTE los mismos resultados.
""")


# Librerías opcionales
try:
    from imblearn.over_sampling import SMOTE
    from xgboost import XGBClassifier
    ADVANCED_LIBS = True
except ImportError:
    print("⚠️  XGBoost no disponible. Instala con: pip install xgboost imbalanced-learn")
    ADVANCED_LIBS = False

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

RUTA_DATOS = './datos/procesados/premier_league_RESTAURADO.csv'
RUTA_MODELOS = './modelos/'

os.makedirs(RUTA_MODELOS, exist_ok=True)

# ============================================================================
# PESOS PERSONALIZADOS - TU MEJOR CONFIGURACIÓN (NO MODIFICAR)
# ============================================================================
PESOS_PERSONALIZADOS = {
    0: 1.0,   # Local
    1: 2.4,   # Empate (más peso para detectar empates)
    2: 1.3    # Visitante
}

# ============================================================================
# FUNCIÓN PARA AGREGAR xG ROLLING
# ============================================================================

def agregar_xg_rolling(df, window=5):
    """
    Agrega features xG rolling (promedios históricos).
    Se calcula en memoria, no modifica el CSV original.
    """
    print("\n🔧 Calculando xG rolling...")
    
    if 'Home_xG' not in df.columns:
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
    
    # Ordenar de forma determinística (Date + índice original como desempate)
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
    df['Season'] = df['Date'].apply(lambda x: f"{x.year}-{x.year+1}" if x.month >= 8 else f"{x.year-1}-{x.year}")
    
    temporadas = sorted(df['Season'].unique())  # Ordenar temporadas
    print(f"   Procesando {len(temporadas)} temporadas...")
    
    for temporada in temporadas:
        mask_temporada = df['Season'] == temporada
        indices_temporada = df[mask_temporada].index.tolist()
        
        if len(indices_temporada) == 0:
            continue
        
        puntos_equipo = {}
        goles_favor = {}
        goles_contra = {}
        partidos_jugados = {}
        
        # IMPORTANTE: Convertir a lista ordenada para determinismo
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
            if pd.notna(row['FTHG']) and pd.notna(row['FTAG']):
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
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    print(f"✅ Cargados: {len(df)} partidos")
    
    # Agregar xG rolling (en memoria)
    df = agregar_xg_rolling(df)
    
    # Agregar features de tabla (en memoria)
    df = agregar_features_tabla(df)
    
    # ========================================================================
    # DEFINIR FEATURES
    # ========================================================================
    
    # Features base (forma reciente)
    features_base = [
        'HT_AvgGoals', 'AT_AvgGoals',
        'HT_AvgShotsTarget', 'AT_AvgShotsTarget',
        'HT_Form_W', 'HT_Form_D', 'HT_Form_L',
        'AT_Form_W', 'AT_Form_D', 'AT_Form_L'
    ]
    
    # Features H2H (CON flag de disponibilidad)
    features_h2h = [
        'H2H_Available',
        'H2H_Matches','H2H_Home_Wins','H2H_Draws','H2H_Away_Wins','H2H_Home_Goals_Avg',
        'H2H_Away_Goals_Avg','H2H_Home_Win_Rate','H2H_BTTS_Rate'
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
    
    # Features de tabla (calculadas en memoria)
    features_tabla = [
        'HT_Position', 'AT_Position',
        'Position_Diff', 'Position_Diff_Weighted',
        'HT_Points', 'AT_Points',
        'Season_Progress', 'Position_Reliability',
        'Match_Type',
        'HT_Pressure', 'AT_Pressure'
    ]
    
    # Combinar features
    features = features_base.copy()
    
    # Agregar cuotas si existen
    cuotas = ['B365H', 'B365D', 'B365A', 'B365CH', 'B365CD', 'B365CA']
    cuotas_disponibles = [col for col in cuotas if col in df.columns]
    features.extend(cuotas_disponibles)
    
    # Agregar xG rolling
    xg_disponibles = [col for col in features_xg if col in df.columns]
    features.extend(xg_disponibles)
    
    # Agregar H2H si están disponibles
    h2h_disponibles = [col for col in features_h2h if col in df.columns]
    features.extend(h2h_disponibles)
    
    # Agregar derivadas H2H si están disponibles
    h2h_derivadas_disponibles = [col for col in features_h2h_derivadas if col in df.columns]
    features.extend(h2h_derivadas_disponibles)
    
    # Agregar features de tabla
    tabla_disponibles = [col for col in features_tabla if col in df.columns]
    features.extend(tabla_disponibles)
    
    # Verificar features finales
    features = [f for f in features if f in df.columns]
    
    print(f"\n📊 Features totales: {len(features)}")
    print(f"   • Base: {len([f for f in features_base if f in features])}")
    print(f"   • Cuotas: {len(cuotas_disponibles)}")
    print(f"   • xG rolling: {len(xg_disponibles)}")
    print(f"   • H2H: {len(h2h_disponibles)}")
    print(f"   • H2H derivadas: {len(h2h_derivadas_disponibles)}")
    print(f"   • Tabla: {len(tabla_disponibles)}")
    
    # Verificar si H2H_Available está presente
    if 'H2H_Available' in features:
        print(f"\n✅ H2H_Available presente en features")
        con_h2h = df['H2H_Available'].sum()
        sin_h2h = len(df) - con_h2h
        print(f"   Partidos CON H2H: {con_h2h} ({con_h2h/len(df)*100:.1f}%)")
        print(f"   Partidos SIN H2H: {sin_h2h} ({sin_h2h/len(df)*100:.1f}%)")
    else:
        print(f"\n⚠️  H2H_Available NO encontrado")
        print(f"   Ejecuta: python 01_preparar_datos.py")
    
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

def entrenar_todos_los_modelos(X_train, y_train, X_test, y_test):
    """Entrena los 3 modelos y devuelve resultados."""
    
    resultados = {}
    
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
        n_jobs=1
    )
    rf_basico.fit(X_train, y_train)
    pred_basico = rf_basico.predict(X_test)
    
    acc_basico = accuracy_score(y_test, pred_basico)
    f1_basico = f1_score(y_test, pred_basico, average='weighted')
    
    resultados['RF_Basico'] = {
        'modelo': rf_basico,
        'predicciones': pred_basico,
        'accuracy': acc_basico,
        'f1_score': f1_basico,
        'nombre': 'Random Forest Básico'
    }
    
    print(f"✅ Accuracy: {acc_basico:.2%} | F1-Score: {f1_basico:.4f}")
    
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
        n_jobs=1
    )
    rf_balanceado.fit(X_train, y_train)
    pred_balanceado = rf_balanceado.predict(X_test)
    
    acc_balanceado = accuracy_score(y_test, pred_balanceado)
    f1_balanceado = f1_score(y_test, pred_balanceado, average='weighted')
    
    resultados['RF_Balanceado'] = {
        'modelo': rf_balanceado,
        'predicciones': pred_balanceado,
        'accuracy': acc_balanceado,
        'f1_score': f1_balanceado,
        'nombre': 'Random Forest Balanceado'
    }
    
    print(f"✅ Accuracy: {acc_balanceado:.2%} | F1-Score: {f1_balanceado:.4f}")
    
    # -------------------------------------------------------------------------
    # MODELO 3: Random Forest Pesos Personalizados (TU MEJOR MODELO)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("MODELO 3: RANDOM FOREST PESOS PERSONALIZADOS")
    print("="*70)
    
    print(f"   Pesos: Local={PESOS_PERSONALIZADOS[0]}, Empate={PESOS_PERSONALIZADOS[1]}, Visitante={PESOS_PERSONALIZADOS[2]}")
    
    rf_custom = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        class_weight=PESOS_PERSONALIZADOS,
        random_state=42,
        n_jobs=1
    )
    rf_custom.fit(X_train, y_train)
    pred_custom = rf_custom.predict(X_test)
    
    acc_custom = accuracy_score(y_test, pred_custom)
    f1_custom = f1_score(y_test, pred_custom, average='weighted')
    
    resultados['RF_Custom'] = {
        'modelo': rf_custom,
        'predicciones': pred_custom,
        'accuracy': acc_custom,
        'f1_score': f1_custom,
        'nombre': 'Random Forest Pesos Personalizados'
    }
    
    print(f"✅ Accuracy: {acc_custom:.2%} | F1-Score: {f1_custom:.4f}")
    
    # ========================================================================
    # ANÁLISIS POR DISPONIBILIDAD H2H (para todos los modelos)
    # ========================================================================
    if 'H2H_Available' in X_test.columns:
        print("\n" + "="*70)
        print("🤝 RENDIMIENTO POR DISPONIBILIDAD H2H")
        print("="*70 + "\n")
        
        mask_con_h2h = X_test['H2H_Available'] == 1
        mask_sin_h2h = X_test['H2H_Available'] == 0
        
        print(f"Partidos CON historial H2H: {mask_con_h2h.sum()}")
        print(f"Partidos SIN historial H2H: {mask_sin_h2h.sum()}")
        print()
        
        # Analizar cada modelo
        for nombre_modelo, datos_modelo in resultados.items():
            y_pred = datos_modelo['predicciones']
            
            print(f"📊 Modelo: {nombre_modelo}")
            print("-" * 50)
            
            if mask_con_h2h.sum() > 0:
                acc_con_h2h = accuracy_score(
                    y_test[mask_con_h2h], 
                    y_pred[mask_con_h2h]
                )
                print(f"   Accuracy CON H2H: {acc_con_h2h:.4f} ({acc_con_h2h*100:.2f}%)")
            
            if mask_sin_h2h.sum() > 0:
                acc_sin_h2h = accuracy_score(
                    y_test[mask_sin_h2h], 
                    y_pred[mask_sin_h2h]
                )
                print(f"   Accuracy SIN H2H: {acc_sin_h2h:.4f} ({acc_sin_h2h*100:.2f}%)")
            
            print()
    
    return resultados

# ============================================================================
# COMPARACIÓN Y SELECCIÓN DEL MEJOR
# ============================================================================

def seleccionar_mejor_modelo(resultados, y_test):
    """Compara modelos y selecciona el mejor según F1-Score."""
    
    print("\n" + "="*70)
    print("FASE 2: COMPARACIÓN DE MODELOS")
    print("="*70)
    
    print(f"{'Modelo':<35} {'Accuracy':<12} {'F1-Score':<12}")
    print("-" * 70)
    
    for key, datos in resultados.items():
        print(f"{datos['nombre']:<35} {datos['accuracy']:>10.2%}  {datos['f1_score']:>10.4f}")
    
    # Elegir mejor por F1-Score
    mejor_key = max(resultados.items(), key=lambda x: x[1]['f1_score'])[0]
    mejor = resultados[mejor_key]
    
    print("\n" + "🏆" * 35)
    print(f"GANADOR: {mejor['nombre']}")
    print(f"   Accuracy: {mejor['accuracy']:.2%}")
    print(f"   F1-Score: {mejor['f1_score']:.4f}")
    print("🏆" * 35)
    
    # Mostrar reporte detallado
    print(f"\n📊 REPORTE DETALLADO DEL MEJOR MODELO:")
    print("-" * 70)
    target_names = ['Local', 'Empate', 'Visitante']
    print(classification_report(y_test, mejor['predicciones'], target_names=target_names))
    
    return mejor_key, mejor

# ============================================================================
# OPTIMIZACIÓN DE HIPERPARÁMETROS
# ============================================================================

def optimizar_modelo(mejor_key, mejor_modelo, X_train, y_train, X_test, y_test):
    """Optimiza el mejor modelo encontrando los mejores hiperparámetros."""
    
    print("\n" + "="*70)
    print("FASE 3: OPTIMIZACIÓN DE HIPERPARÁMETROS")
    print("="*70)
    
    print(f"🔧 Optimizando: {mejor_modelo['nombre']}")
    print("   (Esto puede tardar 3-5 minutos)\n")
    
    # Definir grids de búsqueda según el modelo
    if mejor_key == 'RF_Basico':
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_leaf': [1, 2, 4, 5],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2', None]
        }
        base_model = RandomForestClassifier(random_state=SEED, n_jobs=1)
        X_opt, y_opt = X_train, y_train
        
    elif mejor_key == 'RF_Balanceado':
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_leaf': [1, 2, 4, 5],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2', None]
        }
        base_model = RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=1)
        X_opt, y_opt = X_train, y_train
    
    elif mejor_key == 'RF_Custom':
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_leaf': [1, 2, 4, 5],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2', None]
        }
        # Mantener los pesos personalizados FIJOS
        base_model = RandomForestClassifier(
            class_weight=PESOS_PERSONALIZADOS, 
            random_state=SEED, 
            n_jobs=1
        )
        X_opt, y_opt = X_train, y_train
        print(f"   ⚠️ Pesos FIJOS (no se optimizan): {PESOS_PERSONALIZADOS}")
        
    elif mejor_key == 'XGBoost_SMOTE' and ADVANCED_LIBS:
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
        }
        base_model = XGBClassifier(
            random_state=SEED,
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False,
            n_jobs=1
        )
        # Usar datos balanceados con SMOTE
        X_opt, y_opt = mejor_modelo['datos_balanceados']
    
    else:
        print(f"⚠️ No hay configuración de optimización para {mejor_key}")
        return mejor_modelo['modelo'], mejor_modelo['predicciones'], False
    
    # Búsqueda aleatoria de hiperparámetros
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=30,
        cv=3,
        verbose=2,
        random_state=SEED,
        n_jobs=1,  # n_jobs=1 para reproducibilidad
        scoring='f1_weighted'
    )
    
    random_search.fit(X_opt, y_opt)
    
    # Evaluar modelo optimizado
    modelo_optimizado = random_search.best_estimator_
    pred_optimizado = modelo_optimizado.predict(X_test)
    
    acc_opt = accuracy_score(y_test, pred_optimizado)
    f1_opt = f1_score(y_test, pred_optimizado, average='weighted')
    
    print("\n✅ OPTIMIZACIÓN COMPLETADA")
    print("="*70)
    print("\n📋 MEJORES PARÁMETROS ENCONTRADOS:")
    for param, value in random_search.best_params_.items():
        print(f"   {param}: {value}")
    
    print("\n📊 COMPARACIÓN ANTES/DESPUÉS:")
    print(f"{'Métrica':<20} {'Antes':<15} {'Después':<15} {'Mejora':<10}")
    print("-" * 70)
    
    mejora_acc = ((acc_opt - mejor_modelo['accuracy']) / mejor_modelo['accuracy']) * 100
    mejora_f1 = ((f1_opt - mejor_modelo['f1_score']) / mejor_modelo['f1_score']) * 100
    
    print(f"{'Accuracy':<20} {mejor_modelo['accuracy']:>13.2%} {acc_opt:>13.2%} {mejora_acc:>8.1f}%")
    print(f"{'F1-Score':<20} {mejor_modelo['f1_score']:>13.4f} {f1_opt:>13.4f} {mejora_f1:>8.1f}%")
    
    if f1_opt > mejor_modelo['f1_score']:
        print("\n🎉 ¡El modelo mejoró con la optimización!")
        return modelo_optimizado, pred_optimizado, True
    else:
        print("\n⚠️  La optimización no mejoró el modelo. Usando el original.")
        return mejor_modelo['modelo'], mejor_modelo['predicciones'], False

# ============================================================================
# VISUALIZACIONES
# ============================================================================

def visualizar_resultados(y_test, predictions, nombre_modelo, features, modelo):
    """Genera visualizaciones del modelo final."""
    
    print("\n" + "="*70)
    print("GENERANDO VISUALIZACIONES")
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
        sns.barplot(x='Importancia', y='Feature', data=df_imp.head(15), palette='viridis')
        plt.title(f'Top 15 Características Más Importantes - {nombre_modelo}', 
                fontsize=16, fontweight='bold')
        plt.xlabel('Importancia', fontsize=12)
        plt.ylabel('Característica', fontsize=12)
        plt.tight_layout()
        
        archivo_imp = os.path.join(RUTA_MODELOS, 'feature_importance_final.png')
        plt.savefig(archivo_imp, dpi=150, bbox_inches='tight')
        print(f"✅ Guardado: {archivo_imp}")
        plt.close()
        
        print("\n📋 TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES:")
        print(df_imp.head(10).to_string(index=False))


# ============================================================================
# GUARDAR MODELO FINAL
# ============================================================================

def guardar_modelo_final(modelo, features, nombre_modelo):
    """Guarda el modelo final optimizado."""
    
    print("\n" + "="*70)
    print("GUARDANDO MODELO FINAL")
    print("="*70)
    
    # Guardar modelo
    archivo_modelo = os.path.join(RUTA_MODELOS, 'modelo_final_optimizado.pkl')
    joblib.dump(modelo, archivo_modelo)
    print(f"✅ Modelo guardado: {archivo_modelo}")
    
    # Guardar features
    archivo_features = os.path.join(RUTA_MODELOS, 'features.pkl')
    joblib.dump(features, archivo_features)
    print(f"✅ Features guardadas: {archivo_features}")
    
    # Guardar metadata
    metadata = {
        'nombre_modelo': nombre_modelo,
        'n_features': len(features),
        'features': features
    }
    archivo_meta = os.path.join(RUTA_MODELOS, 'metadata.pkl')
    joblib.dump(metadata, archivo_meta)
    print(f"✅ Metadata guardada: {archivo_meta}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Pipeline de entrenamiento: 3 modelos base, selección del mejor."""
    
    print("\n" + "⚽" * 35)
    print("   PREMIER LEAGUE - ENTRENAMIENTO (3 MODELOS)")
    print("⚽" * 35 + "\n")
    
    # Cargar datos
    resultado = cargar_datos()
    if resultado[0] is None:
        return None, None
    
    X, y, features = resultado
    
    # Split temporal
    print("\n🔪 División de datos (80% entrenamiento, 20% prueba)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    print(f"   Entrenamiento: {len(X_train)} partidos")
    print(f"   Prueba: {len(X_test)} partidos")
    
    # Entrenar todos los modelos
    resultados = entrenar_todos_los_modelos(X_train, y_train, X_test, y_test)
    
    # Seleccionar el mejor
    mejor_key, mejor_modelo = seleccionar_mejor_modelo(resultados, y_test)
    
    # Usar el mejor modelo y optimizarlo
    modelo_final, pred_final, mejorado = optimizar_modelo(
        mejor_key, mejor_modelo, X_train, y_train, X_test, y_test
    )
    if mejorado:
        nombre_final = f"{mejor_modelo['nombre']} (Optimizado)"
    else:
        nombre_final = mejor_modelo['nombre']
    
    # Visualizaciones
    visualizar_resultados(y_test, pred_final, nombre_final, features, modelo_final)
    
    # Guardar modelo final
    guardar_modelo_final(modelo_final, features, nombre_final)
    
    # Resumen final
    print("\n" + "="*70)
    print("✅ ¡ENTRENAMIENTO COMPLETADO CON ÉXITO!")
    print("="*70)
    print(f"\n🏆 Modelo final: {nombre_final}")
    print(f"📊 Accuracy: {accuracy_score(y_test, pred_final):.2%}")
    print(f"📊 F1-Score: {f1_score(y_test, pred_final, average='weighted'):.4f}")
    print(f"\n📁 Archivos generados:")
    print(f"   • modelos/modelo_final_optimizado.pkl")
    print(f"   • modelos/features.pkl")
    print(f"   • modelos/confusion_matrix_final.png")
    print(f"   • modelos/feature_importance_final.png")
    print(f"\n➡️  Siguiente paso: python 03_predecir_partidos.py\n")
    
    return modelo_final, features


if __name__ == "__main__":
    modelo_final, features = main()