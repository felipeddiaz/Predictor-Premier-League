# -*- coding: utf-8 -*-
"""
Premier League Match Predictor
01 - PREPARACION DE DATOS

Carga temporadas brutas, limpia, aplica feature engineering en 4 fases
(rendimiento rolling, forma W/D/L, Head-to-Head, xG rolling) y guarda
el dataset canonico en datos/procesados/premier_league_con_features.csv.
"""

import pandas as pd
import numpy as np
import glob
import os
import warnings

from config import (
    RUTA_TEMPORADAS,
    RUTA_PROCESADOS,
    ARCHIVO_LIMPIO,
    ARCHIVO_FEATURES,
    ROLLING_WINDOW,
    MISSING_THRESHOLD,
    COLUMNAS_ESENCIALES,
)
from utils import calcular_h2h_features, agregar_xg_rolling

warnings.filterwarnings('ignore')

# Alias para compatibilidad interna
RUTA_DATOS = RUTA_TEMPORADAS
RUTA_GUARDADO = RUTA_PROCESADOS

# Crear carpeta de salida si no existe
os.makedirs(RUTA_GUARDADO, exist_ok=True)

# ============================================================================
# FASE 1: CARGA Y LIMPIEZA DE DATOS
# ============================================================================

def cargar_datos_premier_league(ruta_carpeta):
    """
    Carga y limpia los datos de múltiples temporadas de la Premier League.
    """
    print("="*60)
    print("FASE 1: CARGA Y LIMPIEZA DE DATOS")
    print("="*60)
    
    if not os.path.exists(ruta_carpeta):
        print(f"❌ ERROR: La carpeta '{ruta_carpeta}' no existe.")
        print(f"   Crea la carpeta y coloca tus archivos CSV ahí.")
        return None
    
    archivos = glob.glob(os.path.join(ruta_carpeta, "*.csv"))
    
    if len(archivos) == 0:
        print(f"❌ ERROR: No se encontraron archivos CSV en '{ruta_carpeta}'")
        return None
    
    print(f"📂 Se encontraron {len(archivos)} archivos CSV")
    
    lista_dfs = []
    for archivo in archivos:
        try:
            df_temp = pd.read_csv(archivo, encoding='latin1', on_bad_lines='skip')
            lista_dfs.append(df_temp)
            print(f"  ✓ {os.path.basename(archivo)}")
        except Exception as e:
            print(f"  ✗ Error en {os.path.basename(archivo)}: {e}")
    
    if not lista_dfs:
        print("❌ ERROR: Ningún archivo CSV se pudo cargar correctamente")
        return None

    df = pd.concat(lista_dfs, ignore_index=True)
    print(f"\n✅ Total de partidos: {len(df)}")
    
    # Limpiar fechas
    print("\n🔧 Limpiando fechas...")
    if 'Date' not in df.columns:
        print("❌ ERROR: No se encontró la columna 'Date'")
        return None
    
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', dayfirst=True, errors='coerce')
    
    fechas_invalidas = df['Date'].isnull().sum()
    if fechas_invalidas > 0:
        print(f"⚠️  Eliminadas {fechas_invalidas} filas con fechas inválidas")
        df.dropna(subset=['Date'], inplace=True)
    
    df.sort_values(by='Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Eliminar columnas con >MISSING_THRESHOLD datos faltantes
    print("\n🧹 Eliminando columnas inconsistentes...")
    missing_percentage = df.isnull().sum() / len(df)
    columns_to_drop = missing_percentage[missing_percentage > MISSING_THRESHOLD].index.tolist()
    
    if len(columns_to_drop) > 0:
        print(f"  Eliminadas {len(columns_to_drop)} columnas")
        df.drop(columns=columns_to_drop, inplace=True)
    
    # Eliminar filas con datos faltantes en columnas esenciales
    columnas_disponibles = [col for col in COLUMNAS_ESENCIALES if col in df.columns]
    
    filas_antes = len(df)
    df.dropna(subset=columnas_disponibles, inplace=True)
    filas_despues = len(df)
    
    if filas_antes - filas_despues > 0:
        print(f"  Eliminadas {filas_antes - filas_despues} filas con datos faltantes")
    
    # Crear columna objetivo
    if 'FTR' in df.columns:
        ftr_map = {'H': 0, 'D': 1, 'A': 2}
        df['FTR_numeric'] = df['FTR'].map(ftr_map)
        print("\n✅ Columna objetivo creada (0=Local, 1=Empate, 2=Visitante)")
    
    print(f"\n✅ Dataset limpio: {len(df)} partidos x {len(df.columns)} columnas")
    
    return df


# ============================================================================
# FASE 2: INGENIERÍA DE CARACTERÍSTICAS - RENDIMIENTO
# ============================================================================

def crear_features_rendimiento(df, window=ROLLING_WINDOW):
    """
    Crea promedios moviles de goles y tiros a puerta por equipo.

    Usa una asignacion vectorizada (merge) en lugar de iterrows para evitar
    el KeyError latente y reducir la complejidad de O(n^2) a O(n).
    El shift(1) garantiza que el partido actual no contamine su propia media.
    """
    print("\n" + "="*60)
    print("FASE 2: CARACTERÍSTICAS DE RENDIMIENTO")
    print("="*60)

    df_features = df.copy().sort_values('Date').reset_index(drop=True)
    teams = sorted(df_features['HomeTeam'].unique())
    print(f"📊 Calculando promedios moviles para {len(teams)} equipos...")

    # Inicializar con NaN
    for col in ['HT_AvgGoals', 'AT_AvgGoals', 'HT_AvgShotsTarget', 'AT_AvgShotsTarget']:
        df_features[col] = np.nan

    for team in teams:
        mask_home = df_features['HomeTeam'] == team
        mask_away = df_features['AwayTeam'] == team

        # Goles anotados como local
        df_features.loc[mask_home, 'HT_AvgGoals'] = (
            df_features.loc[mask_home, 'FTHG']
            .shift(1).rolling(window, min_periods=1).mean().values
        )
        # Goles anotados como visitante
        df_features.loc[mask_away, 'AT_AvgGoals'] = (
            df_features.loc[mask_away, 'FTAG']
            .shift(1).rolling(window, min_periods=1).mean().values
        )
        # Tiros a puerta como local
        df_features.loc[mask_home, 'HT_AvgShotsTarget'] = (
            df_features.loc[mask_home, 'HST']
            .shift(1).rolling(window, min_periods=1).mean().values
        )
        # Tiros a puerta como visitante
        df_features.loc[mask_away, 'AT_AvgShotsTarget'] = (
            df_features.loc[mask_away, 'AST']
            .shift(1).rolling(window, min_periods=1).mean().values
        )

    feature_cols = ['HT_AvgGoals', 'AT_AvgGoals', 'HT_AvgShotsTarget', 'AT_AvgShotsTarget']
    df_features[feature_cols] = df_features[feature_cols].fillna(0)

    print("✅ Características de rendimiento creadas")
    return df_features


# ============================================================================
# FASE 3: INGENIERÍA DE CARACTERÍSTICAS - RESULTADOS
# ============================================================================

def crear_features_resultados(df, window=ROLLING_WINDOW):
    """
    Crea características basadas en resultados recientes (W/D/L).
    """
    print("\n" + "="*60)
    print("FASE 3: CARACTERÍSTICAS DE RESULTADOS")
    print("="*60)
    
    df_result = df.copy()
    
    home_games = df[['Date', 'HomeTeam']].rename(columns={'HomeTeam': 'Team'})
    home_games['Result'] = df['FTR'].map({'H': 'W', 'D': 'D', 'A': 'L'})
    
    away_games = df[['Date', 'AwayTeam']].rename(columns={'AwayTeam': 'Team'})
    away_games['Result'] = df['FTR'].map({'H': 'L', 'D': 'D', 'A': 'W'})
    
    all_games = pd.concat([home_games, away_games]).sort_values(['Team', 'Date'])
    
    result_dummies = pd.get_dummies(all_games['Result'])
    all_games = pd.concat([all_games, result_dummies], axis=1)
    
    for col in ['W', 'D', 'L']:
        all_games[f'Form_{col}'] = all_games.groupby('Team')[col].transform(
            lambda x: x.rolling(window=window, min_periods=1).sum().shift(1)
        ).fillna(0)
    
    form_cols = ['Date', 'Team'] + [f'Form_{r}' for r in ['W', 'D', 'L']]
    team_form_lookup = all_games[form_cols]
    
    df_result = df_result.merge(
        team_form_lookup, 
        left_on=['Date', 'HomeTeam'], 
        right_on=['Date', 'Team'], 
        how='left'
    ).drop(columns='Team')
    
    df_result = df_result.merge(
        team_form_lookup, 
        left_on=['Date', 'AwayTeam'], 
        right_on=['Date', 'Team'], 
        how='left', 
        suffixes=('_HT', '_AT')
    ).drop(columns='Team')
    
    rename_dict = {
        'Form_W_HT': 'HT_Form_W', 'Form_D_HT': 'HT_Form_D', 'Form_L_HT': 'HT_Form_L',
        'Form_W_AT': 'AT_Form_W', 'Form_D_AT': 'AT_Form_D', 'Form_L_AT': 'AT_Form_L'
    }
    df_result.rename(columns=rename_dict, inplace=True)
    
    form_columns = [col for col in df_result.columns if 'Form_' in col]
    df_result[form_columns] = df_result[form_columns].fillna(0)
    
    print("✅ Características de resultados creadas")
    
    return df_result


# ============================================================================
# FASE 4: INGENIERÍA DE CARACTERÍSTICAS - HEAD-TO-HEAD
# ============================================================================
# calcular_h2h_features y agregar_xg_rolling viven en utils.py (version
# canonica). Aqui solo se define el orquestador de la fase H2H.

def crear_features_h2h(df):
    """
    Agrega features H2H a todo el dataset.

    Usa calcular_h2h_features de utils.py como implementacion canonica.
    Las features derivadas se zerean correctamente para partidos sin historial.

    Args:
        df: DataFrame con partidos (Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR).

    Returns:
        DataFrame con features H2H y derivadas agregadas.
    """
    print("\n" + "="*60)
    print("FASE 4: CARACTERÍSTICAS HEAD-TO-HEAD")
    print("="*60)

    df['Date'] = pd.to_datetime(df['Date'])

    h2h_base_cols = [
        'H2H_Available', 'H2H_Matches', 'H2H_Home_Wins', 'H2H_Draws',
        'H2H_Away_Wins', 'H2H_Home_Goals_Avg', 'H2H_Away_Goals_Avg',
        'H2H_Home_Win_Rate', 'H2H_BTTS_Rate',
    ]
    for col in h2h_base_cols:
        df[col] = 0.0

    total = len(df)
    print(f"\n📊 Calculando H2H para {total} partidos...")

    for idx, row in df.iterrows():
        if idx % 200 == 0:
            print(f"  Progreso: {idx}/{total} ({idx/total*100:.1f}%)", end='\r')

        h2h_feats = calcular_h2h_features(
            df=df,
            equipo_local=row['HomeTeam'],
            equipo_visitante=row['AwayTeam'],
            fecha_limite=row['Date'],
        )
        for key, value in h2h_feats.items():
            df.at[idx, key] = value

    print(f"\n✅ Características H2H creadas")

    # Estadisticas
    con_h2h = (df['H2H_Matches'] > 0).sum()
    print(f"\n📊 Estadísticas H2H:")
    print(f"   Partidos con historial: {con_h2h} ({con_h2h/len(df)*100:.1f}%)")
    print(f"   Partidos sin historial: {len(df) - con_h2h}")
    print(f"   Promedio enfrentamientos: {df['H2H_Matches'].mean():.2f}")

    # Features derivadas
    print("\n🔧 Creando features derivadas de H2H...")
    df['H2H_Goal_Diff'] = df['H2H_Home_Goals_Avg'] - df['H2H_Away_Goals_Avg']
    df['H2H_Total_Goals_Avg'] = df['H2H_Home_Goals_Avg'] + df['H2H_Away_Goals_Avg']
    df['H2H_Home_Consistent'] = (
        (df['H2H_Home_Wins'] + df['H2H_Draws']) / df['H2H_Matches'].replace(0, 1)
    )
    df['H2H_Win_Advantage'] = df['H2H_Home_Wins'] - df['H2H_Away_Wins']

    # Zerear TODAS las derivadas para filas sin historial
    mask_sin_h2h = df['H2H_Available'] == 0
    derivadas = ['H2H_Goal_Diff', 'H2H_Win_Advantage', 'H2H_Total_Goals_Avg', 'H2H_Home_Consistent']
    df.loc[mask_sin_h2h, derivadas] = 0

    # Rellenar cualquier NaN residual
    h2h_all_cols = [col for col in df.columns if 'H2H' in col]
    df[h2h_all_cols] = df[h2h_all_cols].fillna(0)

    print("✅ Creadas: H2H_Goal_Diff, H2H_Total_Goals_Avg, H2H_Home_Consistent, H2H_Win_Advantage")
    return df

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Ejecuta todo el pipeline de preparación de datos.
    """
    print("\n" + "⚽" * 30)
    print("   PREMIER LEAGUE - PREPARACIÓN DE DATOS")
    print("⚽" * 30 + "\n")
    
    # Paso 1: Cargar y limpiar
    df_limpio = cargar_datos_premier_league(RUTA_DATOS)
    
    if df_limpio is None:
        print("\n❌ No se pudo continuar. Verifica la ruta de los datos.")
        return None
    
    # Guardar datos limpios
    os.makedirs(RUTA_GUARDADO, exist_ok=True)
    df_limpio.to_csv(ARCHIVO_LIMPIO, index=False)
    print(f"\n💾 Guardado: {ARCHIVO_LIMPIO}")

    # Paso 2: Features de rendimiento
    df_con_rendimiento = crear_features_rendimiento(df_limpio)

    # Paso 3: Features de resultados
    df_con_resultados = crear_features_resultados(df_con_rendimiento)

    # Paso 4: H2H
    try:
        df_final_h2h = crear_features_h2h(df_con_resultados.copy())

        # Paso 5 (opcional): xG rolling — importado de utils
        df_final_h2h = agregar_xg_rolling(df_final_h2h)

        # Backup del archivo anterior si existe
        if os.path.exists(ARCHIVO_FEATURES):
            backup_path = ARCHIVO_FEATURES + '.backup'
            try:
                os.replace(ARCHIVO_FEATURES, backup_path)
                print(f"\n📁 Backup creado: {backup_path}")
            except OSError:
                pass

        df_final_h2h.to_csv(ARCHIVO_FEATURES, index=False)
        print(f"\n💾 Guardado (canonico, CON H2H): {ARCHIVO_FEATURES}")
        df_final = df_final_h2h

    except Exception as e:
        print(f"\n⚠️  No se pudo generar H2H: {e}")
        print("No se ha guardado ningun archivo nuevo.\n")
        df_final = df_con_resultados
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    print(f"📊 Partidos: {len(df_final)}")
    print(f"📊 Características totales: {len(df_final.columns)}")
    print(f"📅 Desde: {df_final['Date'].min().strftime('%d/%m/%Y')}")
    print(f"📅 Hasta: {df_final['Date'].max().strftime('%d/%m/%Y')}")
    
    # Mostrar columnas nuevas
    print("\n📋 Columnas H2H agregadas:")
    h2h_cols = [col for col in df_final.columns if 'H2H' in col]
    for col in h2h_cols:
        print(f"   • {col}")
    
    print("\n✅ ¡PREPARACIÓN COMPLETADA!")
    print("   Siguiente paso: python 02_entrenar_modelo.py\n")
    
    return df_final

if __name__ == "__main__":
    df_final = main()