# -*- coding: utf-8 -*-
"""
Premier League Match Predictor
01 - PREPARACIÓN DE DATOS (Con H2H correctamente integrado)
"""

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN - AJUSTA ESTAS RUTAS SEGÚN TU PROYECTO
# ============================================================================

RUTA_DATOS = RUTA_DATOS = './datos/temporadas/'
RUTA_GUARDADO = './datos/procesados/'

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
    
    # Eliminar columnas con >20% datos faltantes
    print("\n🧹 Eliminando columnas inconsistentes...")
    missing_percentage = df.isnull().sum() / len(df)
    columns_to_drop = missing_percentage[missing_percentage > 0.20].index.tolist()
    
    if len(columns_to_drop) > 0:
        print(f"  Eliminadas {len(columns_to_drop)} columnas")
        df.drop(columns=columns_to_drop, inplace=True)
    
    # Eliminar filas con datos faltantes en columnas esenciales
    columnas_esenciales = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST']
    columnas_disponibles = [col for col in columnas_esenciales if col in df.columns]
    
    filas_antes = len(df)
    df.dropna(subset=columnas_disponibles, inplace=True)
    filas_despues = len(df)
    
    if filas_antes - filas_despues > 0:
        print(f"  Eliminadas {filas_antes - filas_despues} filas con datos faltantes")
    
    # Crear columna objetivo
    if 'FTR' in df.columns:
        df['FTR_numeric'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
        print("\n✅ Columna objetivo creada (0=Local, 1=Empate, 2=Visitante)")
    
    print(f"\n✅ Dataset limpio: {len(df)} partidos x {len(df.columns)} columnas")
    
    return df


# ============================================================================
# FASE 2: INGENIERÍA DE CARACTERÍSTICAS - RENDIMIENTO
# ============================================================================

def crear_features_rendimiento(df):
    """
    Crea características de rendimiento basadas en promedios móviles.
    """
    print("\n" + "="*60)
    print("FASE 2: CARACTERÍSTICAS DE RENDIMIENTO")
    print("="*60)
    
    df_features = df.copy()
    team_stats = {}
    teams = df_features['HomeTeam'].unique()
    
    print(f"📊 Calculando promedios móviles para {len(teams)} equipos...")
    
    for team in teams:
        team_matches = df_features[
            (df_features['HomeTeam'] == team) | 
            (df_features['AwayTeam'] == team)
        ].copy()
        
        team_matches['GoalsScored'] = team_matches.apply(
            lambda row: row['FTHG'] if row['HomeTeam'] == team else row['FTAG'], 
            axis=1
        )
        
        team_matches['ShotsOnTarget'] = team_matches.apply(
            lambda row: row['HST'] if row['HomeTeam'] == team else row['AST'], 
            axis=1
        )
        
        team_matches['AvgGoals_rolling'] = team_matches['GoalsScored'].rolling(
            window=5, min_periods=1
        ).mean().shift(1)
        
        team_matches['AvgShotsOnTarget_rolling'] = team_matches['ShotsOnTarget'].rolling(
            window=5, min_periods=1
        ).mean().shift(1)
        
        team_stats[team] = team_matches
    
    df_features['HT_AvgGoals'] = None
    df_features['AT_AvgGoals'] = None
    df_features['HT_AvgShotsTarget'] = None
    df_features['AT_AvgShotsTarget'] = None
    
    for index, row in df_features.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        df_features.at[index, 'HT_AvgGoals'] = team_stats[home_team].loc[index, 'AvgGoals_rolling']
        df_features.at[index, 'HT_AvgShotsTarget'] = team_stats[home_team].loc[index, 'AvgShotsOnTarget_rolling']
        df_features.at[index, 'AT_AvgGoals'] = team_stats[away_team].loc[index, 'AvgGoals_rolling']
        df_features.at[index, 'AT_AvgShotsTarget'] = team_stats[away_team].loc[index, 'AvgShotsOnTarget_rolling']
    
    feature_cols = ['HT_AvgGoals', 'AT_AvgGoals', 'HT_AvgShotsTarget', 'AT_AvgShotsTarget']
    df_features[feature_cols] = df_features[feature_cols].fillna(0)
    
    print("✅ Características de rendimiento creadas")
    
    return df_features


# ============================================================================
# FASE 3: INGENIERÍA DE CARACTERÍSTICAS - RESULTADOS
# ============================================================================

def crear_features_resultados(df, window=5):
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

def calcular_h2h_features(df, equipo_local, equipo_visitante, fecha_limite=None, ultimos_n=5):
    """
    Calcula features de Head-to-Head CON flag de disponibilidad.
    
    NUEVA LÓGICA:
    - Si hay historial → H2H_Available = 1, features reales
    - Si NO hay historial → H2H_Available = 0, features neutras
    """
    
    # Filtrar enfrentamientos
    mask = (
        ((df['HomeTeam'] == equipo_local) & (df['AwayTeam'] == equipo_visitante)) |
        ((df['HomeTeam'] == equipo_visitante) & (df['AwayTeam'] == equipo_local))
    )
    if fecha_limite is not None:
        mask = mask & (pd.to_datetime(df['Date']) < pd.to_datetime(fecha_limite))

    h2h = df[mask].sort_values('Date', ascending=False).head(ultimos_n)
    
    # Si NO hay historial → retornar NaN (no valores neutros)
    if len(h2h) == 0:
        return {
            'H2H_Matches': 0,
            'H2H_Home_Wins': np.nan,      # ← NaN en lugar de 0
            'H2H_Draws': np.nan,
            'H2H_Away_Wins': np.nan,
            'H2H_Home_Goals_Avg': np.nan,  # ← NaN en lugar de 1.5
            'H2H_Away_Goals_Avg': np.nan,
            'H2H_Home_Win_Rate': np.nan,   # ← NaN en lugar de 0.33
            'H2H_BTTS_Rate': np.nan
        }
    
    # Si SÍ hay historial → calcular features reales
    resultados_local = []
    goles_local = []
    goles_visitante = []
    btts_count = 0
    
    for _, partido in h2h.iterrows():
        if partido['HomeTeam'] == equipo_local:
            goles_local.append(partido['FTHG'])
            goles_visitante.append(partido['FTAG'])
            
            if partido['FTR'] == 'H':
                resultados_local.append('W')
            elif partido['FTR'] == 'D':
                resultados_local.append('D')
            else:
                resultados_local.append('L')
        else:
            goles_local.append(partido['FTAG'])
            goles_visitante.append(partido['FTHG'])
            
            if partido['FTR'] == 'A':
                resultados_local.append('W')
            elif partido['FTR'] == 'D':
                resultados_local.append('D')
            else:
                resultados_local.append('L')
        
        if partido['FTHG'] > 0 and partido['FTAG'] > 0:
            btts_count += 1
    
    wins = resultados_local.count('W')
    draws = resultados_local.count('D')
    losses = resultados_local.count('L')
    total = len(resultados_local)
    
    return {
        'H2H_Available': 1,  # ← NUEVO: HAY HISTORIAL
        'H2H_Matches': total,
        'H2H_Home_Wins': wins,
        'H2H_Draws': draws,
        'H2H_Away_Wins': losses,
        'H2H_Home_Goals_Avg': np.mean(goles_local) if goles_local else 0,
        'H2H_Away_Goals_Avg': np.mean(goles_visitante) if goles_visitante else 0,
        'H2H_Home_Win_Rate': wins / total if total > 0 else 0.33,
        'H2H_BTTS_Rate': btts_count / total if total > 0 else 0.5
    }


def crear_features_h2h(df):
    """
    Agrega features H2H a todo el dataset.
    ESTA ES LA VERSIÓN OPTIMIZADA QUE SE INTEGRA EN EL PIPELINE PRINCIPAL
    
    Args:
        df: DataFrame con partidos (debe tener Date, HomeTeam, AwayTeam, etc.)
        
    Returns:
        DataFrame con features H2H agregadas
    """
    
    print("\n" + "="*60)
    print("FASE 4: CARACTERÍSTICAS HEAD-TO-HEAD")
    print("="*60)
    
    # Asegurar que Date es datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Inicializar columnas H2H
    h2h_cols = [
        'H2H_Available',  # ← NUEVO
        'H2H_Matches', 'H2H_Home_Wins', 'H2H_Draws', 'H2H_Away_Wins',
        'H2H_Home_Goals_Avg', 'H2H_Away_Goals_Avg', 'H2H_Home_Win_Rate',
        'H2H_BTTS_Rate'
    ]
    
    for col in h2h_cols:
        df[col] = 0.0
    
    # Calcular H2H para cada partido
    total = len(df)
    print(f"\n📊 Calculando H2H para {total} partidos...")
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  Progreso: {idx}/{total} ({idx/total*100:.1f}%)", end='\r')
        
        # Calcular H2H hasta este partido (sin incluirlo)
        h2h_features = calcular_h2h_features(
            df=df,
            equipo_local=row['HomeTeam'],
            equipo_visitante=row['AwayTeam'],
            fecha_limite=row['Date'],
            ultimos_n=5
        )
        
        # Asignar features
        for key, value in h2h_features.items():
            df.at[idx, key] = value
    
    print(f"\n✅ Características H2H creadas")
    
    # Estadísticas
    print("\n📊 Estadísticas H2H:")
    print(f"   Partidos con historial: {(df['H2H_Matches'] > 0).sum()} ({(df['H2H_Matches'] > 0).sum()/len(df)*100:.1f}%)")
    print(f"   Partidos sin historial: {(df['H2H_Matches'] == 0).sum()}")
    print(f"   Promedio enfrentamientos: {df['H2H_Matches'].mean():.2f}")
    if (df['H2H_Matches'] > 0).sum() > 0:
        print(f"   Win rate local promedio: {df[df['H2H_Matches']>0]['H2H_Home_Win_Rate'].mean():.1%}")
    
    # ================================================================
    print("\n🔧 Creando features derivadas de H2H...")
    
    # Features derivadas H2H
    df['H2H_Goal_Diff'] = df['H2H_Home_Goals_Avg'] - df['H2H_Away_Goals_Avg']
    df['H2H_Total_Goals_Avg'] = df['H2H_Home_Goals_Avg'] + df['H2H_Away_Goals_Avg']
    df['H2H_Home_Consistent'] = ((df['H2H_Home_Wins'] + df['H2H_Draws']) / 
                                df['H2H_Matches'].replace(0, 1))
    
    df['H2H_Win_Advantage'] = df['H2H_Home_Wins'] - df['H2H_Away_Wins']

    # Cuando H2H_Available = 0, poner derivadas en 0
    df.loc[df['H2H_Available'] == 0, ['H2H_Goal_Diff', 'H2H_Win_Advantage']] = 0
    
    print("✅ Creadas:")
    print("   • H2H_Goal_Diff")
    print("   • H2H_Total_Goals_Avg")
    print("   • H2H_Home_Consistent")
    print("   • H2H_Win_Advantage")
    
    # Al final de crear_features_h2h()
    h2h_cols = [col for col in df.columns if 'H2H' in col and col != 'H2H_Matches']
    df[h2h_cols] = df[h2h_cols].fillna(0)  # Rellenar NaN con 0

    return df

# En tu 01_preparar_datos.py, agregar función:

def crear_features_xg_rolling(df, window=5):
    """
    Crea promedios móviles de xG para cada equipo.
    """
    print("\n" + "="*60)
    print("CARACTERÍSTICAS xG ROLLING")
    print("="*60)
    
    # Verificar que existan columnas xG
    if 'Home_xG' not in df.columns:
        print("⚠️  No hay datos de xG disponibles")
        return df
    
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Inicializar columnas
    df['HT_xG_Avg'] = np.nan
    df['AT_xG_Avg'] = np.nan
    df['HT_xGA_Avg'] = np.nan
    df['AT_xGA_Avg'] = np.nan
    
    teams = df['HomeTeam'].unique()
    
    for team in teams:
        # Partidos como local
        mask_home = df['HomeTeam'] == team
        # Partidos como visitante
        mask_away = df['AwayTeam'] == team
        
        # xG creado por el equipo (rolling)
        team_xg_home = df.loc[mask_home, 'Home_xG'].shift(1).rolling(window, min_periods=1).mean()
        team_xg_away = df.loc[mask_away, 'Away_xG'].shift(1).rolling(window, min_periods=1).mean()
        
        # xG concedido por el equipo (rolling)
        team_xga_home = df.loc[mask_home, 'Away_xG'].shift(1).rolling(window, min_periods=1).mean()
        team_xga_away = df.loc[mask_away, 'Home_xG'].shift(1).rolling(window, min_periods=1).mean()
        
        # Asignar cuando juega como LOCAL
        df.loc[mask_home, 'HT_xG_Avg'] = team_xg_home.values
        df.loc[mask_home, 'HT_xGA_Avg'] = team_xga_home.values
        
        # Asignar cuando juega como VISITANTE
        df.loc[mask_away, 'AT_xG_Avg'] = team_xg_away.values
        df.loc[mask_away, 'AT_xGA_Avg'] = team_xga_away.values
    
    # Features derivadas
    df['xG_Diff'] = df['HT_xG_Avg'] - df['AT_xG_Avg']
    df['xG_Total'] = df['HT_xG_Avg'] + df['AT_xG_Avg']
    
    # Rellenar NaN
    xg_cols = ['HT_xG_Avg', 'AT_xG_Avg', 'HT_xGA_Avg', 'AT_xGA_Avg', 'xG_Diff', 'xG_Total']
    df[xg_cols] = df[xg_cols].fillna(0)
    
    print(f"✅ Features xG rolling creadas")
    print(f"   Partidos con xG histórico: {(df['HT_xG_Avg'] > 0).sum()}")
    
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
    archivo_limpio = os.path.join(RUTA_GUARDADO, 'premier_league_limpio.csv')
    df_limpio.to_csv(archivo_limpio, index=False)
    print(f"\n💾 Guardado: {archivo_limpio}")
    
    # Paso 2: Features de rendimiento
    df_con_rendimiento = crear_features_rendimiento(df_limpio)
    
    # Paso 3: Features de resultados
    df_con_resultados = crear_features_resultados(df_con_rendimiento)
    
    # Paso 4: (opcional) H2H - por defecto no se modifica df_final aquí
    df_final = df_con_resultados

    try:
        df_final_h2h = crear_features_h2h(df_final.copy())
        archivo_final = os.path.join(RUTA_GUARDADO, 'premier_league_con_features.csv')

        # Si existe un archivo previo, mover a backup (simple .backup)
        if os.path.exists(archivo_final):
            backup_path = archivo_final + '.backup'
            try:
                os.replace(archivo_final, backup_path)
                print(f"\n📁 Backup creado: {backup_path}")
            except Exception:
                # si el replace falla, continuamos e intentamos sobreescribir
                pass

        # Guardar la versión final (con H2H) usando el nombre canónico
        df_final_h2h.to_csv(archivo_final, index=False)
        print(f"\n💾 Guardado (canónico, CON H2H): {archivo_final}")

        # Usar la versión con H2H para posteriores pasos/summary
        df_final = df_final_h2h

    except Exception as e:
            # Si la generación H2H falla, NO guardamos una versión sin H2H
            # para evitar crear el archivo no deseado. Informamos al usuario.
        print(f"\n⚠️ No se pudo generar H2H: {e}")
        print("No se ha guardado ningun 'premier_league_con_features.csv' nuevo.\n")
    
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