# -*- coding: utf-8 -*-
"""
utils.py — Funciones de feature engineering compartidas.

Contiene las implementaciones canónicas de todas las funciones que antes
estaban duplicadas en 5+ archivos. Cualquier corrección se hace aquí y
se propaga automáticamente.
"""

import pandas as pd
import numpy as np
import os
import warnings

from config import ROLLING_WINDOW, H2H_ULTIMOS_N, ARCHIVO_XG_RAW

warnings.filterwarnings('ignore')


# ============================================================================
# XG MERGE
# ============================================================================

# Mapeo de nombres → nombre canónico usado en el dataset principal
# Cubre tanto la columna 'team' (nombres largos) como 'opponent' (nombres cortos alternativos)
_XG_TEAM_MAP = {
    # Columna 'team' (nombres largos fbref)
    'Brighton And Hove Albion': 'Brighton',
    'Ipswich Town':             'Ipswich',
    'Leeds United':             'Leeds',
    'Leicester City':           'Leicester',
    'Luton Town':               'Luton',
    'Manchester City':          'Man City',
    'Manchester United':        'Man United',
    'Newcastle United':         'Newcastle',
    'Norwich City':             'Norwich',
    'Nottingham Forest':        "Nott'm Forest",
    'Tottenham Hotspur':        'Tottenham',
    'West Bromwich Albion':     'West Brom',
    'West Ham United':          'West Ham',
    'Wolverhampton Wanderers':  'Wolves',
    # Columna 'opponent' (nombres cortos alternativos fbref)
    'Manchester Utd':           'Man United',
    'Newcastle Utd':            'Newcastle',
    "Nott'ham Forest":          "Nott'm Forest",
    'Sheffield Utd':            'Sheffield United',
    'Ipswich Town':             'Ipswich',
    'Leeds United':             'Leeds',
    'Leicester City':           'Leicester',
    'Luton Town':               'Luton',
    'Norwich City':             'Norwich',
}


def merge_xg_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lee final_matches_xg.csv y agrega columnas Home_xG / Away_xG al DataFrame.

    El CSV tiene una fila por equipo por partido (venue = Home/Away).
    Esta función pivota ese formato a una fila por partido y hace un merge
    por fecha + equipos, normalizando nombres y formatos de fecha.

    Si el archivo no existe o el merge falla, devuelve df sin cambios
    (agregar_xg_rolling luego emitirá el warning de "no disponibles").

    Args:
        df: DataFrame con columnas Date (datetime o str), HomeTeam, AwayTeam.

    Returns:
        DataFrame con columnas Home_xG y Away_xG agregadas donde haya match.
    """
    if not os.path.exists(ARCHIVO_XG_RAW):
        print(f"   ⚠️  No se encontró {ARCHIVO_XG_RAW} — xG no disponible")
        return df

    print("\n🔧 Cargando datos de xG...")
    xg = pd.read_csv(ARCHIVO_XG_RAW)

    # Normalizar nombres de equipos en el CSV de xG (tanto team como opponent)
    xg['team']     = xg['team'].replace(_XG_TEAM_MAP)
    xg['opponent'] = xg['opponent'].replace(_XG_TEAM_MAP)

    # Normalizar fecha del CSV de xG (formato DD/MM/YYYY) → datetime
    xg['date'] = pd.to_datetime(xg['date'], dayfirst=True, errors='coerce')
    xg = xg.dropna(subset=['date', 'team', 'venue', 'xg'])

    # Separar filas de local y visitante
    # venue=Home → team=local, opponent=visitante → Home_xG
    home_xg = xg[xg['venue'] == 'Home'][['date', 'team', 'opponent', 'xg']].copy()
    home_xg.columns = ['Date', 'HomeTeam', 'AwayTeam', 'Home_xG']

    # venue=Away → team=visitante, opponent=local → Away_xG
    # Reordenamos opponent/team para que el merge sea por HomeTeam+AwayTeam
    away_xg = xg[xg['venue'] == 'Away'][['date', 'opponent', 'team', 'xg']].copy()
    away_xg.columns = ['Date', 'HomeTeam', 'AwayTeam', 'Away_xG']

    # Normalizar fecha del DataFrame principal (ya viene en formato YYYY-MM-DD)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Merge: primero agrega Home_xG, luego Away_xG
    df = df.merge(home_xg, on=['Date', 'HomeTeam', 'AwayTeam'], how='left')
    df = df.merge(away_xg, on=['Date', 'HomeTeam', 'AwayTeam'], how='left')

    con_xg = df['Home_xG'].notna().sum()
    total = len(df)
    print(f"   ✅ xG mergeado: {con_xg}/{total} partidos con datos ({con_xg/total*100:.1f}%)")

    return df


# ============================================================================
# XG ROLLING
# ============================================================================

def agregar_xg_rolling(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Agrega promedios móviles de xG por equipo.

    El shift(1) garantiza que el partido actual no se incluya en su propia
    ventana (sin data leakage). Los equipos se ordenan alfabéticamente para
    asegurar determinismo entre ejecuciones.

    Args:
        df: DataFrame con columnas Home_xG y Away_xG.
        window: Tamaño de la ventana rolling.

    Returns:
        DataFrame con columnas HT_xG_Avg, AT_xG_Avg, HT_xGA_Avg, AT_xGA_Avg,
        xG_Diff y xG_Total agregadas.
    """
    if 'Home_xG' not in df.columns or 'Away_xG' not in df.columns:
        print("   ⚠️  No hay datos de xG disponibles")
        return df

    print("\n🔧 Calculando xG rolling...")

    # Ordenar de forma deterministica
    df = df.sort_values(['Date', 'HomeTeam', 'AwayTeam']).reset_index(drop=True)

    df['HT_xG_Avg'] = np.nan
    df['AT_xG_Avg'] = np.nan
    df['HT_xGA_Avg'] = np.nan
    df['AT_xGA_Avg'] = np.nan

    for team in sorted(df['HomeTeam'].unique()):
        mask_home = df['HomeTeam'] == team
        mask_away = df['AwayTeam'] == team

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

    df['xG_Diff'] = df['HT_xG_Avg'] - df['AT_xG_Avg']
    df['xG_Total'] = df['HT_xG_Avg'] + df['AT_xG_Avg']

    xg_cols = ['HT_xG_Avg', 'AT_xG_Avg', 'HT_xGA_Avg', 'AT_xGA_Avg', 'xG_Diff', 'xG_Total']
    df[xg_cols] = df[xg_cols].fillna(0)

    con_xg = (df['HT_xG_Avg'] > 0).sum()
    print(f"   ✅ xG rolling agregado: {con_xg} partidos con datos históricos")

    return df


# ============================================================================
# FEATURES DE POSICIÓN EN TABLA
# ============================================================================

def agregar_features_tabla(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la posición en tabla antes de cada partido (sin leakage).

    El criterio de desempate es el estándar de la Premier League:
    Puntos → Diferencia de goles → Goles a favor → Goles en contra (asc).

    Procesa cada temporada de forma independiente. Los equipos y fechas se
    ordenan de forma deterministica para reproducibilidad.

    Args:
        df: DataFrame con columnas Date, HomeTeam, AwayTeam, FTHG, FTAG.

    Returns:
        DataFrame con columnas de posición de tabla agregadas.
    """
    print("\n🔧 Calculando features de posición en tabla...")

    df = df.sort_values(['Date', 'HomeTeam', 'AwayTeam']).reset_index(drop=True)

    tabla_cols = [
        'HT_Position', 'AT_Position', 'HT_Points', 'AT_Points', 'Position_Diff',
        'HT_Points_Above', 'HT_Points_Below', 'AT_Points_Above', 'AT_Points_Below',
        'Matchday', 'Season_Progress', 'Position_Reliability',
    ]
    for col in tabla_cols:
        df[col] = np.nan

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df['Season'] = df['Date'].apply(
        lambda x: f"{x.year}-{x.year+1}" if pd.notna(x) and x.month >= 8
        else (f"{x.year-1}-{x.year}" if pd.notna(x) else None)
    )
    temporadas = sorted(df['Season'].dropna().unique())
    print(f"   Procesando {len(temporadas)} temporadas...")

    for temporada in temporadas:
        mask_temporada = df['Season'] == temporada
        indices_temporada = df[mask_temporada].index.tolist()
        if not indices_temporada:
            continue

        puntos_equipo: dict = {}
        goles_favor: dict = {}
        goles_contra: dict = {}
        partidos_jugados: dict = {}

        equipos_temporada = sorted(
            set(df.loc[mask_temporada, 'HomeTeam'].tolist())
            | set(df.loc[mask_temporada, 'AwayTeam'].tolist())
        )
        for equipo in equipos_temporada:
            puntos_equipo[equipo] = 0
            goles_favor[equipo] = 0
            goles_contra[equipo] = 0
            partidos_jugados[equipo] = 0

        fechas_unicas = sorted(df.loc[mask_temporada, 'Date'].unique())
        fecha_to_jornada = {fecha: i + 1 for i, fecha in enumerate(fechas_unicas)}
        total_jornadas = len(fechas_unicas)

        for idx in indices_temporada:
            row = df.loc[idx]
            home = row['HomeTeam']
            away = row['AwayTeam']
            fecha = row['Date']

            jornada = fecha_to_jornada.get(fecha, 1)

            # ANTES del partido: asignar posición actual
            if partidos_jugados.get(home, 0) > 0 or partidos_jugados.get(away, 0) > 0:
                tabla = [
                    {
                        'Equipo': eq,
                        'Puntos': puntos_equipo[eq],
                        'GD': goles_favor[eq] - goles_contra[eq],
                        'GF': goles_favor[eq],
                        'GC': goles_contra[eq],
                        'PJ': partidos_jugados[eq],
                    }
                    for eq in equipos_temporada
                    if partidos_jugados[eq] > 0
                ]

                if tabla:
                    tabla_df = (
                        pd.DataFrame(tabla)
                        .sort_values(
                            by=['Puntos', 'GD', 'GF', 'GC'],
                            ascending=[False, False, False, True],
                        )
                        .reset_index(drop=True)
                    )
                    tabla_df['Posicion'] = range(1, len(tabla_df) + 1)

                    for equipo, col_pos, col_pts, col_above, col_below in [
                        (home, 'HT_Position', 'HT_Points', 'HT_Points_Below', 'HT_Points_Above'),
                        (away, 'AT_Position', 'AT_Points', 'AT_Points_Below', 'AT_Points_Above'),
                    ]:
                        pos_vals = tabla_df[tabla_df['Equipo'] == equipo]['Posicion'].values
                        if len(pos_vals) > 0:
                            pos = int(pos_vals[0])
                            pts = puntos_equipo[equipo]
                            df.at[idx, col_pos] = pos
                            df.at[idx, col_pts] = pts

                            above = tabla_df[tabla_df['Posicion'] == pos - 1]['Puntos'].values
                            if len(above) > 0:
                                df.at[idx, col_above] = above[0] - pts

                            below = tabla_df[tabla_df['Posicion'] == pos + 1]['Puntos'].values
                            if len(below) > 0:
                                df.at[idx, col_below] = pts - below[0]

                    pos_home = tabla_df[tabla_df['Equipo'] == home]['Posicion'].values
                    pos_away = tabla_df[tabla_df['Equipo'] == away]['Posicion'].values
                    if len(pos_home) > 0 and len(pos_away) > 0:
                        df.at[idx, 'Position_Diff'] = pos_away[0] - pos_home[0]

            df.at[idx, 'Matchday'] = jornada
            df.at[idx, 'Season_Progress'] = jornada / max(total_jornadas, 1)

            min_pj = min(partidos_jugados.get(home, 0), partidos_jugados.get(away, 0))
            df.at[idx, 'Position_Reliability'] = min(min_pj / 10, 1.0)

            # DESPUÉS del partido: actualizar tabla
            if pd.notna(row.get('FTHG')) and pd.notna(row.get('FTAG')):
                gh = int(row['FTHG'])
                ga = int(row['FTAG'])
                goles_favor[home] += gh
                goles_contra[home] += ga
                goles_favor[away] += ga
                goles_contra[away] += gh
                if gh > ga:
                    puntos_equipo[home] += 3
                elif gh < ga:
                    puntos_equipo[away] += 3
                else:
                    puntos_equipo[home] += 1
                    puntos_equipo[away] += 1
                partidos_jugados[home] += 1
                partidos_jugados[away] += 1

    # Features derivadas
    df['HT_Level'] = pd.to_numeric(
        pd.cut(df['HT_Position'], bins=[0, 6, 14, 20], labels=[3, 2, 1]),
        errors='coerce',
    )
    df['AT_Level'] = pd.to_numeric(
        pd.cut(df['AT_Position'], bins=[0, 6, 14, 20], labels=[3, 2, 1]),
        errors='coerce',
    )
    df['Match_Type'] = df['HT_Level'] - df['AT_Level']
    df['HT_Pressure'] = (df['HT_Points_Below'].fillna(0) + df['HT_Points_Above'].fillna(0)) / 2
    df['AT_Pressure'] = (df['AT_Points_Below'].fillna(0) + df['AT_Points_Above'].fillna(0)) / 2
    df['Position_Diff_Weighted'] = df['Position_Diff'] * df['Position_Reliability']

    # Rellenar NaN
    fill_map = {
        'HT_Position': 10, 'AT_Position': 10,
        'HT_Level': 2, 'AT_Level': 2,
        'Season_Progress': 0.5,
        'Position_Reliability': 0,
    }
    all_tabla_cols = tabla_cols + [
        'HT_Level', 'AT_Level', 'Match_Type',
        'HT_Pressure', 'AT_Pressure', 'Position_Diff_Weighted',
    ]
    for col in all_tabla_cols:
        if col in df.columns:
            df[col] = df[col].fillna(fill_map.get(col, 0))

    if 'Season' in df.columns:
        df = df.drop(columns=['Season'])

    con_posicion = (df['HT_Position'] != 10).sum()
    print(f"   ✅ Features de tabla calculadas: {con_posicion} partidos con posición real")

    return df


# ============================================================================
# FEATURES DERIVADAS DE CUOTAS
# ============================================================================

def agregar_features_cuotas_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega features derivadas de las cuotas de apuestas Bet365.

    Las probabilidades NO se normalizan aquí (se usan crudas 1/cuota) para
    que sean consistentes entre training y predicción. El margen de la casa
    queda así implícito en los valores.

    Requiere columnas: B365H, B365D, B365A, B365CH, B365CD, B365CA.

    Args:
        df: DataFrame con columnas de cuotas.

    Returns:
        DataFrame con features de probabilidades y movimiento de mercado.
    """
    print("\n🔧 Calculando features derivadas de cuotas...")

    cols_necesarias = ['B365H', 'B365D', 'B365A', 'B365CH', 'B365CD', 'B365CA']
    cols_faltantes = [c for c in cols_necesarias if c not in df.columns]
    if cols_faltantes:
        print(f"   ⚠️  Faltan columnas: {cols_faltantes}")
        return df

    # Probabilidades implícitas (apertura)
    df['Prob_H'] = 1 / df['B365H']
    df['Prob_D'] = 1 / df['B365D']
    df['Prob_A'] = 1 / df['B365A']

    # Probabilidades implícitas (cierre)
    prob_c_h = 1 / df['B365CH']
    prob_c_d = 1 / df['B365CD']
    prob_c_a = 1 / df['B365CA']

    # Movimiento de mercado
    df['Prob_Move_H'] = prob_c_h - df['Prob_H']
    df['Prob_Move_D'] = prob_c_d - df['Prob_D']
    df['Prob_Move_A'] = prob_c_a - df['Prob_A']
    df['Market_Move_Strength'] = (
        df['Prob_Move_H'].abs()
        + df['Prob_Move_D'].abs()
        + df['Prob_Move_A'].abs()
    )

    # Estructura del mercado
    prob_max = df[['Prob_H', 'Prob_D', 'Prob_A']].max(axis=1)
    prob_min = df[['Prob_H', 'Prob_D', 'Prob_A']].min(axis=1)
    df['Prob_Spread'] = prob_max - prob_min
    df['Market_Confidence'] = prob_max - (1 / 3)
    df['Home_Advantage_Prob'] = df['Prob_H'] - df['Prob_A']

    features_nuevas = [
        'Prob_H', 'Prob_D', 'Prob_A',
        'Prob_Move_H', 'Prob_Move_D', 'Prob_Move_A',
        'Market_Move_Strength', 'Prob_Spread',
        'Market_Confidence', 'Home_Advantage_Prob',
    ]
    for col in features_nuevas:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    con_datos = (df['Prob_H'] > 0).sum()
    print(f"   ✅ Features de cuotas derivadas: {con_datos} partidos con datos")

    return df


# ============================================================================
# HEAD-TO-HEAD
# ============================================================================

def calcular_h2h_features(
    df: pd.DataFrame,
    equipo_local: str,
    equipo_visitante: str,
    fecha_limite=None,
    ultimos_n: int = H2H_ULTIMOS_N,
) -> dict:
    """
    Calcula features Head-to-Head entre dos equipos ANTES de un partido.

    Incluye H2H_Available como flag explícito de disponibilidad, lo que
    permite al modelo distinguir entre "0 victorias" y "sin historial".

    Args:
        df: DataFrame histórico de partidos.
        equipo_local: Nombre del equipo local en el partido a predecir.
        equipo_visitante: Nombre del equipo visitante.
        fecha_limite: Solo considera partidos anteriores a esta fecha.
        ultimos_n: Máximo número de enfrentamientos a considerar.

    Returns:
        Dict con features H2H. Siempre incluye H2H_Available.
    """
    mask = (
        ((df['HomeTeam'] == equipo_local) & (df['AwayTeam'] == equipo_visitante))
        | ((df['HomeTeam'] == equipo_visitante) & (df['AwayTeam'] == equipo_local))
    )
    if fecha_limite is not None:
        mask = mask & (pd.to_datetime(df['Date']) < pd.to_datetime(fecha_limite))

    h2h = df[mask].sort_values('Date', ascending=False).head(ultimos_n)

    if len(h2h) == 0:
        return {
            'H2H_Available': 0,
            'H2H_Matches': 0,
            'H2H_Home_Wins': 0,
            'H2H_Draws': 0,
            'H2H_Away_Wins': 0,
            'H2H_Home_Goals_Avg': 0,
            'H2H_Away_Goals_Avg': 0,
            'H2H_Home_Win_Rate': 0.33,
            'H2H_BTTS_Rate': 0.5,
        }

    resultados_local = []
    goles_local = []
    goles_visitante = []
    btts_count = 0

    for _, partido in h2h.iterrows():
        if partido['HomeTeam'] == equipo_local:
            goles_local.append(partido['FTHG'])
            goles_visitante.append(partido['FTAG'])
            resultado_map = {'H': 'W', 'D': 'D', 'A': 'L'}
        else:
            goles_local.append(partido['FTAG'])
            goles_visitante.append(partido['FTHG'])
            resultado_map = {'A': 'W', 'D': 'D', 'H': 'L'}
        resultados_local.append(resultado_map.get(partido['FTR'], 'D'))

        if partido['FTHG'] > 0 and partido['FTAG'] > 0:
            btts_count += 1

    wins = resultados_local.count('W')
    draws = resultados_local.count('D')
    losses = resultados_local.count('L')
    total = len(resultados_local)

    home_goals_avg = float(np.mean(goles_local)) if goles_local else 0.0
    away_goals_avg = float(np.mean(goles_visitante)) if goles_visitante else 0.0
    win_rate = wins / total if total > 0 else 0.33

    return {
        'H2H_Available': 1,
        'H2H_Matches': total,
        'H2H_Home_Wins': wins,
        'H2H_Draws': draws,
        'H2H_Away_Wins': losses,
        'H2H_Home_Goals_Avg': home_goals_avg,
        'H2H_Away_Goals_Avg': away_goals_avg,
        'H2H_Home_Win_Rate': win_rate,
        'H2H_BTTS_Rate': btts_count / total if total > 0 else 0.5,
        # Derived H2H features (consistent with 01_preparar_datos.py)
        'H2H_Goal_Diff': home_goals_avg - away_goals_avg,
        'H2H_Win_Advantage': wins - losses,
        'H2H_Total_Goals_Avg': home_goals_avg + away_goals_avg,
        'H2H_Home_Consistent': 1 if win_rate >= 0.6 else 0,
    }
