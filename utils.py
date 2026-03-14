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

from config import ROLLING_WINDOW, H2H_ULTIMOS_N, ARCHIVO_XG_RAW, RUTA_RAW

warnings.filterwarnings('ignore')


# ============================================================================
# ENSEMBLE LGBM + XGB (clase serializable via joblib)
# ============================================================================

from sklearn.base import BaseEstimator, ClassifierMixin

class EnsembleLGBM_XGB(BaseEstimator, ClassifierMixin):
    """
    Ensemble de votacion ponderada entre LightGBM y XGBoost.

    LightGBM maneja NaN nativamente; XGBoost necesita fillna(0).
    El ensemble aplica el fillna internamente para XGBoost.

    Parametros:
        lgbm_model   : LGBMClassifier ya entrenado
        xgb_model    : XGBClassifier ya entrenado
        lgbm_weight  : peso de LightGBM en el promedio (default 0.25)
        xgb_weight   : peso de XGBoost en el promedio  (default 0.75)
    """

    def __init__(self, lgbm_model=None, xgb_model=None,
                 lgbm_weight: float = 0.25, xgb_weight: float = 0.75):
        self.lgbm_model  = lgbm_model
        self.xgb_model   = xgb_model
        self.lgbm_weight = lgbm_weight
        self.xgb_weight  = xgb_weight
        self.classes_    = [0, 1, 2]

    def predict_proba(self, X):
        X_fill = X.fillna(0) if hasattr(X, 'fillna') else X
        proba_lgbm = self.lgbm_model.predict_proba(X)
        proba_xgb  = self.xgb_model.predict_proba(X_fill)
        return self.lgbm_weight * proba_lgbm + self.xgb_weight * proba_xgb

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


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

    # ── xG Global: rolling usando TODOS los partidos (home+away) ──────────
    # El venue-específico de arriba solo usa ~19 partidos/temporada por venue.
    # El global duplica la muestra efectiva → rolling más estable.
    home_xg = df[['Date', 'HomeTeam', 'Home_xG', 'Away_xG']].copy()
    home_xg.columns = ['Date', 'Team', 'xGF', 'xGA']

    away_xg = df[['Date', 'AwayTeam', 'Away_xG', 'Home_xG']].copy()
    away_xg.columns = ['Date', 'Team', 'xGF', 'xGA']

    long_xg = pd.concat([home_xg, away_xg], ignore_index=True)
    long_xg = long_xg.sort_values(['Team', 'Date']).reset_index(drop=True)

    def _roll(s, w=window):
        return s.shift(1).rolling(w, min_periods=1).mean()

    grp_xg = long_xg.groupby('Team')
    long_xg['xGF_global'] = grp_xg['xGF'].transform(_roll)
    long_xg['xGA_global'] = grp_xg['xGA'].transform(_roll)

    # Separar home/away para merge
    n_home = len(df)
    home_global = long_xg.iloc[:n_home][['xGF_global', 'xGA_global']].copy()
    home_global.index = df.index
    away_global = long_xg.iloc[n_home:][['xGF_global', 'xGA_global']].copy()
    away_global.index = df.index

    df['HT_xG_Global'] = home_global['xGF_global']
    df['AT_xG_Global'] = away_global['xGF_global']
    df['HT_xGA_Global'] = home_global['xGA_global']
    df['AT_xGA_Global'] = away_global['xGA_global']
    df['xG_Global_Diff'] = df['HT_xG_Global'] - df['AT_xG_Global']

    xg_cols = ['HT_xG_Avg', 'AT_xG_Avg', 'HT_xGA_Avg', 'AT_xGA_Avg',
               'xG_Diff', 'xG_Total',
               'HT_xG_Global', 'AT_xG_Global', 'HT_xGA_Global', 'AT_xGA_Global',
               'xG_Global_Diff']
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

    P3-Audit: Ahora solo usa cuotas de APERTURA (B365H, B365D, B365A).
    Las cuotas de cierre (B365CH/CD/CA) se eliminaron porque no están
    disponibles al momento de apostar (leakage implícito).

    Requiere columnas: B365H, B365D, B365A.

    Args:
        df: DataFrame con columnas de cuotas.

    Returns:
        DataFrame con features de probabilidades derivadas de apertura.
    """
    print("\n   Calculando features derivadas de cuotas (solo apertura)...")

    cols_necesarias = ['B365H', 'B365D', 'B365A']
    cols_faltantes = [c for c in cols_necesarias if c not in df.columns]
    if cols_faltantes:
        print(f"   Faltan columnas: {cols_faltantes}")
        return df

    # Probabilidades implícitas (apertura solamente)
    df['Prob_H'] = 1 / df['B365H']
    df['Prob_D'] = 1 / df['B365D']
    df['Prob_A'] = 1 / df['B365A']

    # Estructura del mercado (solo apertura)
    prob_max = df[['Prob_H', 'Prob_D', 'Prob_A']].max(axis=1)
    prob_min = df[['Prob_H', 'Prob_D', 'Prob_A']].min(axis=1)
    df['Prob_Spread'] = prob_max - prob_min
    df['Market_Confidence'] = prob_max - (1 / 3)
    df['Home_Advantage_Prob'] = df['Prob_H'] - df['Prob_A']

    features_nuevas = [
        'Prob_H', 'Prob_D', 'Prob_A',
        'Prob_Spread', 'Market_Confidence', 'Home_Advantage_Prob',
    ]
    for col in features_nuevas:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    con_datos = (df['Prob_H'] > 0).sum()
    print(f"   Features de cuotas derivadas (apertura): {con_datos} partidos con datos")

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
        'H2H_Home_Consistent': (wins + draws) / total if total > 0 else 0.5,
    }


# ============================================================================
# ASIAN HANDICAP
# ============================================================================

def agregar_features_asian_handicap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features derivadas del Asian Handicap a partir de columnas raw del CSV.

    P3-Audit: Ahora solo usa columnas de APERTURA. Eliminadas AHCh, B365CAHH,
    B365CAHA y todas las features derivadas de cierre (AH_Line_Move,
    AH_Close_Move_H, AH_Close_Move_A) por no estar disponibles pre-partido.

    Columnas raw requeridas (football-data.co.uk):
        AHh        handicap apertura local  (ej: -1.5, 0.25, 1.0)
        B365AHH    cuota Bet365 apertura local con handicap
        B365AHA    cuota Bet365 apertura visitante con handicap
        Prob_H     probabilidad implicita 1X2 local (de agregar_features_cuotas_derivadas)
        Prob_A     probabilidad implicita 1X2 visitante

    Features generadas:
        AH_Line          handicap de apertura (señal de fuerza relativa del mercado)
        AH_Implied_Home  prob implicita local desde cuotas AH apertura (1/B365AHH normalizada)
        AH_Implied_Away  prob implicita visitante
        AH_Edge_Home     diferencia entre prob AH y prob 1X2 del mercado para el local
        AH_Market_Conf   que tan lejos esta la cuota de la linea justa (1.909 sin margen)

    No modifica el CSV. Se llama en memoria antes de entrenar/predecir.
    """
    print("\n   Calculando features Asian Handicap (solo apertura)...")

    required = ['AHh', 'B365AHH', 'B365AHA']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"   AVISO: columnas AH no encontradas: {missing}. Saltando.")
        for col in ['AH_Line', 'AH_Implied_Home', 'AH_Implied_Away',
                    'AH_Edge_Home', 'AH_Market_Conf']:
            df[col] = 0.0
        return df

    # Linea de handicap (apertura)
    df['AH_Line'] = df['AHh']

    # Probabilidades implicitas desde cuotas AH apertura
    # AH es mercado binario (sin empate): normalizamos las dos cuotas
    raw_h = 1.0 / df['B365AHH'].replace(0, np.nan)
    raw_a = 1.0 / df['B365AHA'].replace(0, np.nan)
    total = raw_h + raw_a
    df['AH_Implied_Home'] = (raw_h / total).fillna(0.5)
    df['AH_Implied_Away'] = (raw_a / total).fillna(0.5)

    # Edge AH vs mercado 1X2: diferencia entre lo que dice el AH y las cuotas 1X2
    if 'Prob_H' in df.columns and 'Prob_A' in df.columns:
        prob_1x2_h = df['Prob_H']
        prob_1x2_a = df['Prob_A']
        total_1x2 = prob_1x2_h + prob_1x2_a
        prob_1x2_h_norm = (prob_1x2_h / total_1x2).fillna(0.5)
        df['AH_Edge_Home'] = df['AH_Implied_Home'] - prob_1x2_h_norm
    else:
        df['AH_Edge_Home'] = 0.0

    # Confianza del mercado AH: distancia de la cuota ideal sin margen (1.909)
    ah_fair = 1.909
    df['AH_Market_Conf'] = (ah_fair - df['B365AHH'].clip(upper=ah_fair)).fillna(0.0)

    # Rellenar posibles NaN residuales
    ah_cols = ['AH_Line', 'AH_Implied_Home', 'AH_Implied_Away',
               'AH_Edge_Home', 'AH_Market_Conf']
    df[ah_cols] = df[ah_cols].fillna(0.0)

    con_datos = (df['AH_Line'] != 0).sum()
    print(f"   Asian Handicap (apertura): {con_datos} partidos con datos")
    print(f"   Handicap promedio local: {df['AH_Line'].mean():.3f}")

    return df


# ============================================================================
# FEATURES ROLLING EXTRA
# ============================================================================

def agregar_features_rolling_extra(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega 3 features rolling que mejoran el F1 del modelo XGBoost:
      - HT_Goals_Diff  : rolling-5 de diferencia de goles del local (como home)
      - AT_Goals_Diff  : rolling-5 de diferencia de goles del visitante (como away)
      - AT_HTR_Rate    : rolling-5 de % partidos ganando al descanso (visitante)

    Usa groupby+shift+rolling (vectorizado) para evitar loops lentos por equipo.
    El shift(1) garantiza que cada partido solo ve datos ANTERIORES a ese partido.
    """
    window = ROLLING_WINDOW

    # Asegurar orden temporal
    df = df.sort_values('Date').reset_index(drop=True)

    # HT_Goals_Diff: diferencia goles rolling del equipo LOCAL cuando juega en casa
    df['_HT_gd'] = df['FTHG'].fillna(0) - df['FTAG'].fillna(0)
    df['HT_Goals_Diff'] = (
        df.groupby('HomeTeam')['_HT_gd']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        .fillna(0)
    )

    # AT_Goals_Diff: diferencia goles rolling del equipo VISITANTE cuando juega fuera
    df['_AT_gd'] = df['FTAG'].fillna(0) - df['FTHG'].fillna(0)
    df['AT_Goals_Diff'] = (
        df.groupby('AwayTeam')['_AT_gd']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        .fillna(0)
    )

    # AT_HTR_Rate: % partidos ganando al descanso del VISITANTE como away
    df['_AT_htr'] = (df['HTAG'].fillna(0) > df['HTHG'].fillna(0)).astype(float)
    df['AT_HTR_Rate'] = (
        df.groupby('AwayTeam')['_AT_htr']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        .fillna(0)
    )

    # P3-Audit: PS_vs_Avg_H eliminada (usaba PSCH cuota de cierre Pinnacle).

    # Limpiar columnas temporales
    df.drop(columns=['_HT_gd', '_AT_gd', '_AT_htr'], inplace=True, errors='ignore')

    return df


# ============================================================================
# FEATURES MULTI-ESCALA (rolling window=10)
# ============================================================================

ROLLING_WINDOW_LONG = 10
ROLLING_WINDOW_SHORT = 3

def agregar_features_multi_escala(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega features rolling con ventana corta (3) y larga (10)
    para capturar tendencias inmediatas y de medio plazo.

    Features calculadas:
      - HT_Pts3 / AT_Pts3       : puntos últimos 3 partidos
      - HT_GoalsFor3 / AT_GoalsFor3
      - HT_xG_Avg_3 / AT_xG_Avg_3
      - HT_Pts10 / AT_Pts10     : puntos últimos 10 partidos
      - HT_GoalsFor10 / AT_GoalsFor10
      - HT_xG_Avg_10 / AT_xG_Avg_10
      - HT_Form_Momentum / AT_Form_Momentum (ratio corto/largo)
      - Form_Momentum_Diff

    Usa formato long (home+away) y shift(1) para evitar leakage.
    """
    w_long = ROLLING_WINDOW_LONG
    w_short = ROLLING_WINDOW_SHORT
    df = df.sort_values('Date').reset_index(drop=True)

    print(f"\n🔧 Calculando features multi-escala (window={w_short}/{w_long})...")

    # ── Formato long: una fila por equipo por partido ─────────────────────
    home = df[['Date', 'HomeTeam', 'FTHG', 'FTAG']].copy()
    home.columns = ['Date', 'Team', 'GF', 'GA']

    away = df[['Date', 'AwayTeam', 'FTAG', 'FTHG']].copy()
    away.columns = ['Date', 'Team', 'GF', 'GA']

    for sub in (home, away):
        sub['GF'] = pd.to_numeric(sub['GF'], errors='coerce').fillna(0)
        sub['GA'] = pd.to_numeric(sub['GA'], errors='coerce').fillna(0)
        sub['Pts'] = np.where(sub['GF'] > sub['GA'], 3,
                     np.where(sub['GF'] == sub['GA'], 1, 0))

    long_df = pd.concat([home, away], ignore_index=True)
    long_df = long_df.sort_values(['Team', 'Date']).reset_index(drop=True)

    grp = long_df.groupby('Team')
    long_df['Pts_3']      = grp['Pts'].transform(lambda x: x.shift(1).rolling(w_short, min_periods=1).sum())
    long_df['GoalsFor_3'] = grp['GF'].transform(lambda x: x.shift(1).rolling(w_short, min_periods=1).sum())
    long_df['Pts_10']     = grp['Pts'].transform(lambda x: x.shift(1).rolling(w_long, min_periods=1).sum())
    long_df['GoalsFor_10'] = grp['GF'].transform(lambda x: x.shift(1).rolling(w_long, min_periods=1).sum())

    # Separar home/away
    n = len(df)
    home_stats = long_df.iloc[:n][['Pts_3', 'GoalsFor_3', 'Pts_10', 'GoalsFor_10']].copy()
    home_stats.index = df.index
    away_stats = long_df.iloc[n:][['Pts_3', 'GoalsFor_3', 'Pts_10', 'GoalsFor_10']].copy()
    away_stats.index = df.index

    df['HT_Pts3']        = home_stats['Pts_3'].fillna(0)
    df['AT_Pts3']        = away_stats['Pts_3'].fillna(0)
    df['HT_GoalsFor3']   = home_stats['GoalsFor_3'].fillna(0)
    df['AT_GoalsFor3']   = away_stats['GoalsFor_3'].fillna(0)
    df['HT_Pts10']       = home_stats['Pts_10'].fillna(0)
    df['AT_Pts10']       = away_stats['Pts_10'].fillna(0)
    df['HT_GoalsFor10']  = home_stats['GoalsFor_10'].fillna(0)
    df['AT_GoalsFor10']  = away_stats['GoalsFor_10'].fillna(0)

    # ── xG rolling window=10 (solo si hay datos de xG) ───────────────────
    if 'Home_xG' in df.columns and 'Away_xG' in df.columns:
        home_xg = df[['Date', 'HomeTeam', 'Home_xG']].copy()
        home_xg.columns = ['Date', 'Team', 'xGF']
        away_xg = df[['Date', 'AwayTeam', 'Away_xG']].copy()
        away_xg.columns = ['Date', 'Team', 'xGF']

        long_xg = pd.concat([home_xg, away_xg], ignore_index=True)
        long_xg = long_xg.sort_values(['Team', 'Date']).reset_index(drop=True)

        long_xg['xGF_3'] = long_xg.groupby('Team')['xGF'].transform(
            lambda x: x.shift(1).rolling(w_short, min_periods=1).mean()
        )
        long_xg['xGF_10'] = long_xg.groupby('Team')['xGF'].transform(
            lambda x: x.shift(1).rolling(w_long, min_periods=1).mean()
        )

        df['HT_xG_Avg_3'] = long_xg.iloc[:n]['xGF_3'].values
        df['AT_xG_Avg_3'] = long_xg.iloc[n:]['xGF_3'].values
        df['HT_xG_Avg_10'] = long_xg.iloc[:n]['xGF_10'].values
        df['AT_xG_Avg_10'] = long_xg.iloc[n:]['xGF_10'].values
    else:
        df['HT_xG_Avg_3'] = 0.0
        df['AT_xG_Avg_3'] = 0.0
        df['HT_xG_Avg_10'] = 0.0
        df['AT_xG_Avg_10'] = 0.0

    # Momentum ratio: forma corta vs larga
    eps = 1e-6
    df['HT_Form_Momentum'] = (df['HT_Pts3'] / max(w_short, 1)) / ((df['HT_Pts10'] / max(w_long, 1)) + eps)
    df['AT_Form_Momentum'] = (df['AT_Pts3'] / max(w_short, 1)) / ((df['AT_Pts10'] / max(w_long, 1)) + eps)
    df['Form_Momentum_Diff'] = df['HT_Form_Momentum'] - df['AT_Form_Momentum']

    multi_cols = [
        'HT_Pts3', 'AT_Pts3', 'HT_GoalsFor3', 'AT_GoalsFor3',
        'HT_Pts10', 'AT_Pts10', 'HT_GoalsFor10', 'AT_GoalsFor10',
        'HT_xG_Avg_3', 'AT_xG_Avg_3', 'HT_xG_Avg_10', 'AT_xG_Avg_10',
        'HT_Form_Momentum', 'AT_Form_Momentum', 'Form_Momentum_Diff',
    ]
    for col in multi_cols:
        df[col] = df[col].fillna(0)

    print(f"   ✅ Features multi-escala: {len(multi_cols)} columnas agregadas")
    return df


# ============================================================================
# FEATURES EWM (DECAY EXPONENCIAL)
# ============================================================================

def agregar_features_ewm(df: pd.DataFrame, span: int = 5) -> pd.DataFrame:
    """
    Agrega features con decay exponencial (EWM) para forma reciente.

    Calcula EWM para:
      - Pts, Goals For/Against, Shots Target, xG y xGA (si disponibles)
    """
    print(f"\n🔧 Calculando features EWM (span={span})...")

    df = df.sort_values('Date').reset_index(drop=True)

    # Formato long
    home = df[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'HST']].copy()
    home.columns = ['Date', 'Team', 'GF', 'GA', 'ST']
    away = df[['Date', 'AwayTeam', 'FTAG', 'FTHG', 'AST']].copy()
    away.columns = ['Date', 'Team', 'GF', 'GA', 'ST']

    for sub in (home, away):
        sub['GF'] = pd.to_numeric(sub['GF'], errors='coerce').fillna(0)
        sub['GA'] = pd.to_numeric(sub['GA'], errors='coerce').fillna(0)
        sub['ST'] = pd.to_numeric(sub['ST'], errors='coerce').fillna(0)
        sub['Pts'] = np.where(sub['GF'] > sub['GA'], 3,
                     np.where(sub['GF'] == sub['GA'], 1, 0))

    long_df = pd.concat([home, away], ignore_index=True)
    long_df = long_df.sort_values(['Team', 'Date']).reset_index(drop=True)

    grp = long_df.groupby('Team')
    long_df['Pts_EWM'] = grp['Pts'].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    long_df['GF_EWM'] = grp['GF'].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    long_df['GA_EWM'] = grp['GA'].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    long_df['ST_EWM'] = grp['ST'].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())

    n = len(df)
    home_stats = long_df.iloc[:n][['Pts_EWM', 'GF_EWM', 'GA_EWM', 'ST_EWM']].copy()
    away_stats = long_df.iloc[n:][['Pts_EWM', 'GF_EWM', 'GA_EWM', 'ST_EWM']].copy()
    home_stats.index = df.index
    away_stats.index = df.index

    df['HT_Pts_EWM5'] = home_stats['Pts_EWM'].fillna(0)
    df['AT_Pts_EWM5'] = away_stats['Pts_EWM'].fillna(0)
    df['HT_GoalsFor_EWM5'] = home_stats['GF_EWM'].fillna(0)
    df['AT_GoalsFor_EWM5'] = away_stats['GF_EWM'].fillna(0)
    df['HT_GoalsAgainst_EWM5'] = home_stats['GA_EWM'].fillna(0)
    df['AT_GoalsAgainst_EWM5'] = away_stats['GA_EWM'].fillna(0)
    df['HT_ShotsTarget_EWM5'] = home_stats['ST_EWM'].fillna(0)
    df['AT_ShotsTarget_EWM5'] = away_stats['ST_EWM'].fillna(0)

    # xG EWM si hay datos
    if 'Home_xG' in df.columns and 'Away_xG' in df.columns:
        home_xg = df[['Date', 'HomeTeam', 'Home_xG', 'Away_xG']].copy()
        home_xg.columns = ['Date', 'Team', 'xGF', 'xGA']
        away_xg = df[['Date', 'AwayTeam', 'Away_xG', 'Home_xG']].copy()
        away_xg.columns = ['Date', 'Team', 'xGF', 'xGA']

        long_xg = pd.concat([home_xg, away_xg], ignore_index=True)
        long_xg = long_xg.sort_values(['Team', 'Date']).reset_index(drop=True)

        grp_xg = long_xg.groupby('Team')
        long_xg['xGF_EWM'] = grp_xg['xGF'].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
        long_xg['xGA_EWM'] = grp_xg['xGA'].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())

        df['HT_xG_EWM5'] = long_xg.iloc[:n]['xGF_EWM'].values
        df['AT_xG_EWM5'] = long_xg.iloc[n:]['xGF_EWM'].values
        df['HT_xGA_EWM5'] = long_xg.iloc[:n]['xGA_EWM'].values
        df['AT_xGA_EWM5'] = long_xg.iloc[n:]['xGA_EWM'].values
    else:
        df['HT_xG_EWM5'] = 0.0
        df['AT_xG_EWM5'] = 0.0
        df['HT_xGA_EWM5'] = 0.0
        df['AT_xGA_EWM5'] = 0.0

    ewm_cols = [
        'HT_Pts_EWM5', 'AT_Pts_EWM5',
        'HT_GoalsFor_EWM5', 'AT_GoalsFor_EWM5',
        'HT_GoalsAgainst_EWM5', 'AT_GoalsAgainst_EWM5',
        'HT_ShotsTarget_EWM5', 'AT_ShotsTarget_EWM5',
        'HT_xG_EWM5', 'AT_xG_EWM5', 'HT_xGA_EWM5', 'AT_xGA_EWM5',
    ]
    df[ewm_cols] = df[ewm_cols].fillna(0)
    print(f"   ✅ Features EWM: {len(ewm_cols)} columnas agregadas")
    return df


# ============================================================================
# FEATURES ELO (clubelo.com)
# ============================================================================

# Mapeo nombre canónico de equipo → nombre de archivo CSV en datos/raw/elo/
_ELO_TEAM_TO_FILE = {
    'Arsenal': 'Arsenal', 'Aston Villa': 'AstonVilla', 'Bournemouth': 'Bournemouth',
    'Brentford': 'Brentford', 'Brighton': 'Brighton', 'Burnley': 'Burnley',
    'Chelsea': 'Chelsea', 'Crystal Palace': 'CrystalPalace', 'Everton': 'Everton',
    'Fulham': 'Fulham', 'Ipswich': 'Ipswich', 'Leeds': 'Leeds', 'Leicester': 'Leicester',
    'Liverpool': 'Liverpool', 'Luton': 'Luton', 'Man City': 'ManCity',
    'Man United': 'ManUnited', 'Newcastle': 'Newcastle', 'Norwich': 'Norwich',
    "Nott'm Forest": 'NottmForest', 'Sheffield United': 'SheffieldUnited',
    'Southampton': 'Southampton', 'Sunderland': 'Sunderland', 'Tottenham': 'Tottenham',
    'Watford': 'Watford', 'West Brom': 'WestBrom', 'West Ham': 'WestHam', 'Wolves': 'Wolves',
}

_elo_cache = None  # cache global para no releer los CSVs en cada llamada


def _cargar_elo_all() -> pd.DataFrame:
    """Carga todos los CSVs de ELO en un único DataFrame, con cache."""
    global _elo_cache
    if _elo_cache is not None:
        return _elo_cache

    elo_dir = os.path.join(RUTA_RAW, 'elo')
    frames = []
    for team_name, file_stem in _ELO_TEAM_TO_FILE.items():
        path = os.path.join(elo_dir, f'{file_stem}.csv')
        if os.path.exists(path):
            e = pd.read_csv(path)
            e['From'] = pd.to_datetime(e['From'])
            e['To'] = pd.to_datetime(e['To'])
            e['team_name'] = team_name
            frames.append(e[['team_name', 'Elo', 'From', 'To']])

    if not frames:
        raise FileNotFoundError(f"No se encontraron CSVs de ELO en {elo_dir}")

    elo_all = pd.concat(frames, ignore_index=True).sort_values('From').reset_index(drop=True)
    _elo_cache = elo_all
    return elo_all


def _lookup_elo_for_team(team_series: pd.Series, date_series: pd.Series,
                          elo_all: pd.DataFrame, fallback_mean: dict, global_mean: float) -> pd.Series:
    """
    Para cada (team, date) devuelve el ELO vigente en esa fecha.
    Estrategia vectorizada por equipo: merge_asof por fecha dentro de cada grupo.
    """
    result = pd.Series(index=team_series.index, dtype=float)

    for team in team_series.unique():
        mask = team_series == team
        dates = date_series[mask]

        # Subset ELO de este equipo, ordenado por From
        elo_team = elo_all[elo_all['team_name'] == team].sort_values('From').reset_index(drop=True)

        if elo_team.empty:
            result[mask] = fallback_mean.get(team, global_mean)
            continue

        # Construir DataFrame de consulta
        query_df = pd.DataFrame({'Date': dates}).sort_values('Date')

        # Eliminar filas con From/To nulos (pueden existir en algunos CSVs)
        elo_clean = elo_team[['From', 'To', 'Elo']].dropna(subset=['From', 'To'])

        if elo_clean.empty:
            result[mask] = fallback_mean.get(team, global_mean)
            continue

        # merge_asof: para cada fecha en query_df, busca el registro ELO más reciente
        # cuyo 'From' sea <= Date (último ELO vigente antes o en la fecha del partido)
        merged = pd.merge_asof(
            query_df,
            elo_clean,
            left_on='Date',
            right_on='From',
            direction='backward'
        )

        # Verificar que la fecha esté dentro del rango [From, To]
        # Si To < Date → el ELO estaba expirado; igualmente lo usamos (es el más reciente disponible)
        # Esto cubre partidos recientes donde el CSV no se actualizó
        elo_values = merged['Elo'].fillna(fallback_mean.get(team, global_mean))
        elo_values.index = query_df.index  # restaurar índice original

        result[mask] = elo_values

    return result


def agregar_features_elo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega 3 features ELO al DataFrame:
      - HT_ELO   : rating ELO del equipo local en la fecha del partido
      - AT_ELO   : rating ELO del equipo visitante en la fecha del partido
      - ELO_Diff : HT_ELO - AT_ELO

    Usa pd.merge_asof por equipo (vectorizado) para evitar el loop lento
    de .iterrows() que usa el script de experimento de referencia.

    Fuente de datos: datos/raw/elo/<Equipo>.csv (28 archivos de clubelo.com)
    """
    df = df.copy()

    # Asegurar que Date sea datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Cargar todos los ELOs (con cache)
    elo_all = _cargar_elo_all()

    # Calcular fallbacks por equipo y global
    fallback_mean = elo_all.groupby('team_name')['Elo'].mean().to_dict()
    global_mean = elo_all['Elo'].mean()

    # Calcular ELO para locales y visitantes
    df['HT_ELO'] = _lookup_elo_for_team(df['HomeTeam'], df['Date'], elo_all, fallback_mean, global_mean)
    df['AT_ELO'] = _lookup_elo_for_team(df['AwayTeam'], df['Date'], elo_all, fallback_mean, global_mean)
    df['ELO_Diff'] = df['HT_ELO'] - df['AT_ELO']

    con_elo = df['HT_ELO'].notna().sum()
    print(f"   ELO: {con_elo}/{len(df)} partidos con datos ({con_elo/len(df)*100:.1f}%)")
    print(f"   ELO local promedio: {df['HT_ELO'].mean():.1f} | visitante: {df['AT_ELO'].mean():.1f}")

    return df


# ==========================================================================
# FEATURES DE STRENGTH OF RECENT SCHEDULE (SoR)
# ==========================================================================

def agregar_features_sor(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Strength of Recent Schedule (SoR): promedio de Elo de rivales en los
    ultimos `window` partidos, usando solo partidos anteriores.
    """
    if 'HT_Elo' not in df.columns or 'AT_Elo' not in df.columns:
        print("   SoR: faltan features de Elo, saltando.")
        df['HT_SoR5'] = 0.0
        df['AT_SoR5'] = 0.0
        return df

    df = df.sort_values('Date').reset_index(drop=True)

    home_rows = df[['Date', 'HomeTeam', 'AT_Elo']].copy()
    home_rows.columns = ['Date', 'Team', 'Opp_Elo']

    away_rows = df[['Date', 'AwayTeam', 'HT_Elo']].copy()
    away_rows.columns = ['Date', 'Team', 'Opp_Elo']

    long_df = pd.concat([home_rows, away_rows], ignore_index=True)
    long_df = long_df.sort_values(['Team', 'Date']).reset_index(drop=True)

    grp = long_df.groupby('Team')
    long_df['SoR'] = grp['Opp_Elo'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

    n = len(df)
    df['HT_SoR5'] = long_df.iloc[:n]['SoR'].values
    df['AT_SoR5'] = long_df.iloc[n:]['SoR'].values

    df['HT_SoR5'] = df['HT_SoR5'].fillna(df['HT_SoR5'].median() if pd.notna(df['HT_SoR5'].median()) else 0.0)
    df['AT_SoR5'] = df['AT_SoR5'].fillna(df['AT_SoR5'].median() if pd.notna(df['AT_SoR5'].median()) else 0.0)

    print("   SoR: features agregadas (HT_SoR5 / AT_SoR5)")
    return df


# ============================================================================
# FEATURES DE FORMA Y MOMENTUM
# ============================================================================

def agregar_features_forma_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega features de forma reciente y momentum para cada equipo:

    Features globales (ultimos 5 partidos de cualquier venue):
      - HT_WinRate5       : % victorias local en ultimos 5 partidos (cualquier venue)
      - AT_WinRate5       : % victorias visitante en ultimos 5 partidos (cualquier venue)
      - HT_Streak         : racha actual (>0 = victorias consecutivas, <0 = derrotas)
      - AT_Streak         : racha actual del visitante
      - HT_Pts5           : puntos ganados por local en ultimos 5 partidos (max 15)
      - AT_Pts5           : puntos ganados por visitante en ultimos 5 partidos
      - HT_GoalsFor5      : goles marcados por local en ultimos 5 partidos (total)
      - AT_GoalsFor5      : goles marcados por visitante en ultimos 5 partidos
      - HT_GoalsAgainst5  : goles encajados por local en ultimos 5 partidos
      - AT_GoalsAgainst5  : goles encajados por visitante en ultimos 5 partidos
      - Momentum_Diff     : diferencia de puntos (HT_Pts5 - AT_Pts5), indica quien llega mejor

    Features especificas de venue (rendimiento como local vs visitante):
      - HT_HomeWinRate5   : % victorias del local jugando EN CASA (ultimos 5 en casa)
      - AT_AwayWinRate5   : % victorias del visitante jugando FUERA (ultimos 5 fuera)
      - HT_HomeGoals5     : goles marcados por local en sus ultimos 5 partidos en casa
      - AT_AwayGoals5     : goles marcados por visitante en sus ultimos 5 partidos fuera

    Usa groupby + shift + rolling vectorizado. El shift(1) garantiza
    que cada partido solo ve datos ANTERIORES (sin data leakage).
    """
    window = ROLLING_WINDOW
    df = df.sort_values('Date').reset_index(drop=True)

    # ── Construir tabla de resultados por equipo (formato long) ──────────────
    # Para cada partido, creamos dos filas: una para el local y una para el visitante
    home_rows = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].copy()
    home_rows.columns = ['Date', 'Team', 'Opponent', 'GF', 'GA']
    home_rows['IsHome'] = 1
    home_rows['Result'] = np.where(
        home_rows['GF'] > home_rows['GA'], 1,
        np.where(home_rows['GF'] < home_rows['GA'], -1, 0)
    )  # 1=win, 0=draw, -1=loss
    home_rows['Pts'] = np.where(home_rows['Result'] == 1, 3,
                        np.where(home_rows['Result'] == 0, 1, 0))

    away_rows = df[['Date', 'AwayTeam', 'HomeTeam', 'FTAG', 'FTHG']].copy()
    away_rows.columns = ['Date', 'Team', 'Opponent', 'GF', 'GA']
    away_rows['IsHome'] = 0
    away_rows['Result'] = np.where(
        away_rows['GF'] > away_rows['GA'], 1,
        np.where(away_rows['GF'] < away_rows['GA'], -1, 0)
    )
    away_rows['Pts'] = np.where(away_rows['Result'] == 1, 3,
                        np.where(away_rows['Result'] == 0, 1, 0))

    # Rellenar NaN de goles con 0 antes de calcular
    for col in ['GF', 'GA', 'Result', 'Pts']:
        home_rows[col] = pd.to_numeric(home_rows[col], errors='coerce').fillna(0)
        away_rows[col] = pd.to_numeric(away_rows[col], errors='coerce').fillna(0)

    long_df = pd.concat([home_rows, away_rows], ignore_index=True)
    long_df = long_df.sort_values(['Team', 'Date']).reset_index(drop=True)

    # ── Calcular rolling stats por equipo (cualquier venue) ─────────────────
    def rolling_shift(series, w=window):
        return series.shift(1).rolling(w, min_periods=1).mean()

    def rolling_sum_shift(series, w=window):
        return series.shift(1).rolling(w, min_periods=1).sum()

    grp = long_df.groupby('Team')

    long_df['WinRate5']      = grp['Result'].transform(
        lambda x: (x.shift(1) == 1).rolling(window, min_periods=1).mean()
    )
    long_df['Pts5']          = grp['Pts'].transform(rolling_sum_shift)
    long_df['GoalsFor5']     = grp['GF'].transform(rolling_sum_shift)
    long_df['GoalsAgainst5'] = grp['GA'].transform(rolling_sum_shift)

    # Racha: suma acumulativa de victorias/derrotas consecutivas
    def calcular_racha(series):
        """Racha: +N victorias consecutivas, -N derrotas consecutivas, 0 si empate."""
        results = series.values
        rachas = np.zeros(len(results))
        for i in range(1, len(results)):
            r = results[i - 1]
            if r == 1:
                rachas[i] = max(rachas[i - 1], 0) + 1
            elif r == -1:
                rachas[i] = min(rachas[i - 1], 0) - 1
            else:
                rachas[i] = 0
        return pd.Series(rachas, index=series.index)

    long_df['Streak'] = grp['Result'].transform(calcular_racha)

    # ── Calcular rolling stats por equipo SOLO como local / solo fuera ───────
    home_long = long_df[long_df['IsHome'] == 1].copy()
    away_long = long_df[long_df['IsHome'] == 0].copy()

    home_grp = home_long.groupby('Team')
    away_grp = away_long.groupby('Team')

    home_long['HomeWinRate5'] = home_grp['Result'].transform(
        lambda x: (x.shift(1) == 1).rolling(window, min_periods=1).mean()
    )
    home_long['HomeGoals5'] = home_grp['GF'].transform(rolling_sum_shift)

    away_long['AwayWinRate5'] = away_grp['Result'].transform(
        lambda x: (x.shift(1) == 1).rolling(window, min_periods=1).mean()
    )
    away_long['AwayGoals5'] = away_grp['GF'].transform(rolling_sum_shift)

    # ── Merge de vuelta al DataFrame original ───────────────────────────────
    # Para el equipo LOCAL: tomamos filas IsHome=1
    home_stats = home_long[['Date', 'Team', 'WinRate5', 'Pts5', 'GoalsFor5',
                              'GoalsAgainst5', 'Streak', 'HomeWinRate5', 'HomeGoals5']].copy()
    home_stats.columns = ['Date', 'HomeTeam', 'HT_WinRate5', 'HT_Pts5',
                           'HT_GoalsFor5', 'HT_GoalsAgainst5', 'HT_Streak',
                           'HT_HomeWinRate5', 'HT_HomeGoals5']

    # Para el equipo VISITANTE: tomamos filas IsHome=0
    away_stats = away_long[['Date', 'Team', 'WinRate5', 'Pts5', 'GoalsFor5',
                              'GoalsAgainst5', 'Streak', 'AwayWinRate5', 'AwayGoals5']].copy()
    away_stats.columns = ['Date', 'AwayTeam', 'AT_WinRate5', 'AT_Pts5',
                           'AT_GoalsFor5', 'AT_GoalsAgainst5', 'AT_Streak',
                           'AT_AwayWinRate5', 'AT_AwayGoals5']

    df = df.merge(home_stats, on=['Date', 'HomeTeam'], how='left')
    df = df.merge(away_stats, on=['Date', 'AwayTeam'], how='left')

    # Feature derivada: diferencia de momentum
    df['Momentum_Diff'] = df['HT_Pts5'].fillna(0) - df['AT_Pts5'].fillna(0)

    # Rellenar NaN residuales
    forma_cols = [
        'HT_WinRate5', 'AT_WinRate5', 'HT_Streak', 'AT_Streak',
        'HT_Pts5', 'AT_Pts5', 'HT_GoalsFor5', 'AT_GoalsFor5',
        'HT_GoalsAgainst5', 'AT_GoalsAgainst5', 'Momentum_Diff',
        'HT_HomeWinRate5', 'AT_AwayWinRate5', 'HT_HomeGoals5', 'AT_AwayGoals5',
    ]
    for col in forma_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    print(f"\n🔧 Features de forma/momentum agregadas: {len(forma_cols)} columnas")
    return df


# ============================================================================
# FEATURES PINNACLE MOVE (señal sharp money)
# ============================================================================

def agregar_features_pinnacle_move(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega features de probabilidades implícitas Pinnacle de APERTURA.

    P3-Audit: Reemplaza la versión anterior que usaba cuotas de cierre (PSCH/PSCD/PSCA).
    Las closing odds de Pinnacle no están disponibles al momento de apostar,
    por lo que usarlas como features constituye leakage implícito.

    Ahora solo usa cuotas de apertura (PSH/PSD/PSA), que SÍ están disponibles
    antes del partido y representan la evaluación inicial del mercado más eficiente.

    Features generadas:
      - Pinnacle_Open_H : prob implícita apertura Pinnacle local (1/PSH, normalizada)
      - Pinnacle_Open_A : prob implícita apertura Pinnacle visitante (1/PSA, normalizada)
      - Pinnacle_Conf   : max(Open_H, Open_A) - 1/3 (confianza del mercado Pinnacle)
    """
    cols_necesarias = ['PSH', 'PSD', 'PSA']
    disponibles = [c for c in cols_necesarias if c in df.columns]

    if len(disponibles) < 3:
        print(f"   Pinnacle Opening: faltan columnas {set(cols_necesarias)-set(disponibles)}, saltando.")
        for col in ['Pinnacle_Open_H', 'Pinnacle_Open_A', 'Pinnacle_Conf']:
            df[col] = 0.0
        return df

    # Probabilidades implícitas Pinnacle apertura (normalizadas sin vig)
    prob_h = 1.0 / df['PSH'].replace(0, np.nan)
    prob_d = 1.0 / df['PSD'].replace(0, np.nan)
    prob_a = 1.0 / df['PSA'].replace(0, np.nan)
    total  = (prob_h + prob_d + prob_a).replace(0, np.nan)

    df['Pinnacle_Open_H'] = (prob_h / total).fillna(0)
    df['Pinnacle_Open_A'] = (prob_a / total).fillna(0)
    df['Pinnacle_Conf']   = (df[['Pinnacle_Open_H', 'Pinnacle_Open_A']].max(axis=1) - 1/3).fillna(0)

    n_validos = (df['Pinnacle_Open_H'] > 0).sum()
    print(f"\n   Pinnacle Opening agregado: {n_validos} partidos con datos")
    return df


# ============================================================================
# FEATURES DE ARBITRO
# ============================================================================

def agregar_features_arbitro(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Agrega features rolling del arbitro designado para cada partido.

    Features calculadas (usando los ultimos `window` partidos arbitrados ANTES del partido):
      - Ref_Home_WinRate   : % de victorias locales con este arbitro
      - Ref_Goals_Avg      : promedio de goles totales por partido
      - Ref_Yellow_Avg     : promedio de tarjetas amarillas totales (HY+AY)
      - Ref_Home_Yellow    : promedio de amarillas al equipo local
      - Ref_Away_Yellow    : promedio de amarillas al equipo visitante

    El shift(1) garantiza que no se usa informacion del partido actual.
    """
    if 'Referee' not in df.columns:
        print("   Arbitro: columna Referee no encontrada, saltando.")
        for col in ['Ref_Home_WinRate','Ref_Goals_Avg','Ref_Yellow_Avg',
                    'Ref_Home_Yellow','Ref_Away_Yellow']:
            df[col] = 0.0
        return df

    df = df.sort_values('Date').reset_index(drop=True)

    # Calcular variables base para cada partido
    df['_ref_home_win'] = (df['FTR'] == 'H').astype(float)
    df['_ref_goals']    = df['FTHG'].fillna(0) + df['FTAG'].fillna(0)
    df['_ref_yellow']   = df['HY'].fillna(0) + df['AY'].fillna(0)
    df['_ref_home_y']   = df['HY'].fillna(0)
    df['_ref_away_y']   = df['AY'].fillna(0)

    def rolling_ref(series, ref_series, w=window):
        """Rolling mean por arbitro con shift(1) para evitar data leakage."""
        result = series.copy().astype(float)
        for ref in ref_series.unique():
            if pd.isna(ref):
                continue
            mask = ref_series == ref
            idx  = series.index[mask]
            vals = series.loc[idx]
            rolled = vals.shift(1).rolling(w, min_periods=3).mean()
            result.loc[idx] = rolled
        return result

    print("\n🔧 Calculando features de arbitro...")
    df['Ref_Home_WinRate'] = rolling_ref(df['_ref_home_win'], df['Referee'])
    df['Ref_Goals_Avg']    = rolling_ref(df['_ref_goals'],    df['Referee'])
    df['Ref_Yellow_Avg']   = rolling_ref(df['_ref_yellow'],   df['Referee'])
    df['Ref_Home_Yellow']  = rolling_ref(df['_ref_home_y'],   df['Referee'])
    df['Ref_Away_Yellow']  = rolling_ref(df['_ref_away_y'],   df['Referee'])

    # Rellenar NaN de arbitros con pocos partidos con la media global
    ref_cols = ['Ref_Home_WinRate','Ref_Goals_Avg','Ref_Yellow_Avg',
                'Ref_Home_Yellow','Ref_Away_Yellow']
    for col in ref_cols:
        media_global = df[col].median()
        df[col] = df[col].fillna(media_global if pd.notna(media_global) else 0.0)

    # Limpiar columnas temporales
    df.drop(columns=['_ref_home_win','_ref_goals','_ref_yellow',
                     '_ref_home_y','_ref_away_y'], inplace=True, errors='ignore')

    n_refs = df['Referee'].nunique()
    print(f"   Arbitro: {n_refs} arbitros distintos, 5 features calculadas")
    return df


# ============================================================================
# FEATURES DE DESCANSO / FATIGA (multi-competición)
# ============================================================================

def agregar_features_descanso(df: pd.DataFrame,
                               ruta_fixtures: str | None = None) -> pd.DataFrame:
    """
    Agrega features de días de descanso y congestión de partidos.

    Usa un CSV externo (datos/raw/fbref_fixtures.csv) con el historial de
    partidos de todas las competiciones (Premier League + Champions League +
    Europa League + FA Cup + Carabao Cup) para calcular, para cada partido
    de PL, cuántos días descansó cada equipo desde su partido anterior
    en cualquier competición.

    Features calculadas:
      - HT_Days_Rest   : días desde el último partido del local (cualquier comp.)
      - AT_Days_Rest   : días desde el último partido del visitante
      - Rest_Diff      : HT_Days_Rest - AT_Days_Rest
      - HT_Had_Europa  : 1 si el local jugó UCL o UEL en los últimos 4 días
      - AT_Had_Europa  : 1 si el visitante jugó UCL o UEL en los últimos 4 días
      - HT_Games_15d   : partidos del local en los últimos 15 días
      - AT_Games_15d   : partidos del visitante en los últimos 15 días

    Valores por defecto (cuando no hay datos del fixture externo):
      HT/AT_Days_Rest = 7  (semana estándar entre partidos de PL)
      HT/AT_Had_Europa = 0
      HT/AT_Games_15d  = 2

    Args:
        df:             DataFrame principal con columnas Date, HomeTeam, AwayTeam.
        ruta_fixtures:  Ruta al CSV de fixtures externos. Si es None, usa
                        config.RUTA_RAW + 'fbref_fixtures.csv'.

    Returns:
        DataFrame con las 7 features de descanso añadidas.

    Nota sobre leakage:
        Solo se consideran partidos con fecha ANTERIOR (estrictamente <) a
        la fecha del partido de PL que se está procesando. El partido actual
        no se cuenta en ninguna ventana.
    """
    from config import RUTA_RAW

    if ruta_fixtures is None:
        ruta_fixtures = os.path.join(RUTA_RAW, 'fbref_fixtures.csv')

    # Valores por defecto que se asignan si no hay datos de fixtures
    DEFAULTS = {
        'HT_Days_Rest':  7.0,
        'AT_Days_Rest':  7.0,
        'Rest_Diff':     0.0,
        'HT_Had_Europa': 0.0,
        'AT_Had_Europa': 0.0,
        'HT_Games_15d':  2.0,
        'AT_Games_15d':  2.0,
    }

    for col, val in DEFAULTS.items():
        df[col] = val

    if not os.path.exists(ruta_fixtures):
        print(f"\n   Descanso: fixtures no encontrados en {ruta_fixtures}")
        print("   Usando valores por defecto. Ejecuta:")
        print("   python herramientas/descargar_fixtures_europeos.py --api-key TU_KEY")
        return df

    print("\n Calculando features de descanso / fatiga...")

    # --- Cargar y preparar el CSV de fixtures externos ---
    fixtures = pd.read_csv(ruta_fixtures, parse_dates=['Date'])
    fixtures = fixtures[['Date', 'Team', 'Comp']].copy()
    fixtures = fixtures.dropna(subset=['Date', 'Team'])
    fixtures['Date'] = pd.to_datetime(fixtures['Date'])

    # Normalizar fechas del DataFrame principal
    df = df.sort_values('Date').reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'])

    # Competiciones europeas que cuentan como "Europa"
    COMPS_EUROPA = {'Champions Lg', 'Europa Lg'}

    # Índice: dict team → lista de (Date, Comp) ordenados
    calendario: dict[str, list[tuple]] = {}
    for _, row in fixtures.iterrows():
        team = row['Team']
        if team not in calendario:
            calendario[team] = []
        calendario[team].append((row['Date'], row['Comp']))

    # Ordenar calendarios por fecha
    for team in calendario:
        calendario[team].sort(key=lambda x: x[0])

    def calcular_rest(team: str, fecha_partido: pd.Timestamp) -> dict:
        """Calcula métricas de descanso para un equipo antes de una fecha."""
        if team not in calendario:
            return {}

        partidos_previos = [
            (d, c) for d, c in calendario[team] if d < fecha_partido
        ]
        if not partidos_previos:
            return {}

        # Días desde el último partido
        ultimo_dia, _ = partidos_previos[-1]
        days_rest = (fecha_partido - ultimo_dia).days

        # ¿Jugó Europa (UCL/UEL) en los últimos 4 días?
        cutoff_europa = fecha_partido - pd.Timedelta(days=4)
        had_europa = int(any(
            d >= cutoff_europa and c in COMPS_EUROPA
            for d, c in partidos_previos
        ))

        # Partidos en los últimos 15 días
        cutoff_15d = fecha_partido - pd.Timedelta(days=15)
        games_15d = sum(d >= cutoff_15d for d, _ in partidos_previos)

        return {
            'days_rest':  float(days_rest),
            'had_europa': float(had_europa),
            'games_15d':  float(games_15d),
        }

    # --- Calcular features para cada partido ---
    ht_days, at_days = [], []
    ht_europa, at_europa = [], []
    ht_games, at_games = [], []

    for _, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        fecha = row['Date']

        h = calcular_rest(home, fecha)
        a = calcular_rest(away, fecha)

        ht_days.append(h.get('days_rest',  DEFAULTS['HT_Days_Rest']))
        at_days.append(a.get('days_rest',  DEFAULTS['AT_Days_Rest']))
        ht_europa.append(h.get('had_europa', DEFAULTS['HT_Had_Europa']))
        at_europa.append(a.get('had_europa', DEFAULTS['AT_Had_Europa']))
        ht_games.append(h.get('games_15d',  DEFAULTS['HT_Games_15d']))
        at_games.append(a.get('games_15d',  DEFAULTS['AT_Games_15d']))

    df['HT_Days_Rest']  = ht_days
    df['AT_Days_Rest']  = at_days
    df['Rest_Diff']     = df['HT_Days_Rest'] - df['AT_Days_Rest']
    df['HT_Had_Europa'] = ht_europa
    df['AT_Had_Europa'] = at_europa
    df['HT_Games_15d']  = ht_games
    df['AT_Games_15d']  = at_games
    df['Calendar_Congestion_Diff'] = df['HT_Games_15d'] - df['AT_Games_15d']

    # Estadísticas de cobertura
    con_datos = (df['HT_Days_Rest'] != DEFAULTS['HT_Days_Rest']).sum()
    total = len(df)
    europa_partidos = (df['HT_Had_Europa'] + df['AT_Had_Europa'] > 0).sum()
    print(f"   Descanso: {con_datos}/{total} partidos con datos reales "
          f"({con_datos/total*100:.1f}%)")
    print(f"   Partidos con contexto europeo: {europa_partidos} "
          f"({europa_partidos/total*100:.1f}%)")

    return df


# ============================================================================
# ELO RATINGS
# ============================================================================

def agregar_features_elo(df: pd.DataFrame,
                         k: float = 20.0,
                         home_advantage: float = 50.0,
                         initial_elo: float = 1500.0,
                         season_regression: float = 0.33) -> pd.DataFrame:
    """
    Calcula Elo ratings incrementales para cada equipo y genera features.

    El sistema Elo asigna un rating numérico a cada equipo y lo actualiza
    después de cada partido. Es una feature compacta que resume la fuerza
    histórica del equipo.

    Features generadas:
        HT_Elo: Elo del local ANTES del partido
        AT_Elo: Elo del visitante ANTES del partido
        Elo_Diff: HT_Elo - AT_Elo
        Elo_WinProb_H: Probabilidad implícita Elo de victoria local
    """
    print("   Calculando Elo ratings...")

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)

    elos = {}

    def _get_elo(team):
        return elos.get(team, initial_elo)

    def _expected(elo_a, elo_b):
        return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))

    prev_season = None
    ht_elo_list = []
    at_elo_list = []

    for _, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        date = row['Date']

        # Regresión al inicio de nueva temporada (agosto)
        if pd.notna(date):
            month = date.month
            year = date.year
            season_key = f"{year}" if month >= 8 else f"{year - 1}"
            if prev_season is not None and season_key != prev_season:
                mean_elo = np.mean(list(elos.values())) if elos else initial_elo
                for team in list(elos.keys()):
                    elos[team] = elos[team] + season_regression * (mean_elo - elos[team])
            prev_season = season_key

        elo_h = _get_elo(home)
        elo_a = _get_elo(away)
        ht_elo_list.append(elo_h)
        at_elo_list.append(elo_a)

        # Actualizar Elo con resultado
        fthg = row.get('FTHG', np.nan)
        ftag = row.get('FTAG', np.nan)

        if pd.notna(fthg) and pd.notna(ftag):
            if fthg > ftag:
                s_h, s_a = 1.0, 0.0
            elif fthg < ftag:
                s_h, s_a = 0.0, 1.0
            else:
                s_h, s_a = 0.5, 0.5

            exp_h = _expected(elo_h + home_advantage, elo_a)
            exp_a = 1.0 - exp_h

            elos[home] = elo_h + k * (s_h - exp_h)
            elos[away] = elo_a + k * (s_a - exp_a)

    df['HT_Elo'] = ht_elo_list
    df['AT_Elo'] = at_elo_list
    df['Elo_Diff'] = df['HT_Elo'] - df['AT_Elo']
    df['Elo_WinProb_H'] = 1.0 / (1.0 + 10.0 ** (-(df['Elo_Diff'] + home_advantage) / 400.0))

    print(f"   Elo: {len(elos)} equipos rastreados, "
          f"rango [{min(elos.values()):.0f} - {max(elos.values()):.0f}]")

    return df
