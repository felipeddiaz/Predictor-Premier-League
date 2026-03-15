# -*- coding: utf-8 -*-
"""
config.py — Configuración centralizada del proyecto.

Todas las rutas, constantes y parámetros hardcodeados del proyecto
viven aquí. Los demás scripts importan desde este módulo.
"""

import os

# Raiz del proyecto: directorio donde vive este config.py.
# Usar rutas absolutas garantiza que todos los scripts funcionen
# independientemente de desde donde se ejecuten (raiz, pipeline/, jornada/, etc.)
_RAIZ = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# RUTAS
# ============================================================================

RUTA_TEMPORADAS = os.path.join(_RAIZ, 'datos', 'temporadas') + os.sep
RUTA_PROCESADOS = os.path.join(_RAIZ, 'datos', 'procesados') + os.sep
RUTA_RAW        = os.path.join(_RAIZ, 'datos', 'raw') + os.sep
RUTA_MODELOS    = os.path.join(_RAIZ, 'modelos') + os.sep
RUTA_PORTAFOLIO = os.path.join(_RAIZ, 'portafolio_imagenes') + os.sep

# Archivos de datos
ARCHIVO_LIMPIO = os.path.join(RUTA_PROCESADOS, 'premier_league_limpio.csv')
ARCHIVO_FEATURES = os.path.join(RUTA_PROCESADOS, 'archive', 'premier_league_RESTAURADO.csv')
ARCHIVO_XG_RAW = os.path.join(RUTA_RAW, 'final_matches_xg.csv')

# Archivos de modelo
ARCHIVO_MODELO = os.path.join(RUTA_MODELOS, 'modelo_final_optimizado.pkl')
ARCHIVO_FEATURES_PKL = os.path.join(RUTA_MODELOS, 'features.pkl')
ARCHIVO_METADATA = os.path.join(RUTA_MODELOS, 'metadata.pkl')
ARCHIVO_MODELO_VB = os.path.join(RUTA_MODELOS, 'modelo_value_betting.pkl')
ARCHIVO_FEATURES_VB = os.path.join(RUTA_MODELOS, 'features_value_betting.pkl')

# ============================================================================
# PARÁMETROS DE FEATURE ENGINEERING
# ============================================================================

ROLLING_WINDOW = 5          # Ventana para promedios móviles
H2H_ULTIMOS_N = 5           # Número de enfrentamientos H2H a considerar
MISSING_THRESHOLD = 0.20    # Umbral de datos faltantes para eliminar columna

# Columnas esenciales que no pueden tener NaN
COLUMNAS_ESENCIALES = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST']

# ============================================================================
# PARÁMETROS DE ENTRENAMIENTO
# ============================================================================

RANDOM_SEED = 42
TEST_SIZE = 0.20            # 80/20 split temporal
N_FEATURES_SELECCION = 35   # Seleccionar top-N features por importancia XGBoost

# Pesos de clase optimizados por Optuna 
PESOS_OPTIMOS = {
    0: 2.5498,  # Local
    1: 6.0558,  # Empate
    2: 3.8179,  # Visitante
}

# Pesos optimizados para XGBoost
PESOS_XGB = {
    0: 0.9469,  # Local
    1: 1.5348,  # Empate
    2: 1.1303,  # Visitante
}

# Pesos optimizados para RF Balanceado (Optuna co-optimizacion)
PESOS_RF_BAL = {
    0: 2.2932,  # Local
    1: 4.2476,  # Empate
    2: 2.8255,  # Visitante
}

# Hiperparámetros óptimos encontrados por Optuna (modelo CON cuotas)
PARAMS_OPTIMOS = {
    'n_estimators':      600,
    'max_depth':         7,
    'min_samples_leaf':  2,
    'min_samples_split': 9,
    'max_features':      'log2',
    'class_weight':      PESOS_OPTIMOS,
    'random_state':      RANDOM_SEED,
    'n_jobs':            -1,
}

# Hiperparámetros para el modelo SIN cuotas (03_entrenar_sin_cuotas.py)
# Valores iniciales = mismos que PARAMS_OPTIMOS.
# Para afinarlos: ejecutar visualizar_busqueda.py con MODO_SIN_CUOTAS = True
# y actualizar estos valores con los que imprima el script.
PARAMS_OPTIMOS_VB = {
    'n_estimators': 229,
    'max_depth': 8,
    'min_samples_leaf': 3,
    'class_weight': PESOS_OPTIMOS,
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}

# Hiperparámetros XGBoost para el modelo SIN cuotas (03_entrenar_sin_cuotas.py)
# Valores iniciales = mismos que PARAMS_XGB. Sin cuotas el modelo necesita
# más regularización para no sobreajustar a features más ruidosas.
PARAMS_XGB_VB = {
    'n_estimators': 500,
    'max_depth': 5,
    'learning_rate': 0.01,
    'subsample': 0.6,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 0.6,
    'reg_alpha': 1.0,
    'reg_lambda': 3.0,
    'min_child_weight': 25,
    'gamma': 1.5,
    'early_stopping_rounds': 50,
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    'eval_metric': 'mlogloss',
}

# Hiperparámetros XGBoost para el modelo CON cuotas (02_entrenar_modelo.py)
# Optimizados por búsqueda en grilla sobre test=15% temporal, F1=0.5697 -> 0.5726
PARAMS_XGB = {
    'n_estimators': 350,
    'max_depth': 6,
    'learning_rate': 0.00566,
    'subsample': 0.5697,
    'colsample_bytree': 0.8134,
    'colsample_bylevel': 0.6459,
    'reg_alpha': 0.7067,
    'reg_lambda': 2.0347,
    'min_child_weight': 20,
    'gamma': 0.9796,
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    'eval_metric': 'mlogloss',
}

# ============================================================================
# PARÁMETROS DE VALUE BETTING
# ============================================================================

# Capa 1: Ajuste conservador (shrinkage hacia distribución uniforme)
# probs_adj = alpha * probs_modelo + (1-alpha) * [1/3, 1/3, 1/3]
# Valor por defecto. Se re-calibra empíricamente por calibrar_shrinkage()
# en 02/03_entrenar*.py buscando el alpha que minimiza Brier Score.
FACTOR_CONSERVADOR = 1.00       # Valor actualizado por grid search (Fase 3)

# Capa 2: Filtros de calidad
UMBRAL_EDGE_MINIMO = 0.10       # Edge minimo recomendado por sensibilidad (max Sharpe)
UMBRAL_EDGE_DC = 0.15           # Umbral minimo para doble oportunidad (fallback)
MARGEN_DC_SINTETICO = 0.95      # Margen aplicado a cuotas DC sinteticas
CUOTA_MAXIMA = 4.0              # No apostar en underdogs extremos
PROBABILIDAD_MINIMA = 0.45      # Probabilidad mínima del modelo

# Capa 3: Kelly Criterion
KELLY_FRACTION = 0.25           # Kelly fraccionario (25% del Kelly completo)
STAKE_MAXIMO_PCT = 0.025        # Máximo 2.5% del bankroll por apuesta

# General
BANKROLL_DEFAULT = 5000
MONEDA = "$"

# ROI anualizado: asumiendo ~50 apuestas por año
APUESTAS_ANUALES_ESTIMADAS = 50

# ============================================================================
# FEATURES CANÓNICAS
# ============================================================================
# Lista definitiva de features por categoría. Usada en training y predicción
# para garantizar consistencia.

FEATURES_BASE = [
    'HT_AvgGoals', 'AT_AvgGoals',
    'HT_AvgShotsTarget', 'AT_AvgShotsTarget',
    'HT_Form_W', 'HT_Form_D', 'HT_Form_L',
    'AT_Form_W', 'AT_Form_D', 'AT_Form_L',
]

FEATURES_CUOTAS = [
    'B365H', 'B365D', 'B365A',
    # P3-Audit: Eliminadas B365CH/CD/CA (cuotas de cierre).
    # Las closing odds no están disponibles al momento de apostar,
    # usarlas como features introduce leakage implícito.
]

FEATURES_CUOTAS_DERIVADAS = [
    'Prob_H', 'Prob_D', 'Prob_A',
    # P3-Audit: Eliminadas Prob_Move_H/D/A y Market_Move_Strength.
    # Estas dependían de cuotas de cierre (B365CH/CD/CA) que no están
    # disponibles al momento de la decisión de apuesta.
    'Prob_Spread',
    'Market_Confidence',
    'Home_Advantage_Prob',
]

FEATURES_XG = [
    'HT_xG_Avg', 'AT_xG_Avg',
    'HT_xGA_Avg', 'AT_xGA_Avg',
    'xG_Diff', 'xG_Total',
]

# xG Global: rolling usando TODOS los partidos (home+away), no solo venue-específico
FEATURES_XG_GLOBAL = [
    'HT_xG_Global', 'AT_xG_Global',
    'HT_xGA_Global', 'AT_xGA_Global',
    'xG_Global_Diff',
]

# Multi-escala: rolling window=10 para tendencias de medio plazo
FEATURES_MULTI_ESCALA = [
    'HT_Pts3', 'AT_Pts3',
    'HT_GoalsFor3', 'AT_GoalsFor3',
    'HT_xG_Avg_3', 'AT_xG_Avg_3',
    'HT_Pts10', 'AT_Pts10',
    'HT_GoalsFor10', 'AT_GoalsFor10',
    'HT_xG_Avg_10', 'AT_xG_Avg_10',
    'HT_Form_Momentum', 'AT_Form_Momentum',
    'Form_Momentum_Diff',
]

# Features EWM (decay exponencial)
FEATURES_EWM = [
    'HT_Pts_EWM5', 'AT_Pts_EWM5',
    'HT_GoalsFor_EWM5', 'AT_GoalsFor_EWM5',
    'HT_GoalsAgainst_EWM5', 'AT_GoalsAgainst_EWM5',
    'HT_ShotsTarget_EWM5', 'AT_ShotsTarget_EWM5',
    'HT_xG_EWM5', 'AT_xG_EWM5',
    'HT_xGA_EWM5', 'AT_xGA_EWM5',
]

FEATURES_H2H = [
    
    'H2H_Matches', 
    'H2H_Home_Goals_Avg', 'H2H_Away_Goals_Avg',
    'H2H_Home_Win_Rate', 'H2H_BTTS_Rate',
]

FEATURES_H2H_DERIVADAS = [
    'H2H_Goal_Diff',
    'H2H_Win_Advantage',
    'H2H_Total_Goals_Avg',
    'H2H_Home_Consistent',
]

FEATURES_TABLA = [
    'HT_Position', 'AT_Position',
    'Position_Diff', 'Position_Diff_Weighted',
    'HT_Points', 'AT_Points',
    'Season_Progress', 'Position_Reliability',
    'Match_Type',
    'HT_Pressure', 'AT_Pressure'
]

FEATURES_ASIAN_HANDICAP = [
    'AHh',                    # Handicap apertura (raw)
    # P3-Audit: Eliminadas AHCh, AH_Line_Move, AH_Close_Move_H, AH_Close_Move_A.
    # Todas dependían de líneas/cuotas de cierre no disponibles pre-partido.
    'AH_Implied_Home',        # Prob implícita local desde cuotas AH apertura
    'AH_Implied_Away',        # Prob implícita visitante desde cuotas AH apertura
    'AH_Edge_Home',           # Edge: prob AH vs prob 1X2 para local
    'AH_Market_Conf',         # Confianza del mercado AH
]

# Features rolling extra — calculadas en utils.agregar_features_rolling_extra()
FEATURES_ROLLING_EXTRA = [
    'HT_Goals_Diff',   # Diferencia de goles rolling (local como home)
    'AT_Goals_Diff',   # Diferencia de goles rolling (visitante como away)
    'AT_HTR_Rate',     # % partidos ganando al descanso (visitante)
    # P3-Audit: Eliminada PS_vs_Avg_H (usaba PSCH cuota de cierre Pinnacle)
]

# Features Pinnacle (apertura) — calculadas en utils.agregar_features_pinnacle_opening()
# P3-Audit: Reemplazadas features de cierre (PSCH/PSCD/PSCA) por apertura (PSH/PSD/PSA).
# Las probabilidades de apertura Pinnacle SÍ están disponibles antes de apostar
# y son una señal valiosa del mercado más eficiente.
FEATURES_PINNACLE = [
    'Pinnacle_Open_H',     # Prob implícita apertura Pinnacle local (normalizada)
    'Pinnacle_Open_A',     # Prob implícita apertura Pinnacle visitante (normalizada)
    'Pinnacle_Conf',       # Confianza mercado Pinnacle: max(Open_H, Open_A) - 1/3
]

# Features de árbitro — calculadas en utils.agregar_features_arbitro()
FEATURES_REFEREE = [
    'Ref_Home_WinRate',    # % victorias locales con este árbitro (rolling 20)
    'Ref_Goals_Avg',       # Promedio goles totales por partido arbitrado
    'Ref_Yellow_Avg',      # Promedio tarjetas amarillas totales (HY+AY)
    'Ref_Home_Yellow',     # Promedio amarillas al equipo local
    'Ref_Away_Yellow',     # Promedio amarillas al equipo visitante
]

# Features de forma y momentum — calculadas en utils.agregar_features_forma_momentum()
FEATURES_FORMA_MOMENTUM = [
    'HT_WinRate5',         # % victorias local últimos 5 partidos (cualquier venue)
    'AT_WinRate5',         # % victorias visitante últimos 5 partidos
    'HT_Streak',           # Racha actual local (+victorias, -derrotas)
    'AT_Streak',           # Racha actual visitante
    'HT_Pts5',             # Puntos local últimos 5 partidos
    'AT_Pts5',             # Puntos visitante últimos 5 partidos
    'HT_GoalsFor5',        # Goles marcados local últimos 5
    'AT_GoalsFor5',        # Goles marcados visitante últimos 5
    'HT_GoalsAgainst5',    # Goles encajados local últimos 5
    'AT_GoalsAgainst5',    # Goles encajados visitante últimos 5
    'Momentum_Diff',       # HT_Pts5 - AT_Pts5
    'HT_HomeWinRate5',     # % victorias local jugando EN CASA (últimos 5 en casa)
    'AT_AwayWinRate5',     # % victorias visitante jugando FUERA (últimos 5 fuera)
    'HT_HomeGoals5',       # Goles local en últimos 5 en casa
    'AT_AwayGoals5',       # Goles visitante en últimos 5 fuera
]

# Features Elo — calculadas en utils.agregar_features_elo()
FEATURES_ELO = [
    'HT_Elo',          # Elo del local antes del partido
    'AT_Elo',          # Elo del visitante antes del partido
    'Elo_Diff',        # HT_Elo - AT_Elo
    'Elo_WinProb_H',   # Prob implícita Elo de victoria local
]

# Features de descanso y fatiga (requieren datos/raw/fbref_fixtures.csv)
# Generadas por: python herramientas/descargar_fixtures_europeos.py
# Fuente: UCL + UEL + FA Cup + EFL Cup (2016-2025)
FEATURES_DESCANSO = [
    'HT_Days_Rest',   # Días desde último partido del local (cualquier comp.)
    'AT_Days_Rest',   # Días desde último partido del visitante
    'Rest_Diff',      # HT_Days_Rest - AT_Days_Rest
    'HT_Had_Europa',  # 1 si el local jugó UCL/UEL en los últimos 4 días
    'AT_Had_Europa',  # 1 si el visitante jugó UCL/UEL en los últimos 4 días
    'HT_Games_15d',   # Partidos del local en los últimos 15 días
    'AT_Games_15d',   # Partidos del visitante en los últimos 15 días
    'Calendar_Congestion_Diff',  # Diferencia de congestión de calendario
]

# Strength of Recent Schedule (SoR)
FEATURES_SOR = [
    'HT_SoR5', 'AT_SoR5'
]

# Features de interaccion — calculadas en utils.agregar_features_interaccion()
FEATURES_INTERACCION = [
    'HT_xG_Efficiency',    # Goles reales / xG esperados (local)
    'AT_xG_Efficiency',    # Goles reales / xG esperados (visitante)
    'xG_Efficiency_Diff',  # Diferencia de eficiencia xG
    'HT_Form_vs_Elo',      # Puntos recientes * dificultad calendario (local)
    'AT_Form_vs_Elo',      # Puntos recientes * dificultad calendario (visitante)
    'Form_Quality_Diff',   # Diferencia de forma ponderada
    'HT_Fatigue_Score',    # Congestion * Europa (local)
    'AT_Fatigue_Score',    # Congestion * Europa (visitante)
    'Fatigue_Diff',        # Diferencia de fatiga
]

# P1-Audit: Features con cuotas de apertura (sin closing lines)
# Usadas por el modelo principal (02_entrenar_modelo.py)
FEATURES_CON_CUOTAS_APERTURA = (
    FEATURES_BASE
    + FEATURES_CUOTAS
    + FEATURES_CUOTAS_DERIVADAS
    + FEATURES_XG
    + FEATURES_XG_GLOBAL
    + FEATURES_MULTI_ESCALA
    + FEATURES_EWM
    + FEATURES_H2H
    + FEATURES_H2H_DERIVADAS
    + FEATURES_TABLA
    + FEATURES_ASIAN_HANDICAP
    + FEATURES_ROLLING_EXTRA
    + FEATURES_PINNACLE
    + FEATURES_REFEREE
    + FEATURES_FORMA_MOMENTUM
    + FEATURES_DESCANSO
    + FEATURES_ELO
    + FEATURES_SOR
)

# Modelo estructural (sin cuotas) — usado por 03_entrenar_sin_cuotas.py
FEATURES_ESTRUCTURALES = (
    FEATURES_BASE             # 10: rendimiento, forma W/D/L
    + FEATURES_XG             #  6: xG rolling (venue-específico)
    + FEATURES_XG_GLOBAL      #  5: xG rolling global (todas las venues)
    + FEATURES_MULTI_ESCALA   #  6: rolling window=10 (medio plazo)
    + FEATURES_EWM            # 12: decay exponencial (forma reciente)
    + FEATURES_H2H            #  5: historial directo
    + FEATURES_H2H_DERIVADAS  #  4: derivadas H2H
    + FEATURES_TABLA          # 11: posición, puntos, presión
    + FEATURES_FORMA_MOMENTUM # 15: forma específica local/visitante + momentum
    + FEATURES_REFEREE        #  5: árbitro
    + FEATURES_DESCANSO       #  7: días de descanso, fatiga, congestión
    + FEATURES_ELO            #  4: Elo ratings
    + FEATURES_SOR            #  2: strength of schedule
    + FEATURES_INTERACCION    #  9: interacciones (xG efficiency, forma*SoR, fatiga)
    # Total: 111 features — cero cuotas, cero señales de mercado
)

# P3-Audit: ALL_FEATURES ahora apunta a la versión limpia (solo apertura).
ALL_FEATURES = FEATURES_CON_CUOTAS_APERTURA
