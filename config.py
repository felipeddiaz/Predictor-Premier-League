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

# Pesos de clase optimizados por Optuna (100 trials RF, maximizando F1 weighted)
PESOS_OPTIMOS = {
    0: 2.5498,  # Local
    1: 6.0558,  # Empate
    2: 3.8179,  # Visitante
}

# Pesos optimizados para XGBoost (busqueda en grilla, F1=0.5726 en test 15%)
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

# Capa 1: Ajuste conservador
FACTOR_CONSERVADOR = 0.60       # 40% de regresión hacia distribución uniforme

# Capa 2: Filtros de calidad
UMBRAL_EDGE_MINIMO = 0.10       # Edge mínimo requerido (10%)
CUOTA_MAXIMA = 5.0              # No apostar en underdogs extremos
PROBABILIDAD_MINIMA = 0.35      # Probabilidad mínima del modelo

# Capa 3: Kelly Criterion
KELLY_FRACTION = 0.25           # Kelly fraccionario (25% del Kelly completo)
STAKE_MAXIMO_PCT = 0.025        # Máximo 2.5% del bankroll por apuesta

# General
BANKROLL_DEFAULT = 2000
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
    'B365CH', 'B365CD', 'B365CA',
]

FEATURES_CUOTAS_DERIVADAS = [
    'Prob_H', 'Prob_D', 'Prob_A',
    'Prob_Move_H', 'Prob_Move_D', 'Prob_Move_A',
    'Market_Move_Strength',
    'Prob_Spread',
    'Market_Confidence',
    'Home_Advantage_Prob',
]

FEATURES_XG = [
    'HT_xG_Avg', 'AT_xG_Avg',
    'HT_xGA_Avg', 'AT_xGA_Avg',
    'xG_Diff', 'xG_Total',
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
    'AHh',                    # Handicap apertura r=0.44
    'AHCh',                   # Handicap cierre
    'AH_Move',                # Movimiento de línea
    'AH_Magnitude',           # Magnitud absoluta
    'AH_Home_Favored',        # Local es favorito
    'AH_Close_Match',         # Partido parejo
    'AH_Big_Favorite',        # Hay gran favorito
]

# Features rolling extra — calculadas en utils.agregar_features_rolling_extra()
FEATURES_ROLLING_EXTRA = [
    'HT_Goals_Diff',   # Diferencia de goles rolling (local como home)
    'AT_Goals_Diff',   # Diferencia de goles rolling (visitante como away)
    'AT_HTR_Rate',     # % partidos ganando al descanso (visitante)
    'PS_vs_Avg_H',     # Pinnacle vs mercado promedio local (sharp signal)
]

ALL_FEATURES = (
    FEATURES_BASE
    + FEATURES_CUOTAS
    + FEATURES_CUOTAS_DERIVADAS
    + FEATURES_XG
    + FEATURES_H2H
    + FEATURES_H2H_DERIVADAS
    + FEATURES_TABLA
    + FEATURES_ASIAN_HANDICAP
    + FEATURES_ROLLING_EXTRA
)   
