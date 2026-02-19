# -*- coding: utf-8 -*-
"""
config.py — Configuración centralizada del proyecto.

Todas las rutas, constantes y parámetros hardcodeados del proyecto
viven aquí. Los demás scripts importan desde este módulo.
"""

import os

# ============================================================================
# RUTAS
# ============================================================================

RUTA_TEMPORADAS = './datos/temporadas/'
RUTA_PROCESADOS = './datos/procesados/'
RUTA_RAW = './datos/raw/'
RUTA_MODELOS = './modelos/'
RUTA_PORTAFOLIO = './portafolio_imagenes/'

# Archivos de datos
ARCHIVO_LIMPIO = os.path.join(RUTA_PROCESADOS, 'premier_league_limpio.csv')
ARCHIVO_FEATURES = os.path.join(RUTA_PROCESADOS, 'premier_league_con_features.csv')
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

# Pesos de clase optimizados por Optuna (100 trials, maximizando F1 weighted)
PESOS_OPTIMOS = {
    0: 1.2486,  # Local
    1: 3.3228,  # Empate
    2: 1.9519   # Visitante
}

# Hiperparámetros óptimos encontrados por Optuna (modelo CON cuotas)
PARAMS_OPTIMOS = {
    'n_estimators': 229,
    'max_depth': 8,
    'min_samples_leaf': 3,
    'class_weight': PESOS_OPTIMOS,
    'random_state': RANDOM_SEED,
    'n_jobs': -1
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
# Valores iniciales conservadores. Para optimizar: ejecutar visualizar_busqueda.py
# con MODO_XGB = True y copiar aquí los valores que imprima el script.
# scale_pos_weight no se usa aquí — XGBoost multiclase usa sample_weight por clase.
PARAMS_XGB = {
    'n_estimators': 300,
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
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
    'H2H_Available',
    'H2H_Matches', 'H2H_Home_Wins', 'H2H_Draws', 'H2H_Away_Wins',
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
    'HT_Pressure', 'AT_Pressure',
]

ALL_FEATURES = (
    FEATURES_BASE
    + FEATURES_CUOTAS
    + FEATURES_CUOTAS_DERIVADAS
    + FEATURES_XG
    + FEATURES_H2H
    + FEATURES_H2H_DERIVADAS
    + FEATURES_TABLA
)
