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
ARCHIVO_XG_RAW       = os.path.join(RUTA_RAW, 'final_matches_xg.csv')
ARCHIVO_FIXTURES_EXT = os.path.join(RUTA_RAW, 'fbref_fixtures.csv')

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
# Dos conjuntos de features:
#
#   ALL_FEATURES          — Modelo híbrido (con mercado). Fuente de verdad:
#                           modelos/features.pkl (55 features del modelo guardado).
#                           Validar: python -c "import pickle,config; assert set(pickle.load(open(config.ARCHIVO_FEATURES_PKL,'rb')))==set(config.ALL_FEATURES)"
#
#   FEATURES_ESTRUCTURALES — Modelo estructural (sin cuotas). Solo información
#                            disponible ANTES del partido sin depender de casas
#                            de apuestas. Base para Fase 1 del refactor.

# ----------------------------------------------------------------------------
# Grupos de features estructurales (sin mercado)
# Cada grupo lista TODAS las features que genera su función en utils.py.
# ALL_FEATURES usa subconjuntos de estos grupos (solo las que sobrevivieron
# en el modelo híbrido); FEATURES_ESTRUCTURALES los usa completos.
# ----------------------------------------------------------------------------

FEATURES_BASE = [
    # Rendimiento promedio rolling (últimos N partidos, shift(1) sin leakage)
    'HT_AvgGoals', 'AT_AvgGoals',
    'HT_AvgShotsTarget', 'AT_AvgShotsTarget',
    # Forma reciente (W/D/L de los últimos 5)
    'HT_Form_W', 'HT_Form_D', 'HT_Form_L',
    'AT_Form_W', 'AT_Form_D', 'AT_Form_L',
]

# Subconjunto de FEATURES_BASE presente en el modelo híbrido (features.pkl)
_BASE_HIBRIDO = [
    'HT_AvgGoals', 'AT_AvgGoals',
    'HT_AvgShotsTarget', 'AT_AvgShotsTarget',
    'HT_Form_W', 'AT_Form_W',
]

FEATURES_XG = [
    # Expected Goals rolling (requiere datos de fbref)
    'HT_xG_Avg', 'AT_xG_Avg',
    'HT_xGA_Avg', 'AT_xGA_Avg',
    'xG_Diff', 'xG_Total',
]

# Subconjunto de FEATURES_XG presente en el modelo híbrido
_XG_HIBRIDO = [
    'HT_xG_Avg', 'AT_xG_Avg',
    'AT_xGA_Avg',
    'xG_Diff', 'xG_Total',
]

FEATURES_H2H = [
    # Historial de enfrentamientos directos
    'H2H_Matches',
    'H2H_Home_Goals_Avg', 'H2H_Away_Goals_Avg',
    'H2H_Home_Win_Rate', 'H2H_BTTS_Rate',
    'H2H_Total_Goals_Avg',
    # Derivadas H2H
    'H2H_Goal_Diff', 'H2H_Win_Advantage', 'H2H_Home_Consistent',
]

# Subconjunto de FEATURES_H2H presente en el modelo híbrido
_H2H_HIBRIDO = [
    'H2H_Matches',
    'H2H_Away_Goals_Avg',
    'H2H_Home_Win_Rate', 'H2H_BTTS_Rate',
    'H2H_Total_Goals_Avg',
]

FEATURES_TABLA = [
    # Posición y puntos en tabla antes del partido (sin leakage)
    'HT_Position', 'AT_Position',
    'Position_Diff', 'Position_Diff_Weighted',
    'HT_Points', 'AT_Points',
    'Season_Progress', 'Position_Reliability',
    'Match_Type',
    'HT_Pressure', 'AT_Pressure',
]

# Subconjunto de FEATURES_TABLA presente en el modelo híbrido
_TABLA_HIBRIDO = [
    'AT_Position',
    'Position_Diff', 'Position_Diff_Weighted',
    'HT_Points', 'AT_Points',
    'Season_Progress', 'Position_Reliability',
    'Match_Type',
    'HT_Pressure',
]

FEATURES_FORMA_MOMENTUM = [
    # Forma específica local/visitante y momentum
    'HT_HomeWinRate5', 'HT_HomeGoals5',
    'HT_GoalsFor5', 'AT_GoalsFor5',
    'HT_Streak', 'Momentum_Diff',
]

FEATURES_REFEREE = [
    # Estadísticas históricas del árbitro asignado
    'Ref_Home_WinRate', 'Ref_Goals_Avg',
    'Ref_Yellow_Avg', 'Ref_Away_Yellow',
]

# ----------------------------------------------------------------------------
# Grupos exclusivos del modelo híbrido (con mercado)
# ----------------------------------------------------------------------------

FEATURES_CUOTAS_DERIVADAS = [
    # Probabilidades implícitas del mercado Bet365
    'Prob_A',
    'Prob_Move_H', 'Prob_Move_D', 'Prob_Move_A',
    'Prob_Spread',
]

FEATURES_ASIAN_HANDICAP = [
    # Asian Handicap (señal de mercado sharp)
    'AHh',             # Línea de apertura
    'AHCh',            # Línea de cierre
    'AH_Line_Move',    # Movimiento de línea
    'AH_Implied_Home', # Probabilidad implícita AH del local
    'AH_Edge_Home',    # Edge AH para el local
    'AH_Market_Conf',  # Confianza del mercado AH
    'AH_Close_Move_H', # Movimiento de cierre AH local
]

FEATURES_PINNACLE = [
    # Señales Pinnacle (mercado sharp de referencia)
    'Pinnacle_Move_H', 'Pinnacle_Move_D', 'Pinnacle_Move_A',
    'Pinnacle_Sharp_H', 'Pinnacle_Sharp_A',
    'Pinnacle_Conf',
]

FEATURES_ROLLING_EXTRA = [
    # Rolling extra sin cuotas
    'HT_Goals_Diff',   # Diferencia de goles rolling (local como home)
    # Con señal de mercado (excluida del modelo estructural)
    'PS_vs_Avg_H',     # Pinnacle vs mercado promedio local (sharp signal)
]

# ----------------------------------------------------------------------------
# Features legacy (existían antes, no están en ningún modelo activo)
# Conservadas para referencia histórica.
# ----------------------------------------------------------------------------

FEATURES_LEGACY = [
    # Cuotas B365 raw (reemplazadas por Prob_* derivadas)
    'B365H', 'B365D', 'B365A',
    'B365CH', 'B365CD', 'B365CA',
    # Cuotas derivadas no retenidas por el modelo híbrido
    'Prob_H', 'Prob_D',
    'Market_Move_Strength', 'Market_Confidence', 'Home_Advantage_Prob',
    # AH con nombres incorrectos (renombradas en utils.py)
    'AH_Move', 'AH_Magnitude', 'AH_Home_Favored', 'AH_Close_Match', 'AH_Big_Favorite',
    # Rolling extra no retenidas
    'AT_Goals_Diff', 'AT_HTR_Rate',
]

# ----------------------------------------------------------------------------
# ALL_FEATURES — Modelo híbrido (con mercado)
# FUENTE DE VERDAD: modelos/features.pkl (55 features)
# ----------------------------------------------------------------------------

ALL_FEATURES = (
    _BASE_HIBRIDO
    + FEATURES_CUOTAS_DERIVADAS
    + _XG_HIBRIDO
    + _H2H_HIBRIDO
    + _TABLA_HIBRIDO
    + FEATURES_ASIAN_HANDICAP
    + FEATURES_ROLLING_EXTRA
    + FEATURES_PINNACLE
    + FEATURES_REFEREE
    + FEATURES_FORMA_MOMENTUM
)

# ----------------------------------------------------------------------------
# FEATURES_ESTRUCTURALES — Modelo estructural (sin cuotas, Fase 1)
# Solo información disponible antes del partido sin depender de casas de apuestas.
# Total: 38 features.
# ----------------------------------------------------------------------------

# Features de descanso y fatiga (requieren datos de api-football.com)
# Generar con: python herramientas/descargar_fixtures_europeos.py --api-key KEY
# Fuente: datos/raw/fbref_fixtures.csv (UCL + UEL + FA Cup + Carabao Cup + PL)
FEATURES_DESCANSO = [
    'HT_Days_Rest',   # Días desde último partido del local (cualquier comp.)
    'AT_Days_Rest',   # Días desde último partido del visitante
    'Rest_Diff',      # HT_Days_Rest - AT_Days_Rest
    'HT_Had_Europa',  # 1 si el local jugó UCL/UEL en los últimos 4 días
    'AT_Had_Europa',  # 1 si el visitante jugó UCL/UEL en los últimos 4 días
    'HT_Games_15d',   # Partidos del local en los últimos 15 días
    'AT_Games_15d',   # Partidos del visitante en los últimos 15 días
]

FEATURES_ESTRUCTURALES = (
    FEATURES_BASE             # 10: rendimiento, forma W/D/L
    + FEATURES_XG             #  6: xG rolling
    + FEATURES_H2H            #  9: historial directo + derivadas
    + FEATURES_TABLA          # 11: posición, puntos, presión
    + FEATURES_FORMA_MOMENTUM #  6: forma específica local/visitante + momentum
    + FEATURES_REFEREE        #  4: árbitro
    + FEATURES_DESCANSO       #  7: días de descanso, fatiga, congestión
    # Total: 53 features — cero cuotas, cero señales de mercado
    # Excluye: FEATURES_CUOTAS_DERIVADAS, FEATURES_ASIAN_HANDICAP,
    #          FEATURES_PINNACLE, FEATURES_ROLLING_EXTRA (PS_vs_Avg_H es Pinnacle)
)
