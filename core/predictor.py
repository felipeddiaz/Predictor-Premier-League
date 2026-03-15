# -*- coding: utf-8 -*-
"""
predictor.py — Clase Predictor encapsulada.

Encapsula toda la logica de prediccion de partidos:
  - Carga del modelo, features y datos historicos
  - Prediccion de un partido individual
  - Prediccion de una jornada completa
  - Generacion de reportes PDF y Excel

Uso:
    from core.predictor import Predictor
    from jornada.jornada_config import CONFIG_JORNADA

    predictor = Predictor()
    predictor.cargar()
    predicciones = predictor.predecir_jornada(CONFIG_JORNADA)
    predictor.generar_reporte(predicciones, CONFIG_JORNADA.numero, formato='pdf')
    predictor.generar_reporte(predicciones, CONFIG_JORNADA.numero, formato='excel')
"""

import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from config import (
    ARCHIVO_FEATURES,
    ARCHIVO_MODELO,
    ARCHIVO_FEATURES_PKL,
    ARCHIVO_MODELO_VB,
    ARCHIVO_FEATURES_VB,
    ARCHIVO_MODELO_OU,
    ARCHIVO_FEATURES_OU,
    ARCHIVO_MODELO_TARJETAS,
    ARCHIVO_FEATURES_TARJETAS,
    ARCHIVO_MODELO_CORNERS,
    ARCHIVO_FEATURES_CORNERS,
    FACTOR_CONSERVADOR,
    UMBRAL_EDGE_MINIMO,
    CUOTA_MAXIMA,
    PROBABILIDAD_MINIMA,
    KELLY_FRACTION,
    STAKE_MAXIMO_PCT,
    BANKROLL_DEFAULT,
    MONEDA,
)
import utils
from utils import (
    calcular_h2h_features,
    EnsembleLGBM_XGB,  # noqa: F401 — necesario para deserializar ensemble
    agregar_features_arbitro,
    agregar_features_goles_binarias,
    agregar_features_tarjetas_binarias,
    agregar_features_corners_binarias,
)
from core.models import Partido, Prediccion, PrediccionBinaria, ConfigJornada

# Carpeta de salida para reportes — siempre en la raiz del proyecto
# (dos niveles arriba de core/)
_RAIZ_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_REPORTES = os.path.join(_RAIZ_PROYECTO, 'reportes')
os.makedirs(RUTA_REPORTES, exist_ok=True)

try:
    from fpdf import FPDF as _FPDF_BASE
    PDF_AVAILABLE = True
except ImportError:
    _FPDF_BASE = object  # type: ignore[assignment,misc]
    PDF_AVAILABLE = False

try:
    from core.sistema_expected_value import calcular_ev, kelly_criterion, analizar_apuesta, eliminar_vig  # noqa: F401
    EV_AVAILABLE = True
except ImportError:
    EV_AVAILABLE = False

    def eliminar_vig(cuota_h, cuota_d, cuota_a):
        p_h, p_d, p_a = 1/cuota_h, 1/cuota_d, 1/cuota_a
        t = p_h + p_d + p_a
        return p_h/t, p_d/t, p_a/t


# ============================================================================
# CLASE AUXILIAR PDF
# ============================================================================

class _PDFReporte(_FPDF_BASE):  # type: ignore[misc]
    """Subclase de FPDF con cabecera y pie de pagina personalizados."""

    def __init__(self, numero_jornada: int):
        if PDF_AVAILABLE:
            super().__init__()
        self._numero_jornada = numero_jornada

    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, f'Premier League - Predicciones Jornada {self._numero_jornada}', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 5, f'Generado: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')


# ============================================================================
# CLASE PREDICTOR
# ============================================================================

class Predictor:
    """
    Encapsula el ciclo completo de prediccion de partidos de Premier League.

    Estado interno:
        _modelo        : Modelo ML cargado (RF o XGBoost calibrado)
        _features      : Lista de features que espera el modelo
        _df_historico  : DataFrame con datos historicos para extraer stats
        _bankroll      : Bankroll configurado para calculos de Kelly

    Metodos publicos:
        cargar()                → Carga modelo + features + historico
        predecir_partido()      → Predice un Partido, devuelve Prediccion
        predecir_jornada()      → Predice todos los partidos de una ConfigJornada
        generar_reporte()       → Genera PDF o Excel con los resultados
    """

    def __init__(self, bankroll: float = BANKROLL_DEFAULT):
        self._modelo = None
        self._features = None
        self._df_historico = None
        self._bankroll = bankroll
        # Modelos binarios
        self._modelo_ou = None
        self._features_ou = None
        self._modelo_tarjetas = None
        self._features_tarjetas = None
        self._modelo_corners = None
        self._features_corners = None

    # -------------------------------------------------------------------------
    # CARGA
    # -------------------------------------------------------------------------

    def cargar(self) -> bool:
        """
        Carga el modelo, features y datos historicos desde disco.

        Prioridad: modelo con cuotas (02) → modelo sin cuotas (03).
        Retorna True si la carga fue exitosa, False si fallo.
        """
        print("=" * 70)
        print("CARGANDO SISTEMA DE PREDICCION")
        print("=" * 70)

        if os.path.exists(ARCHIVO_MODELO):
            modelo_file = ARCHIVO_MODELO
            features_file = ARCHIVO_FEATURES_PKL
        elif os.path.exists(ARCHIVO_MODELO_VB):
            modelo_file = ARCHIVO_MODELO_VB
            features_file = ARCHIVO_FEATURES_VB
        else:
            print("ERROR: No se encontro el modelo.")
            print("   Ejecuta: python 02_entrenar_modelo.py")
            return False

        self._modelo = joblib.load(modelo_file)
        self._features = joblib.load(features_file) if os.path.exists(features_file) else None

        ruta_datos = ARCHIVO_FEATURES
        self._df_historico = pd.read_csv(ruta_datos) if os.path.exists(ruta_datos) else None

        if self._features is None or self._df_historico is None:
            print("ERROR: No se pudieron cargar features o datos historicos.")
            return False

        # Enriquecer el historico con todas las features engineered
        # para que _obtener_stats_equipo pueda extraer tabla/forma/momentum
        try:
            df_enriq = self._df_historico.copy()
            df_enriq['Date'] = pd.to_datetime(df_enriq['Date'], errors='coerce')
            df_enriq = df_enriq.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                df_enriq = utils.agregar_xg_rolling(df_enriq)
                df_enriq = utils.agregar_features_tabla(df_enriq)
                df_enriq = utils.agregar_features_cuotas_derivadas(df_enriq)
                df_enriq = utils.agregar_features_asian_handicap(df_enriq)
                df_enriq = utils.agregar_features_rolling_extra(df_enriq)
                df_enriq = utils.agregar_features_multi_escala(df_enriq)
                df_enriq = utils.agregar_features_ewm(df_enriq)
                df_enriq = utils.agregar_features_forma_momentum(df_enriq)
                df_enriq = utils.agregar_features_descanso(df_enriq)
                df_enriq = utils.agregar_features_elo(df_enriq)
                df_enriq = utils.agregar_features_sor(df_enriq)
                df_enriq = utils.agregar_features_arbitro(df_enriq)
                df_enriq = utils.agregar_features_goles_binarias(df_enriq)
                df_enriq = utils.agregar_features_tarjetas_binarias(df_enriq)
                df_enriq = utils.agregar_features_corners_binarias(df_enriq)
            self._df_enriquecido = df_enriq
        except Exception as e:
            print(f"Advertencia: no se pudo enriquecer el historico ({e}). Usando CSV base.")
            self._df_enriquecido = self._df_historico

        # Cargar modelos binarios (opcionales)
        binarios_cargados = []
        for attr_m, attr_f, ruta_m, ruta_f, label in [
            ('_modelo_ou', '_features_ou', ARCHIVO_MODELO_OU, ARCHIVO_FEATURES_OU, 'Over/Under'),
            ('_modelo_tarjetas', '_features_tarjetas', ARCHIVO_MODELO_TARJETAS, ARCHIVO_FEATURES_TARJETAS, 'Tarjetas'),
            ('_modelo_corners', '_features_corners', ARCHIVO_MODELO_CORNERS, ARCHIVO_FEATURES_CORNERS, 'Corners'),
        ]:
            if os.path.exists(ruta_m) and os.path.exists(ruta_f):
                setattr(self, attr_m, joblib.load(ruta_m))
                setattr(self, attr_f, joblib.load(ruta_f))
                binarios_cargados.append(label)

        print(f"Modelo cargado  : {os.path.basename(modelo_file)}")
        print(f"Features        : {len(self._features)}")
        if binarios_cargados:
            print(f"Modelos binarios: {', '.join(binarios_cargados)}")
        print(f"Historico       : {len(self._df_historico)} partidos\n")
        return True

    # -------------------------------------------------------------------------
    # PREDICCION DE UN PARTIDO
    # -------------------------------------------------------------------------

    def predecir_partido(self, partido: Partido) -> Prediccion | None:
        """
        Predice el resultado de un partido y retorna un objeto Prediccion.

        Aplica:
          - Extraccion de stats historicas del equipo
          - Transformacion de cuotas en features canonicas
          - Features H2H
          - Capa 1: ajuste conservador de probabilidades

        Retorna None si no se pudieron obtener stats del historico.
        """
        if self._modelo is None:
            raise RuntimeError("El sistema no ha sido cargado. Llama a cargar() primero.")

        local = partido.local
        visitante = partido.visitante

        stats_local = self._obtener_stats_equipo(local, es_local=True)
        stats_visitante = self._obtener_stats_equipo(visitante, es_local=False)

        if stats_local is None or stats_visitante is None:
            return None

        features_cuotas = self._transformar_cuotas(partido.cuota_h, partido.cuota_d, partido.cuota_a)

        datos = {
            'HT_AvgGoals': stats_local['AvgGoals'],
            'AT_AvgGoals': stats_visitante['AvgGoals'],
            'HT_AvgShotsTarget': stats_local['AvgShotsTarget'],
            'AT_AvgShotsTarget': stats_visitante['AvgShotsTarget'],
            'HT_Form_W': stats_local['Form_W'],
            'HT_Form_D': stats_local['Form_D'],
            'HT_Form_L': stats_local['Form_L'],
            'AT_Form_W': stats_visitante['Form_W'],
            'AT_Form_D': stats_visitante['Form_D'],
            'AT_Form_L': stats_visitante['Form_L'],
        }
        datos.update(features_cuotas)

        if 'xG_Avg' in stats_local and 'xG_Avg' in stats_visitante:
            datos['HT_xG_Avg'] = stats_local.get('xG_Avg', 0)
            datos['AT_xG_Avg'] = stats_visitante.get('xG_Avg', 0)
            datos['xG_Diff'] = datos['HT_xG_Avg'] - datos['AT_xG_Avg']
            datos['xG_Total'] = datos['HT_xG_Avg'] + datos['AT_xG_Avg']

        if 'xGA_Avg' in stats_local and 'xGA_Avg' in stats_visitante:
            datos['HT_xGA_Avg'] = stats_local.get('xGA_Avg', 0)
            datos['AT_xGA_Avg'] = stats_visitante.get('xGA_Avg', 0)

        # xG global (todas las venues)
        if 'xG_Global' in stats_local and 'xG_Global' in stats_visitante:
            datos['HT_xG_Global'] = stats_local.get('xG_Global', 0)
            datos['AT_xG_Global'] = stats_visitante.get('xG_Global', 0)
            datos['xG_Global_Diff'] = datos['HT_xG_Global'] - datos['AT_xG_Global']
        if 'xGA_Global' in stats_local and 'xGA_Global' in stats_visitante:
            datos['HT_xGA_Global'] = stats_local.get('xGA_Global', 0)
            datos['AT_xGA_Global'] = stats_visitante.get('xGA_Global', 0)

        # Features de tabla (posicion, puntos, presion) y forma/momentum
        # Extraidas del historico enriquecido por _obtener_stats_equipo
        _MAPEO_HT = {
            'HT_Points':      'Points',
            'HT_Pressure':    'Pressure',
            'HT_WinRate5':    'WinRate5',
            'HT_HomeWinRate5':'HomeWinRate5',
            'HT_Pts5':        'Pts5',
            'HT_GoalsFor5':   'GoalsFor5',
            'HT_GoalsAgainst5':'GoalsAgainst5',
            'HT_HomeGoals5':  'HomeGoals5',
            'HT_Streak':      'Streak',
            # multi-escala (window=10)
            'HT_Pts3':        'Pts3',
            'HT_GoalsFor3':   'GoalsFor3',
            'HT_xG_Avg_3':    'xG_Avg_3',
            'HT_Pts10':       'Pts10',
            'HT_GoalsFor10':  'GoalsFor10',
            'HT_xG_Avg_10':   'xG_Avg_10',
            'HT_Form_Momentum': 'Form_Momentum',
            # EWM
            'HT_Pts_EWM5':    'Pts_EWM5',
            'HT_GoalsFor_EWM5': 'GoalsFor_EWM5',
            'HT_GoalsAgainst_EWM5': 'GoalsAgainst_EWM5',
            'HT_ShotsTarget_EWM5': 'ShotsTarget_EWM5',
            'HT_xG_EWM5':     'xG_EWM5',
            'HT_xGA_EWM5':    'xGA_EWM5',
            # SoR
            'HT_SoR5':        'SoR5',
        }
        _MAPEO_AT = {
            'AT_Points':      'Points',
            'AT_Pressure':    'Pressure',
            'AT_WinRate5':    'WinRate5',
            'AT_AwayWinRate5':'AwayWinRate5',
            'AT_Pts5':        'Pts5',
            'AT_GoalsFor5':   'GoalsFor5',
            'AT_GoalsAgainst5':'GoalsAgainst5',
            'AT_AwayGoals5':  'AwayGoals5',
            'AT_Streak':      'Streak',
            # multi-escala (window=10)
            'AT_Pts3':        'Pts3',
            'AT_GoalsFor3':   'GoalsFor3',
            'AT_xG_Avg_3':    'xG_Avg_3',
            'AT_Pts10':       'Pts10',
            'AT_GoalsFor10':  'GoalsFor10',
            'AT_xG_Avg_10':   'xG_Avg_10',
            'AT_Form_Momentum': 'Form_Momentum',
            # EWM
            'AT_Pts_EWM5':    'Pts_EWM5',
            'AT_GoalsFor_EWM5': 'GoalsFor_EWM5',
            'AT_GoalsAgainst_EWM5': 'GoalsAgainst_EWM5',
            'AT_ShotsTarget_EWM5': 'ShotsTarget_EWM5',
            'AT_xG_EWM5':     'xG_EWM5',
            'AT_xGA_EWM5':    'xGA_EWM5',
            # SoR
            'AT_SoR5':        'SoR5',
        }
        for feat_modelo, stat_key in _MAPEO_HT.items():
            if stat_key in stats_local:
                datos[feat_modelo] = stats_local[stat_key]
        for feat_modelo, stat_key in _MAPEO_AT.items():
            if stat_key in stats_visitante:
                datos[feat_modelo] = stats_visitante[stat_key]

        # Position_Diff y Position_Diff_Weighted (requieren posicion de ambos equipos)
        pos_ht = stats_local.get('Position', 10.0)
        pos_at = stats_visitante.get('Position', 10.0)
        datos['Position_Diff'] = float(pos_at - pos_ht)
        datos['Position_Diff_Weighted'] = float((pos_at - pos_ht) / max(pos_ht, pos_at, 1))

        # Season_Progress y Match_Type: extraer del ultimo partido del historico enriquecido
        df_enriq = getattr(self, '_df_enriquecido', None)
        if df_enriq is not None and len(df_enriq) > 0:
            ultimo_global = df_enriq.iloc[-1]
            if 'Season_Progress' in ultimo_global.index and pd.notna(ultimo_global['Season_Progress']):
                datos['Season_Progress'] = float(ultimo_global['Season_Progress'])
            if 'Match_Type' in ultimo_global.index and pd.notna(ultimo_global['Match_Type']):
                datos['Match_Type'] = float(ultimo_global['Match_Type'])

        try:
            h2h_features = calcular_h2h_features(
                df=self._df_historico,
                equipo_local=local,
                equipo_visitante=visitante,
                fecha_limite=None,
            )
            datos.update(h2h_features)
        except Exception:
            datos['H2H_Available'] = 0

        # Asian Handicap — solo si el partido incluye cuotas AH
        datos.update(self._transformar_asian_handicap(partido, features_cuotas))

        # Features de descanso — calcular desde el último partido de cada equipo
        datos.update(self._calcular_descanso_prediccion(local, visitante))

        datos_filtrado = {k: v for k, v in datos.items() if k in self._features}
        # No hacemos fillna(0) aqui: el ensemble maneja NaN internamente
        # (LightGBM usa NaN nativos, XGB hace fillna dentro del ensemble)
        partido_df = pd.DataFrame([datos_filtrado], columns=self._features)

        probs_originales = self._modelo.predict_proba(partido_df)[0]
        probs = self._ajustar_probabilidades_conservador(probs_originales)

        idx_prediccion = int(np.argmax(probs))
        resultado_predicho = ['Local', 'Empate', 'Visitante'][idx_prediccion]

        if idx_prediccion == 0:
            diferencia_valor = probs[0] - features_cuotas['Prob_H']
        elif idx_prediccion == 1:
            diferencia_valor = probs[1] - features_cuotas['Prob_D']
        else:
            diferencia_valor = probs[2] - features_cuotas['Prob_A']

        forma_local = (
            f"{stats_local['Form_W']:.0f}W-"
            f"{stats_local['Form_D']:.0f}D-"
            f"{stats_local['Form_L']:.0f}L"
        )
        forma_visitante = (
            f"{stats_visitante['Form_W']:.0f}W-"
            f"{stats_visitante['Form_D']:.0f}D-"
            f"{stats_visitante['Form_L']:.0f}L"
        )

        return Prediccion(
            partido=partido,
            prob_local=float(probs[0]),
            prob_empate=float(probs[1]),
            prob_visitante=float(probs[2]),
            prob_local_original=float(probs_originales[0]),
            prob_empate_original=float(probs_originales[1]),
            prob_visitante_original=float(probs_originales[2]),
            resultado_predicho=resultado_predicho,
            confianza=float(np.max(probs)),
            diferencia_valor=float(diferencia_valor),
            prob_mercado_local=features_cuotas['Prob_H'],
            prob_mercado_empate=features_cuotas['Prob_D'],
            prob_mercado_visitante=features_cuotas['Prob_A'],
            forma_local=forma_local,
            forma_visitante=forma_visitante,
        )

    # -------------------------------------------------------------------------
    # PREDICCION DE MERCADOS BINARIOS
    # -------------------------------------------------------------------------

    def predecir_mercados_binarios(self, partido: Partido) -> PrediccionBinaria:
        """
        Predice mercados binarios (O/U goles, tarjetas, corners) para un partido.

        Extrae las features del último partido de cada equipo del histórico
        enriquecido y ejecuta cada modelo binario disponible.
        """
        resultado = PrediccionBinaria()
        df = getattr(self, '_df_enriquecido', None)
        if df is None or len(df) == 0:
            return resultado

        local = partido.local
        visitante = partido.visitante

        # Extraer features del último partido de cada equipo
        datos = self._extraer_features_binarias(local, visitante)

        for modelo, features, attr in [
            (self._modelo_ou, self._features_ou, 'prob_over25'),
            (self._modelo_tarjetas, self._features_tarjetas, 'prob_over35_cards'),
            (self._modelo_corners, self._features_corners, 'prob_over95_corners'),
        ]:
            if modelo is None or features is None:
                continue
            try:
                fila = {f: datos.get(f, 0.0) for f in features}
                X = pd.DataFrame([fila], columns=features)
                prob = modelo.predict_proba(X)[0][1]  # P(Over)
                setattr(resultado, attr, float(prob))
            except Exception as e:
                print(f"  Advertencia: {attr} no pudo predecirse ({e})")

        return resultado

    def _extraer_features_binarias(self, local: str, visitante: str) -> dict:
        """Extrae features para modelos binarios del histórico enriquecido."""
        df = self._df_enriquecido
        datos = {}

        for equipo, es_local in [(local, True), (visitante, False)]:
            partidos = df[(df['HomeTeam'] == equipo) | (df['AwayTeam'] == equipo)]
            if len(partidos) == 0:
                continue
            ultimo = partidos.iloc[-1]
            jugaba_local = ultimo['HomeTeam'] == equipo

            # Prefijo del modelo: HT_ si es_local, AT_ si no
            prefix = 'HT' if es_local else 'AT'
            # Prefijo en el DataFrame depende de si jugaba en casa
            src_prefix = 'HT' if jugaba_local else 'AT'
            other_prefix = 'AT' if jugaba_local else 'HT'

            # Features que siguen patrón directo: HT_xxx -> HT_xxx (si local) o AT_xxx
            # El modelo espera HT_ para local y AT_ para visitante
            # El DataFrame tiene HT_ para home y AT_ para away
            feat_map_same = [
                'AvgGoals', 'AvgShotsTarget', 'xG_Avg', 'xGA_Avg',
                'TotalGoals5', 'Over25_Rate5', 'BTTS_Rate5',
                'Shots5', 'ShotsTarget5', 'xG_Residual5',
                'YellowAvg5', 'YellowFor5', 'RedRate5',
                'CornersFor5', 'CornersAgainst5', 'CornersTotal5',
                'Corner_Dominance5', 'Fouls5',
                'Days_Rest', 'Elo',
                'Form_W', 'Form_D', 'Form_L',
                'Position', 'Pressure',
            ]

            for feat in feat_map_same:
                col_src = f'{src_prefix}_{feat}'
                col_dst = f'{prefix}_{feat}'
                if col_src in ultimo.index and pd.notna(ultimo[col_src]):
                    datos[col_dst] = float(ultimo[col_src])
                else:
                    # Try the other prefix (equipo jugaba como away, feature tiene prefijo AT_)
                    col_other = f'{other_prefix}_{feat}'
                    if col_other in ultimo.index and pd.notna(ultimo[col_other]):
                        datos[col_dst] = float(ultimo[col_other])

        # Features derivadas (no prefijadas)
        ht_xg = datos.get('HT_xG_Avg', 0)
        at_xg = datos.get('AT_xG_Avg', 0)
        datos['xG_Diff'] = ht_xg - at_xg
        datos['xG_Total'] = ht_xg + at_xg
        datos['Elo_Diff'] = datos.get('HT_Elo', 1500) - datos.get('AT_Elo', 1500)
        datos['Rest_Diff'] = datos.get('HT_Days_Rest', 7) - datos.get('AT_Days_Rest', 7)
        datos['Position_Diff'] = datos.get('AT_Position', 10) - datos.get('HT_Position', 10)
        datos['Corner_Dominance_Diff'] = datos.get('HT_Corner_Dominance5', 0) - datos.get('AT_Corner_Dominance5', 0)

        # H2H features para binarios
        try:
            h2h = calcular_h2h_features(self._df_historico, local, visitante)
            datos.update(h2h)
        except Exception:
            pass

        # H2H Over25_Rate, Yellow_Avg, Corners_Avg — del df enriquecido
        h2h_partidos = df[
            ((df['HomeTeam'] == local) & (df['AwayTeam'] == visitante)) |
            ((df['HomeTeam'] == visitante) & (df['AwayTeam'] == local))
        ]
        if len(h2h_partidos) > 0:
            ultimo_h2h = h2h_partidos.iloc[-1]
            for col in ['H2H_Over25_Rate', 'H2H_Yellow_Avg', 'H2H_Corners_Avg']:
                if col in ultimo_h2h.index and pd.notna(ultimo_h2h[col]):
                    datos[col] = float(ultimo_h2h[col])

        # Referee features — del último partido del equipo LOCAL (no global)
        partidos_local = df[
            (df['HomeTeam'] == local) | (df['AwayTeam'] == local)
        ]
        if len(partidos_local) > 0:
            ultimo_local = partidos_local.iloc[-1]
            for col in ['Ref_Yellow_Avg', 'Ref_Cards_Total_Avg']:
                if col in ultimo_local.index and pd.notna(ultimo_local[col]):
                    datos[col] = float(ultimo_local[col])

        # Recalcular Ref_Aggressiveness_Interaction con valores del partido actual
        ref_yellow = datos.get('Ref_Yellow_Avg', 0.0)
        ht_yellow = datos.get('HT_YellowAvg5', 0.0)
        at_yellow = datos.get('AT_YellowAvg5', 0.0)
        datos['Ref_Aggressiveness_Interaction'] = ref_yellow * (ht_yellow + at_yellow)

        return datos

    # -------------------------------------------------------------------------
    # PREDICCION DE JORNADA COMPLETA
    # -------------------------------------------------------------------------

    def predecir_jornada(self, config: ConfigJornada) -> list[Prediccion]:
        """
        Predice todos los partidos de una jornada.

        Retorna lista de Prediccion (solo los partidos que pudieron predecirse).
        """
        print("=" * 70)
        print(f"PREDICIENDO {len(config)} PARTIDOS — JORNADA {config.numero}")
        print("=" * 70 + "\n")

        predicciones = []
        total = len(config.partidos)

        tiene_binarios = any([self._modelo_ou, self._modelo_tarjetas, self._modelo_corners])

        for i, partido in enumerate(config.partidos, 1):
            print(f"[{i}/{total}] {partido.local} vs {partido.visitante}...", end=' ')
            pred = self.predecir_partido(partido)
            if pred:
                if tiene_binarios:
                    pred.mercados_binarios = self.predecir_mercados_binarios(partido)
                predicciones.append(pred)
                print(f"{pred.resultado_predicho} ({pred.confianza:.1%})")
            else:
                print("ERROR — equipo no encontrado en historico")

        print(f"\nPredicciones completadas: {len(predicciones)}/{total}\n")
        return predicciones

    # -------------------------------------------------------------------------
    # GENERACION DE REPORTES
    # -------------------------------------------------------------------------

    def generar_reporte(
        self,
        predicciones: list[Prediccion],
        numero_jornada: int,
        formato: str = 'pdf',
    ) -> str | None:
        """
        Genera un reporte con los resultados de la jornada.

        Args:
            predicciones   : Lista de Prediccion devuelta por predecir_jornada()
            numero_jornada : Numero de jornada (para nombre de archivo)
            formato        : 'pdf' o 'excel'

        Retorna la ruta del archivo generado, o None si fallo.
        """
        resultados = [p.a_dict() for p in predicciones]
        # Agregar mercados binarios al dict para los reportes
        for r, p in zip(resultados, predicciones):
            mb = p.mercados_binarios
            if mb:
                r['prob_over25'] = mb.prob_over25
                r['prob_over35_cards'] = mb.prob_over35_cards
                r['prob_over95_corners'] = mb.prob_over95_corners
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"jornada{numero_jornada}_{timestamp}"

        if formato == 'pdf':
            ruta = os.path.join(RUTA_REPORTES, f'predicciones_{tag}.pdf')
            return self._generar_pdf(resultados, numero_jornada, ruta)
        elif formato == 'excel':
            ruta = os.path.join(RUTA_REPORTES, f'predicciones_{tag}.xlsx')
            return self._generar_excel(resultados, ruta)
        else:
            print(f"Formato no reconocido: '{formato}'. Usa 'pdf' o 'excel'.")
            return None

    # -------------------------------------------------------------------------
    # VALUE BETTING — 3 CAPAS
    # -------------------------------------------------------------------------

    def calcular_value_bet(self, prob_modelo: float, cuota_mercado: float,
                           prob_fair: float | None = None) -> dict | None:
        """
        Aplica el filtro de 3 capas a una probabilidad y cuota.

        Capa 1: se aplica antes de llamar a este metodo (en predecir_partido).
        Capa 2: filtros de edge, cuota maxima y probabilidad minima.
        Capa 3: sizing por Kelly fraccional con tope de stake.

        Args:
            prob_fair: probabilidad sin vig (de eliminar_vig). Si None,
                       usa 1/cuota (con vig — sobreestima edge).

        Retorna dict con info de la apuesta, o None si no pasa los filtros.
        """
        prob_mercado = prob_fair if prob_fair is not None else (1.0 / cuota_mercado)
        edge = prob_modelo - prob_mercado
        edge_pct = edge / prob_mercado if prob_mercado > 0 else 0

        if edge_pct < UMBRAL_EDGE_MINIMO:
            return None
        if cuota_mercado > CUOTA_MAXIMA:
            return None
        if prob_modelo < PROBABILIDAD_MINIMA:
            return None

        b = cuota_mercado - 1
        kelly_full = (b * prob_modelo - (1 - prob_modelo)) / b
        kelly_full = max(0, kelly_full)
        kelly_fraction = min(kelly_full * KELLY_FRACTION, STAKE_MAXIMO_PCT)

        stake = self._bankroll * kelly_fraction
        ganancia = stake * (cuota_mercado - 1)
        ev = (prob_modelo * ganancia) - ((1 - prob_modelo) * stake)
        roi = ev / stake if stake > 0 else 0

        return {
            'prob_modelo': prob_modelo,
            'prob_mercado': prob_mercado,
            'cuota': cuota_mercado,
            'edge': edge,
            'edge_pct': edge_pct,
            'kelly_full': kelly_full,
            'kelly_fraction': kelly_fraction,
            'stake': stake,
            'ev': ev,
            'roi': roi,
        }

    # -------------------------------------------------------------------------
    # METODOS PRIVADOS — TRANSFORMACIONES
    # -------------------------------------------------------------------------

    def _ajustar_probabilidades_conservador(self, probabilidades: np.ndarray) -> np.ndarray:
        """Capa 1: mezcla probs del modelo con distribucion uniforme."""
        uniforme = np.array([1 / 3, 1 / 3, 1 / 3])
        ajustadas = FACTOR_CONSERVADOR * probabilidades + (1 - FACTOR_CONSERVADOR) * uniforme
        return ajustadas / ajustadas.sum()

    def _transformar_cuotas(self, cuota_h: float, cuota_d: float, cuota_a: float) -> dict:
        """Convierte cuotas en features canonicas (identicas a las de entrenamiento)."""
        prob_h = 1 / cuota_h
        prob_d = 1 / cuota_d
        prob_a = 1 / cuota_a
        prob_max = max(prob_h, prob_d, prob_a)
        prob_min = min(prob_h, prob_d, prob_a)

        # P3-Audit: Solo features de apertura (eliminadas Prob_Move_* y Market_Move_Strength)
        return {
            'Prob_H': prob_h,
            'Prob_D': prob_d,
            'Prob_A': prob_a,
            'Prob_Spread': prob_max - prob_min,
            'Market_Confidence': prob_max - (1 / 3),
            'Home_Advantage_Prob': prob_h - prob_a,
        }

    def _transformar_asian_handicap(self, partido: Partido, features_cuotas: dict) -> dict:
        """
        Calcula las features de Asian Handicap para prediccion en vivo.

        Si el partido no tiene datos AH (ah_line=None), todas las features
        se rellenan con 0.0 — el modelo las tratara como ausentes y usara
        el resto de features para predecir.

        Los calculos son identicos a agregar_features_asian_handicap() en utils.py
        para garantizar consistencia entre entrenamiento y prediccion.
        """
        # P3-Audit: Solo features de apertura AH (eliminadas closing line features)
        if not partido.tiene_ah():
            return {
                'AH_Line': 0.0,
                'AH_Implied_Home': 0.5,
                'AH_Implied_Away': 0.5,
                'AH_Edge_Home': 0.0,
                'AH_Market_Conf': 0.0,
            }

        ah_line    = partido.ah_line
        ah_cuota_h = partido.ah_cuota_h
        ah_cuota_a = partido.ah_cuota_a

        # Probabilidades implicitas AH (mercado binario, sin empate)
        raw_h = 1.0 / ah_cuota_h if ah_cuota_h else 0.5
        raw_a = 1.0 / ah_cuota_a if ah_cuota_a else 0.5
        total = raw_h + raw_a if (raw_h + raw_a) > 0 else 1.0
        ah_implied_h = raw_h / total
        ah_implied_a = raw_a / total

        # Edge AH vs 1X2: comparamos con prob implicita 1X2 normalizada
        prob_h_1x2 = features_cuotas.get('Prob_H', 0.0)
        prob_a_1x2 = features_cuotas.get('Prob_A', 0.0)
        total_1x2  = prob_h_1x2 + prob_a_1x2
        prob_h_norm = prob_h_1x2 / total_1x2 if total_1x2 > 0 else 0.5
        ah_edge_home = ah_implied_h - prob_h_norm

        # Confianza del mercado AH
        ah_fair = 1.909
        ah_market_conf = max(0.0, ah_fair - ah_cuota_h) if ah_cuota_h else 0.0

        return {
            'AH_Line': float(ah_line),
            'AH_Implied_Home': float(ah_implied_h),
            'AH_Implied_Away': float(ah_implied_a),
            'AH_Edge_Home': float(ah_edge_home),
            'AH_Market_Conf': float(ah_market_conf),
        }

    def _calcular_descanso_prediccion(self, local: str, visitante: str) -> dict:
        """
        Calcula features de descanso para predicción en vivo.

        Usa el histórico enriquecido para encontrar la fecha del último
        partido de cada equipo, y asume la fecha actual como fecha del
        partido a predecir. Si fbref_fixtures.csv está disponible, usa
        todas las competiciones; si no, solo datos PL.
        """
        import os
        from datetime import datetime

        defaults = {
            'HT_Days_Rest': 7.0, 'AT_Days_Rest': 7.0, 'Rest_Diff': 0.0,
            'HT_Had_Europa': 0.0, 'AT_Had_Europa': 0.0,
            'HT_Games_15d': 2.0, 'AT_Games_15d': 2.0,
            'Calendar_Congestion_Diff': 0.0,
        }

        df_enriq = getattr(self, '_df_enriquecido', None)
        if df_enriq is None or 'HT_Days_Rest' not in df_enriq.columns:
            return defaults

        hoy = pd.Timestamp(datetime.now().date())

        def _rest_para_equipo(equipo):
            partidos = df_enriq[
                (df_enriq['HomeTeam'] == equipo) | (df_enriq['AwayTeam'] == equipo)
            ]
            if len(partidos) == 0:
                return 7.0, 0.0, 2.0

            ultimo = partidos.iloc[-1]
            fecha_ultimo = pd.to_datetime(ultimo['Date'])
            days_rest = max(1.0, float((hoy - fecha_ultimo).days))

            # Had_Europa: revisar si el equipo tuvo partido europeo en últimos 4 días
            # desde el último partido del enriquecido
            prefix = 'HT' if ultimo['HomeTeam'] == equipo else 'AT'
            had_europa = float(ultimo.get(f'{prefix}_Had_Europa', 0.0))
            games_15d = float(ultimo.get(f'{prefix}_Games_15d', 2.0))

            return days_rest, had_europa, games_15d

        ht_rest, ht_europa, ht_games = _rest_para_equipo(local)
        at_rest, at_europa, at_games = _rest_para_equipo(visitante)

        return {
            'HT_Days_Rest': ht_rest,
            'AT_Days_Rest': at_rest,
            'Rest_Diff': ht_rest - at_rest,
            'HT_Had_Europa': ht_europa,
            'AT_Had_Europa': at_europa,
            'HT_Games_15d': ht_games,
            'AT_Games_15d': at_games,
            'Calendar_Congestion_Diff': ht_games - at_games,
        }

    # Mapeo de features del modelo al nombre de columna en el DataFrame enriquecido.
    # Formato: 'NombreFeatureModelo': ('columna_si_jugaba_home', 'columna_si_jugaba_away')
    # El parametro es_local indica si el equipo juega como local en el partido a predecir.
    _FEATURES_ENRIQUECIDAS_HT = {
        # features de tabla (HT = equipo que va a jugar de local)
        'Position':           ('HT_Position',          'AT_Position'),
        'Points':             ('HT_Points',             'AT_Points'),
        'Pressure':           ('HT_Pressure',           'AT_Pressure'),
        # forma/momentum (5 ultimos)
        'WinRate5':           ('HT_WinRate5',           'AT_WinRate5'),
        'HomeWinRate5':       ('HT_HomeWinRate5',        'AT_AwayWinRate5'),
        'Pts5':               ('HT_Pts5',               'AT_Pts5'),
        'GoalsFor5':          ('HT_GoalsFor5',          'AT_GoalsFor5'),
        'GoalsAgainst5':      ('HT_GoalsAgainst5',      'AT_GoalsAgainst5'),
        'HomeGoals5':         ('HT_HomeGoals5',         'AT_AwayGoals5'),
        'Streak':             ('HT_Streak',             'AT_Streak'),
        # multi-escala (window=10)
        'Pts3':               ('HT_Pts3',              'AT_Pts3'),
        'GoalsFor3':          ('HT_GoalsFor3',         'AT_GoalsFor3'),
        'xG_Avg_3':           ('HT_xG_Avg_3',          'AT_xG_Avg_3'),
        'Pts10':              ('HT_Pts10',              'AT_Pts10'),
        'GoalsFor10':         ('HT_GoalsFor10',         'AT_GoalsFor10'),
        'xG_Avg_10':          ('HT_xG_Avg_10',         'AT_xG_Avg_10'),
        'Form_Momentum':      ('HT_Form_Momentum',     'AT_Form_Momentum'),
        # EWM
        'Pts_EWM5':           ('HT_Pts_EWM5',          'AT_Pts_EWM5'),
        'GoalsFor_EWM5':      ('HT_GoalsFor_EWM5',     'AT_GoalsFor_EWM5'),
        'GoalsAgainst_EWM5':  ('HT_GoalsAgainst_EWM5', 'AT_GoalsAgainst_EWM5'),
        'ShotsTarget_EWM5':   ('HT_ShotsTarget_EWM5',  'AT_ShotsTarget_EWM5'),
        'xG_EWM5':            ('HT_xG_EWM5',           'AT_xG_EWM5'),
        'xGA_EWM5':           ('HT_xGA_EWM5',          'AT_xGA_EWM5'),
        # SoR
        'SoR5':               ('HT_SoR5',              'AT_SoR5'),
    }
    _FEATURES_ENRIQUECIDAS_AT = {
        # features de tabla (AT = equipo que va a jugar de visitante)
        'Position':           ('AT_Position',          'HT_Position'),
        'Points':             ('AT_Points',             'HT_Points'),
        'Pressure':           ('AT_Pressure',           'HT_Pressure'),
        # forma/momentum (5 ultimos)
        'WinRate5':           ('AT_WinRate5',           'HT_WinRate5'),
        'AwayWinRate5':       ('AT_AwayWinRate5',       'HT_HomeWinRate5'),
        'Pts5':               ('AT_Pts5',               'HT_Pts5'),
        'GoalsFor5':          ('AT_GoalsFor5',          'HT_GoalsFor5'),
        'GoalsAgainst5':      ('AT_GoalsAgainst5',      'HT_GoalsAgainst5'),
        'AwayGoals5':         ('AT_AwayGoals5',         'HT_HomeGoals5'),
        'Streak':             ('AT_Streak',             'HT_Streak'),
        # multi-escala (window=10)
        'Pts3':               ('AT_Pts3',              'HT_Pts3'),
        'GoalsFor3':          ('AT_GoalsFor3',         'HT_GoalsFor3'),
        'xG_Avg_3':           ('AT_xG_Avg_3',          'HT_xG_Avg_3'),
        'Pts10':              ('AT_Pts10',              'HT_Pts10'),
        'GoalsFor10':         ('AT_GoalsFor10',         'HT_GoalsFor10'),
        'xG_Avg_10':          ('AT_xG_Avg_10',         'HT_xG_Avg_10'),
        'Form_Momentum':      ('AT_Form_Momentum',     'HT_Form_Momentum'),
        # EWM
        'Pts_EWM5':           ('AT_Pts_EWM5',          'HT_Pts_EWM5'),
        'GoalsFor_EWM5':      ('AT_GoalsFor_EWM5',     'HT_GoalsFor_EWM5'),
        'GoalsAgainst_EWM5':  ('AT_GoalsAgainst_EWM5', 'HT_GoalsAgainst_EWM5'),
        'ShotsTarget_EWM5':   ('AT_ShotsTarget_EWM5',  'HT_ShotsTarget_EWM5'),
        'xG_EWM5':            ('AT_xG_EWM5',           'HT_xG_EWM5'),
        'xGA_EWM5':           ('AT_xGA_EWM5',          'HT_xGA_EWM5'),
        # SoR
        'SoR5':               ('AT_SoR5',              'HT_SoR5'),
    }

    def _obtener_stats_equipo(self, equipo: str, es_local: bool) -> dict | None:
        """Extrae las ultimas estadisticas conocidas de un equipo del historico enriquecido."""
        df = getattr(self, '_df_enriquecido', self._df_historico)
        partidos = df[(df['HomeTeam'] == equipo) | (df['AwayTeam'] == equipo)]

        if len(partidos) == 0:
            return None

        ultimo = partidos.iloc[-1]
        jugaba_local = ultimo['HomeTeam'] == equipo

        stats = {}

        if es_local:
            stats['AvgGoals'] = ultimo['HT_AvgGoals'] if jugaba_local else ultimo['AT_AvgGoals']
            stats['AvgShotsTarget'] = ultimo['HT_AvgShotsTarget'] if jugaba_local else ultimo['AT_AvgShotsTarget']
            stats['Form_W'] = ultimo['HT_Form_W'] if jugaba_local else ultimo['AT_Form_W']
            stats['Form_D'] = ultimo['HT_Form_D'] if jugaba_local else ultimo['AT_Form_D']
            stats['Form_L'] = ultimo['HT_Form_L'] if jugaba_local else ultimo['AT_Form_L']
            if 'HT_xG_Avg' in ultimo.index:
                stats['xG_Avg'] = ultimo['HT_xG_Avg'] if jugaba_local else ultimo['AT_xG_Avg']
            if 'HT_xGA_Avg' in ultimo.index:
                stats['xGA_Avg'] = ultimo['HT_xGA_Avg'] if jugaba_local else ultimo['AT_xGA_Avg']
            # xG global (todas las venues)
            if 'HT_xG_Global' in ultimo.index:
                stats['xG_Global'] = ultimo['HT_xG_Global'] if jugaba_local else ultimo['AT_xG_Global']
            if 'HT_xGA_Global' in ultimo.index:
                stats['xGA_Global'] = ultimo['HT_xGA_Global'] if jugaba_local else ultimo['AT_xGA_Global']
            # Features enriquecidas (tabla + forma/momentum + multi-escala)
            for feat_key, (col_home, col_away) in self._FEATURES_ENRIQUECIDAS_HT.items():
                col = col_home if jugaba_local else col_away
                if col in ultimo.index:
                    stats[feat_key] = float(ultimo[col]) if pd.notna(ultimo[col]) else 0.0
        else:
            stats['AvgGoals'] = ultimo['AT_AvgGoals'] if not jugaba_local else ultimo['HT_AvgGoals']
            stats['AvgShotsTarget'] = ultimo['AT_AvgShotsTarget'] if not jugaba_local else ultimo['HT_AvgShotsTarget']
            stats['Form_W'] = ultimo['AT_Form_W'] if not jugaba_local else ultimo['HT_Form_W']
            stats['Form_D'] = ultimo['AT_Form_D'] if not jugaba_local else ultimo['HT_Form_D']
            stats['Form_L'] = ultimo['AT_Form_L'] if not jugaba_local else ultimo['HT_Form_L']
            if 'AT_xG_Avg' in ultimo.index:
                stats['xG_Avg'] = ultimo['AT_xG_Avg'] if not jugaba_local else ultimo['HT_xG_Avg']
            if 'AT_xGA_Avg' in ultimo.index:
                stats['xGA_Avg'] = ultimo['AT_xGA_Avg'] if not jugaba_local else ultimo['HT_xGA_Avg']
            # xG global (todas las venues)
            if 'AT_xG_Global' in ultimo.index:
                stats['xG_Global'] = ultimo['AT_xG_Global'] if not jugaba_local else ultimo['HT_xG_Global']
            if 'AT_xGA_Global' in ultimo.index:
                stats['xGA_Global'] = ultimo['AT_xGA_Global'] if not jugaba_local else ultimo['HT_xGA_Global']
            # Features enriquecidas (tabla + forma/momentum + multi-escala)
            for feat_key, (col_away_col, col_home_col) in self._FEATURES_ENRIQUECIDAS_AT.items():
                col = col_away_col if not jugaba_local else col_home_col
                if col in ultimo.index:
                    stats[feat_key] = float(ultimo[col]) if pd.notna(ultimo[col]) else 0.0

        return stats

    # -------------------------------------------------------------------------
    # METODOS PRIVADOS — GENERACION DE REPORTES
    # -------------------------------------------------------------------------

    def _generar_pdf(self, resultados: list[dict], numero_jornada: int, ruta: str) -> str | None:
        """Genera el reporte PDF."""
        if not PDF_AVAILABLE:
            print("No se puede generar PDF. Instala: pip install fpdf2")
            return None

        print("\nGenerando reporte PDF...")

        bankroll = self._bankroll

        pdf = _PDFReporte(numero_jornada)
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Resumen ejecutivo
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Resumen Ejecutivo', 0, 1)
        pdf.set_font('Arial', '', 10)

        alta = len([r for r in resultados if r['confianza'] >= 0.60])
        media = len([r for r in resultados if 0.50 <= r['confianza'] < 0.60])
        baja = len([r for r in resultados if r['confianza'] < 0.50])

        pdf.multi_cell(0, 5,
            f'Total de partidos analizados: {len(resultados)}\n'
            f'Modelo: Hibrido (Rendimiento + Cuotas)\n'
            f'Accuracy del modelo: 51%\n'
            f'Bankroll configurado: {bankroll}{MONEDA}\n\n'
            f'Distribucion por confianza:\n'
            f'  - Alta confianza (>=60%): {alta} partidos\n'
            f'  - Media confianza (50-60%): {media} partidos\n'
            f'  - Baja confianza (<50%): {baja} partidos'
        )
        pdf.ln(3)

        value_bets_count = len([r for r in resultados if r['diferencia_valor'] > 0.03])
        pdf.set_font('Arial', 'B', 11)
        pdf.set_fill_color(255, 235, 59)
        pdf.cell(0, 6, f'VALUE BETS DETECTADAS: {value_bets_count}', 0, 1, 'C', True)
        pdf.set_font('Arial', '', 9)
        if value_bets_count > 0:
            pdf.multi_cell(0, 4,
                f'Se encontraron {value_bets_count} partidos con edge > 3% donde el modelo ve '
                'mas probabilidad que el mercado. Estas son las apuestas recomendadas.')
        else:
            pdf.multi_cell(0, 4,
                'No se detectaron value bets en esta jornada. Es normal. '
                'La paciencia es clave en value betting.')
        pdf.ln(5)

        # Predicciones detalladas
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Predicciones Detalladas por Partido', 0, 1)

        for i, r in enumerate(resultados, 1):
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 8, f"{i}. {r['local']} vs {r['visitante']}", 0, 1)

            nivel_conf = "ALTA" if r['confianza'] >= 0.60 else "MEDIA" if r['confianza'] >= 0.50 else "BAJA - EVITAR"

            if r['prediccion'] == 'Local':
                emoji = '[LOCAL]'
                pred_texto = f"Victoria {r['local']}"
            elif r['prediccion'] == 'Empate':
                emoji = '[EMPATE]'
                pred_texto = "Empate"
            else:
                emoji = '[VISIT]'
                pred_texto = f"Victoria {r['visitante']}"

            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 5, f"   {emoji} {pred_texto}", 0, 1)
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 5, f"   Confianza: {r['confianza']:.1%} - Nivel: {nivel_conf}", 0, 1)
            pdf.cell(0, 5,
                f"   Probabilidades: Local {r['prob_local']:.1%} | "
                f"Empate {r['prob_empate']:.1%} | Visitante {r['prob_visitante']:.1%}", 0, 1)
            pdf.set_font('Arial', 'I', 9)
            pdf.cell(0, 5,
                f"   Mercado: Local {r['prob_mercado_local']:.1%} | "
                f"Empate {r['prob_mercado_empate']:.1%} | Visitante {r['prob_mercado_visitante']:.1%}", 0, 1)
            pdf.set_font('Arial', '', 10)

            dif = r['diferencia_valor']
            if dif > 0.08:
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 5, f"   ** GRAN OPORTUNIDAD: Modelo ve +{dif:.1%} mas que el mercado **", 0, 1)
                pdf.set_font('Arial', '', 10)
            elif dif > 0.05:
                pdf.set_font('Arial', 'B', 9)
                pdf.cell(0, 5, f"   * Oportunidad: Modelo ve +{dif:.1%} mas que el mercado *", 0, 1)
                pdf.set_font('Arial', '', 10)
            elif dif < -0.05:
                pdf.set_font('Arial', 'I', 9)
                pdf.cell(0, 5, f"   Precaucion: Mercado ve mas probabilidad ({dif:.1%})", 0, 1)
                pdf.set_font('Arial', '', 10)

            pdf.cell(0, 5, f"   Cuotas: {r['cuota_h']:.2f} - {r['cuota_d']:.2f} - {r['cuota_a']:.2f}", 0, 1)
            pdf.cell(0, 5,
                f"   Forma: {r['local']} ({r['forma_local']}) | "
                f"{r['visitante']} ({r['forma_visitante']})", 0, 1)

            # Mercados binarios
            p_ou = r.get('prob_over25')
            p_tc = r.get('prob_over35_cards')
            p_cn = r.get('prob_over95_corners')
            if any(v is not None for v in [p_ou, p_tc, p_cn]):
                pdf.set_font('Arial', 'B', 9)
                pdf.cell(0, 5, '   Mercados adicionales:', 0, 1)
                pdf.set_font('Arial', '', 9)
                if p_ou is not None:
                    side = "Over" if p_ou >= 0.5 else "Under"
                    pdf.cell(0, 4, f"     Goles O/U 2.5: {side} ({p_ou:.1%})", 0, 1)
                if p_tc is not None:
                    side = "Over" if p_tc >= 0.5 else "Under"
                    pdf.cell(0, 4, f"     Tarjetas O/U 3.5: {side} ({p_tc:.1%})", 0, 1)
                if p_cn is not None:
                    side = "Over" if p_cn >= 0.5 else "Under"
                    pdf.cell(0, 4, f"     Corners O/U 9.5: {side} ({p_cn:.1%})", 0, 1)
                pdf.set_font('Arial', '', 10)

            pdf.ln(3)

            if i % 3 == 0 and i < len(resultados):
                pdf.add_page()

        # Pagina de recomendaciones
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Recomendaciones Finales', 0, 1)

        # Value bets
        pdf.set_font('Arial', 'B', 12)
        pdf.set_fill_color(255, 235, 59)
        pdf.cell(0, 8, 'VALUE BETS DETECTADAS (Edge > 3%)', 0, 1, 'L', True)
        pdf.ln(2)
        pdf.set_font('Arial', 'I', 9)
        pdf.multi_cell(0, 4,
            'Un Value Bet es cuando el modelo ve significativamente mas probabilidad '
            'que el mercado (edge > 3%). Solo en estos casos tiene sentido apostar a largo plazo.')
        pdf.ln(2)

        value_bets = []
        for r in resultados:
            if r['prediccion'] == 'Local':
                edge = r['prob_local'] - r['prob_mercado_local']
                cuota = r['cuota_h']
                prob = r['prob_local']
            elif r['prediccion'] == 'Empate':
                edge = r['prob_empate'] - r['prob_mercado_empate']
                cuota = r['cuota_d']
                prob = r['prob_empate']
            else:
                edge = r['prob_visitante'] - r['prob_mercado_visitante']
                cuota = r['cuota_a']
                prob = r['prob_visitante']

            if edge > 0.03:
                value_bets.append((r, edge, cuota, prob))

        if value_bets:
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 6, f'SE DETECTARON {len(value_bets)} VALUE BETS:', 0, 1)
            pdf.set_font('Arial', '', 9)
            resumen_bets = []

            for i, (r, edge, cuota, prob) in enumerate(sorted(value_bets, key=lambda x: x[1], reverse=True), 1):
                b = cuota - 1
                kelly_full = (b * prob - (1 - prob)) / b
                kelly_safe = min(max(0, kelly_full * 0.25), 0.10)
                stake = bankroll * kelly_safe
                ev = stake * (cuota * prob - 1)
                roi = (ev / stake) * 100 if stake > 0 else 0
                resumen_bets.append({'stake': stake, 'ev': ev, 'roi': roi})

                pred_txt = r['local'] if r['prediccion'] == 'Local' else r['visitante'] if r['prediccion'] == 'Visitante' else 'Empate'
                pdf.set_font('Arial', 'B', 9)
                pdf.cell(0, 5, f"{i}. {r['local']} vs {r['visitante']}", 0, 1)
                pdf.set_font('Arial', '', 9)
                pdf.cell(0, 4, f"   Apostar: {r['prediccion']} ({pred_txt}) | Cuota: {cuota:.2f}", 0, 1)
                pdf.cell(0, 4, f"   Edge: +{edge:.1%}", 0, 1)
                pdf.cell(0, 4, f"   Expected Value: +{ev:.2f}{MONEDA} (ROI: +{roi:.1f}%)", 0, 1)
                pdf.cell(0, 4, f"   Stake recomendado (Kelly): {stake:.2f}{MONEDA} ({stake/bankroll*100:.1f}% bankroll)", 0, 1)
                pdf.cell(0, 4, f"   Modelo {prob:.1%} vs Mercado {1/cuota:.1%}", 0, 1)
                pdf.ln(2)

            total_stake = sum(b['stake'] for b in resumen_bets)
            total_ev = sum(b['ev'] for b in resumen_bets)
            roi_total = (total_ev / total_stake) * 100 if total_stake > 0 else 0

            pdf.set_font('Arial', 'B', 9)
            pdf.cell(0, 5, 'RESUMEN:', 0, 1)
            pdf.set_font('Arial', '', 9)
            pdf.cell(0, 4, f"Total a invertir: {total_stake:.2f}{MONEDA} ({total_stake/bankroll*100:.1f}% del bankroll)", 0, 1)
            pdf.cell(0, 4, f"Ganancia esperada: +{total_ev:.2f}{MONEDA}", 0, 1)
            pdf.cell(0, 4, f"ROI esperado: +{roi_total:.1f}%", 0, 1)
        else:
            pdf.set_font('Arial', '', 9)
            pdf.cell(0, 5, 'NO HAY VALUE BETS EN ESTA JORNADA', 0, 1)
            pdf.multi_cell(0, 4,
                'Ningun partido tiene edge > 3%. Es normal que no todas las jornadas tengan value bets.')

        pdf.output(ruta)
        print(f"PDF generado: {ruta}\n")
        return ruta

    def _generar_excel(self, resultados: list[dict], ruta: str) -> str | None:
        """Genera el reporte Excel."""
        print("\nGenerando reporte Excel...")

        df = pd.DataFrame(resultados)
        df['Partido'] = df['local'] + ' vs ' + df['visitante']
        df['Prediccion'] = df['prediccion']
        df['Confianza'] = df['confianza'].apply(lambda x: f"{x:.1%}")
        df['Prob Local'] = df['prob_local'].apply(lambda x: f"{x:.1%}")
        df['Prob Empate'] = df['prob_empate'].apply(lambda x: f"{x:.1%}")
        df['Prob Visit'] = df['prob_visitante'].apply(lambda x: f"{x:.1%}")
        df['Cuotas'] = df.apply(lambda x: f"{x['cuota_h']:.2f}-{x['cuota_d']:.2f}-{x['cuota_a']:.2f}", axis=1)
        df['Oportunidad'] = df['diferencia_valor'].apply(lambda x: 'SI' if abs(x) > 0.08 else 'No')

        # Mercados binarios
        if 'prob_over25' in df.columns:
            df['O/U 2.5'] = df['prob_over25'].apply(
                lambda x: f"{'Over' if x >= 0.5 else 'Under'} {x:.0%}" if pd.notna(x) else 'N/A')
        if 'prob_over35_cards' in df.columns:
            df['Tarj 3.5'] = df['prob_over35_cards'].apply(
                lambda x: f"{'Over' if x >= 0.5 else 'Under'} {x:.0%}" if pd.notna(x) else 'N/A')
        if 'prob_over95_corners' in df.columns:
            df['Corn 9.5'] = df['prob_over95_corners'].apply(
                lambda x: f"{'Over' if x >= 0.5 else 'Under'} {x:.0%}" if pd.notna(x) else 'N/A')

        cols = [
            'Partido', 'Prediccion', 'Confianza',
            'Prob Local', 'Prob Empate', 'Prob Visit',
            'Cuotas', 'Oportunidad', 'forma_local', 'forma_visitante',
        ]
        for extra in ['O/U 2.5', 'Tarj 3.5', 'Corn 9.5']:
            if extra in df.columns:
                cols.append(extra)
        df_export = df[cols]
        try:
            df_export.to_excel(ruta, index=False)
            print(f"Excel generado: {ruta}\n")
            return ruta
        except ImportError:
            # Fallback a CSV si openpyxl no esta instalado
            ruta_csv = ruta.replace('.xlsx', '.csv')
            df_export.to_csv(ruta_csv, index=False)
            print(f"Excel no disponible (instalar openpyxl). CSV generado: {ruta_csv}\n")
            return ruta_csv
