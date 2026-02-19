"""
Predictor de Jornada Completa con Reporte PDF
Predice múltiples partidos y genera un reporte profesional
CON VALUE BETTING PROFESIONAL
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

from config import (
    ARCHIVO_FEATURES,
    ARCHIVO_MODELO,
    ARCHIVO_FEATURES_PKL,
    ARCHIVO_MODELO_VB,
    ARCHIVO_FEATURES_VB,
    RUTA_MODELOS,
    FACTOR_CONSERVADOR,
    UMBRAL_EDGE_MINIMO,
    CUOTA_MAXIMA,
    PROBABILIDAD_MINIMA,
    KELLY_FRACTION,
    STAKE_MAXIMO_PCT,
    BANKROLL_DEFAULT,
    MONEDA,
)

# Carpeta de salida para reportes
RUTA_REPORTES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reportes')
os.makedirs(RUTA_REPORTES, exist_ok=True)
from utils import calcular_h2h_features

# Importar FPDF para generar PDFs
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️  FPDF no disponible. Instala con: pip install fpdf2")

try:
    from sistema_expected_value import calcular_ev, kelly_criterion, analizar_apuesta
    EV_AVAILABLE = True
except ImportError:
    EV_AVAILABLE = False

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

RUTA_DATOS = ARCHIVO_FEATURES
BANKROLL = BANKROLL_DEFAULT

# Número de jornada actual — actualizar antes de ejecutar
NUMERO_JORNADA = 15

# ============================================================================
# TUS PARTIDOS DE LA JORNADA
# Actualiza esta lista con los partidos y cuotas de la jornada actual.
# ============================================================================

partidos_jornada = [
    # Martes 2 de Diciembre
    {'local': 'Bournemouth', 'visitante': 'Everton', 'cuota_h': 2.30, 'cuota_d': 3.40, 'cuota_a': 3.10}, # Partido parejo
    {'local': 'Fulham', 'visitante': 'Man City', 'cuota_h': 5.00, 'cuota_d': 4.25, 'cuota_a': 1.67}, # City favorito visitante
    {'local': 'Newcastle', 'visitante': 'Tottenham', 'cuota_h': 2.50, 'cuota_d': 3.60, 'cuota_a': 2.70}, # Muy parejo en St James' Park

    # Miércoles 3 de Diciembre
    {'local': 'Arsenal', 'visitante': 'Brentford', 'cuota_h': 1.30, 'cuota_d': 5.50, 'cuota_a': 10.00}, # Arsenal muy favorito
    {'local': 'Brighton', 'visitante': 'Aston Villa', 'cuota_h': 2.60, 'cuota_d': 3.50, 'cuota_a': 2.60},
    {'local': 'Burnley', 'visitante': 'Crystal Palace', 'cuota_h': 2.90, 'cuota_d': 3.30, 'cuota_a': 2.45},
    {'local': 'Wolves', 'visitante': "Nott'm Forest", 'cuota_h': 2.35, 'cuota_d': 3.40, 'cuota_a': 3.00},
    {'local': 'Leeds', 'visitante': 'Chelsea', 'cuota_h': 4.50, 'cuota_d': 4.00, 'cuota_a': 1.75}, # Chelsea favorito
    {'local': 'Liverpool', 'visitante': 'Sunderland', 'cuota_h': 1.25, 'cuota_d': 6.50, 'cuota_a': 11.00}, # Liverpool muy favorito

    # Jueves 4 de Diciembre
    {'local': 'Man United', 'visitante': 'West Ham', 'cuota_h': 1.65, 'cuota_d': 4.20, 'cuota_a': 5.00} # Man Utd favorito
]
# ============================================================================
# CAPA 1: AJUSTE CONSERVADOR DE PROBABILIDADES
# ============================================================================

def ajustar_probabilidades_conservador(probabilidades, factor=FACTOR_CONSERVADOR):
    """
    CAPA 1: Ajusta probabilidades del modelo de forma conservadora.
    
    Mezcla las probabilidades del modelo con una distribución uniforme
    para reducir el overconfidence.
    """
    uniforme = np.array([1/3, 1/3, 1/3])
    ajustadas = factor * probabilidades + (1 - factor) * uniforme
    ajustadas = ajustadas / ajustadas.sum()
    return ajustadas


# ============================================================================
# CAPA 2 Y 3: FILTROS + KELLY CON TOPES
# ============================================================================

def calcular_value_bet_con_3_capas(prob_modelo, cuota_mercado, bankroll):
    """
    Calcula value bet con las 3 capas de protección integradas.
    
    Returns:
        dict con info completa o None si no pasa filtros
    """
    
    # Probabilidad del mercado
    prob_mercado = 1 / cuota_mercado
    edge = prob_modelo - prob_mercado
    edge_pct = edge / prob_mercado if prob_mercado > 0 else 0
    
    # CAPA 2: FILTROS
    if edge_pct < UMBRAL_EDGE_MINIMO:
        return None
    if cuota_mercado > CUOTA_MAXIMA:
        return None
    if prob_modelo < PROBABILIDAD_MINIMA:
        return None
    
    # CAPA 3: KELLY CON TOPES
    b = cuota_mercado - 1
    kelly_full = (b * prob_modelo - (1 - prob_modelo)) / b
    kelly_full = max(0, kelly_full)
    kelly_fraction = kelly_full * KELLY_FRACTION
    kelly_fraction = min(kelly_fraction, STAKE_MAXIMO_PCT)
    
    stake = bankroll * kelly_fraction
    
    prob_perder = 1 - prob_modelo
    ganancia_si_gana = stake * (cuota_mercado - 1)
    ev = (prob_modelo * ganancia_si_gana) - (prob_perder * stake)
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
        'roi': roi
    }

# ============================================================================
# CARGAR MODELO Y DATOS
# ============================================================================

def cargar_sistema():
    """Carga el modelo y datos históricos."""
    
    print("="*70)
    print("🔄 CARGANDO SISTEMA DE PREDICCIÓN")
    print("="*70)
    
    # Prioridad: modelo con cuotas (02) → modelo sin cuotas (03)
    # El modelo con cuotas es más informativo; el sin cuotas es el fallback
    # para verificar edge estructural independiente del mercado.
    if os.path.exists(ARCHIVO_MODELO):
        modelo_file = ARCHIVO_MODELO       # 02_entrenar_modelo.py
        features_file = ARCHIVO_FEATURES_PKL
    elif os.path.exists(ARCHIVO_MODELO_VB):
        modelo_file = ARCHIVO_MODELO_VB    # 03_entrenar_sin_cuotas.py
        features_file = ARCHIVO_FEATURES_VB
    else:
        print("❌ No se encontró el modelo")
        print("   Ejecuta: python 02_entrenar_modelo.py")
        return None, None, None
    
    modelo = joblib.load(modelo_file)
    features = joblib.load(features_file) if os.path.exists(features_file) else None
    df_historico = pd.read_csv(RUTA_DATOS) if os.path.exists(RUTA_DATOS) else None
    
    if features is None or df_historico is None:
        print("❌ No se pudieron cargar features o datos históricos")
        return None, None, None

    print(f"✅ Modelo cargado: {os.path.basename(modelo_file)}")
    print(f"✅ Features: {len(features)}")
    print(f"✅ Histórico: {len(df_historico)} partidos\n")
    
    return modelo, features, df_historico


# ============================================================================
# FUNCIONES DE PREDICCIÓN
# ============================================================================

def obtener_stats_equipo(equipo, df, es_local=True):
    """Obtiene estadísticas del equipo."""
    
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
        
        # Agregar xG si está disponible
        if 'HT_xG_Avg' in ultimo.index:
            stats['xG_Avg'] = ultimo['HT_xG_Avg'] if jugaba_local else ultimo['AT_xG_Avg']
        if 'HT_xGA_Avg' in ultimo.index:
            stats['xGA_Avg'] = ultimo['HT_xGA_Avg'] if jugaba_local else ultimo['AT_xGA_Avg']
    else:
        stats['AvgGoals'] = ultimo['AT_AvgGoals'] if not jugaba_local else ultimo['HT_AvgGoals']
        stats['AvgShotsTarget'] = ultimo['AT_AvgShotsTarget'] if not jugaba_local else ultimo['HT_AvgShotsTarget']
        stats['Form_W'] = ultimo['AT_Form_W'] if not jugaba_local else ultimo['HT_Form_W']
        stats['Form_D'] = ultimo['AT_Form_D'] if not jugaba_local else ultimo['HT_Form_D']
        stats['Form_L'] = ultimo['AT_Form_L'] if not jugaba_local else ultimo['HT_Form_L']
        
        # Agregar xG si está disponible
        if 'AT_xG_Avg' in ultimo.index:
            stats['xG_Avg'] = ultimo['AT_xG_Avg'] if not jugaba_local else ultimo['HT_xG_Avg']
        if 'AT_xGA_Avg' in ultimo.index:
            stats['xGA_Avg'] = ultimo['AT_xGA_Avg'] if not jugaba_local else ultimo['HT_xGA_Avg']
    
    return stats


def transformar_cuotas(cuota_h, cuota_d, cuota_a):
    """
    Transforma cuotas de apertura en features de probabilidad.

    Los nombres de las features generadas coinciden exactamente con los
    producidos por utils.agregar_features_cuotas_derivadas() durante el
    entrenamiento. Las probabilidades NO se normalizan (consistente con training).
    """
    prob_h = 1 / cuota_h
    prob_d = 1 / cuota_d
    prob_a = 1 / cuota_a

    prob_max = max(prob_h, prob_d, prob_a)
    prob_min = min(prob_h, prob_d, prob_a)

    return {
        'Prob_H': prob_h,
        'Prob_D': prob_d,
        'Prob_A': prob_a,
        # No hay cuotas de cierre en predicción en vivo → movimiento = 0
        'Prob_Move_H': 0.0,
        'Prob_Move_D': 0.0,
        'Prob_Move_A': 0.0,
        'Market_Move_Strength': 0.0,
        'Prob_Spread': prob_max - prob_min,
        'Market_Confidence': prob_max - (1 / 3),
        'Home_Advantage_Prob': prob_h - prob_a,
    }


def predecir_partido(partido, modelo, features, df):
    """Predice un partido y retorna resultados."""
    
    local = partido['local']
    visitante = partido['visitante']
    cuota_h = partido['cuota_h']
    cuota_d = partido['cuota_d']
    cuota_a = partido['cuota_a']
    
    # Obtener stats
    stats_local = obtener_stats_equipo(local, df, es_local=True)
    stats_visitante = obtener_stats_equipo(visitante, df, es_local=False)
    
    if stats_local is None or stats_visitante is None:
        return None
    
    # Transformar cuotas en features con nombres canónicos (idénticos a training)
    features_cuotas = transformar_cuotas(cuota_h, cuota_d, cuota_a)
    
    # Construir features
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
    # Agregar todas las features de cuotas (nombres canónicos)
    datos.update(features_cuotas)

    # Agregar xG si está disponible
    if 'xG_Avg' in stats_local and 'xG_Avg' in stats_visitante:
        datos['HT_xG_Avg'] = stats_local.get('xG_Avg', 0)
        datos['AT_xG_Avg'] = stats_visitante.get('xG_Avg', 0)
        datos['xG_Diff'] = datos['HT_xG_Avg'] - datos['AT_xG_Avg']
        datos['xG_Total'] = datos['HT_xG_Avg'] + datos['AT_xG_Avg']
    
    if 'xGA_Avg' in stats_local and 'xGA_Avg' in stats_visitante:
        datos['HT_xGA_Avg'] = stats_local.get('xGA_Avg', 0)
        datos['AT_xGA_Avg'] = stats_visitante.get('xGA_Avg', 0)
    
    # H2H features (incluye H2H_Available y features derivadas)
    try:
        h2h_features = calcular_h2h_features(
            df=df,
            equipo_local=local,
            equipo_visitante=visitante,
            fecha_limite=None,
        )
        datos.update(h2h_features)
    except Exception:
        # Valores neutros si falla H2H
        datos['H2H_Available'] = 0
    
    datos_filtrado = {k: v for k, v in datos.items() if k in features}
    partido_df = pd.DataFrame([datos_filtrado], columns=features).fillna(0)
    
    # Predicción ORIGINAL del modelo
    probs_originales = modelo.predict_proba(partido_df)[0]
    
    # CAPA 1: AJUSTAR PROBABILIDADES (CONSERVADOR)
    probs = ajustar_probabilidades_conservador(probs_originales)
    
    # Calcular edge para la opción predicha
    idx_prediccion = np.argmax(probs)
    if idx_prediccion == 0:  # Local
        diferencia_valor_calc = probs[0] - features_cuotas['Prob_H']
    elif idx_prediccion == 1:  # Empate
        diferencia_valor_calc = probs[1] - features_cuotas['Prob_D']
    else:  # Visitante
        diferencia_valor_calc = probs[2] - features_cuotas['Prob_A']
    
    resultado = {
        'local': local,
        'visitante': visitante,
        'cuota_h': cuota_h,
        'cuota_d': cuota_d,
        'cuota_a': cuota_a,
        'prob_local_original': probs_originales[0],
        'prob_empate_original': probs_originales[1],
        'prob_visitante_original': probs_originales[2],
        'prob_local': probs[0],
        'prob_empate': probs[1],
        'prob_visitante': probs[2],
        'prediccion': ['Local', 'Empate', 'Visitante'][idx_prediccion],
        'confianza': np.max(probs),
        'prob_mercado_local': features_cuotas['Prob_H'],
        'prob_mercado_empate': features_cuotas['Prob_D'],
        'prob_mercado_visitante': features_cuotas['Prob_A'],
        'forma_local': f"{stats_local['Form_W']:.0f}W-{stats_local['Form_D']:.0f}D-{stats_local['Form_L']:.0f}L",
        'forma_visitante': f"{stats_visitante['Form_W']:.0f}W-{stats_visitante['Form_D']:.0f}D-{stats_visitante['Form_L']:.0f}L",
        'diferencia_valor': diferencia_valor_calc  # Ahora es el edge real
    }
    
    return resultado

# ============================================================================
# GENERAR PDF
# ============================================================================

class PDFReporte(FPDF):
    """Clase personalizada para generar el PDF."""
    
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, f'Premier League - Predicciones Jornada {NUMERO_JORNADA}', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 5, f'Generado: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')


def generar_pdf(resultados, nombre_archivo='predicciones_jornada.pdf'):
    """Genera un PDF con todas las predicciones."""
    
    if not PDF_AVAILABLE:
        print("⚠️  No se puede generar PDF sin fpdf")
        return None
    
    print("\n📄 Generando reporte PDF...")
    
    pdf = PDFReporte()
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
        f'Bankroll configurado: {BANKROLL}{MONEDA}\n\n'
        f'Distribucion por confianza:\n'
        f'  - Alta confianza (>=60%): {alta} partidos\n'
        f'  - Media confianza (50-60%): {media} partidos\n'
        f'  - Baja confianza (<50%): {baja} partidos'
    )
    pdf.ln(3)
    
    # Resumen de Value Bets
    value_bets_count = len([r for r in resultados if r['diferencia_valor'] > 0.03])
    
    pdf.set_font('Arial', 'B', 11)
    pdf.set_fill_color(255, 235, 59)  # Fondo amarillo
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
    
    # Análisis de oportunidades
    oportunidades = []
    for r in resultados:
        if r['prediccion'] == 'Local':
            dif = r['prob_local'] - r['prob_mercado_local']
        elif r['prediccion'] == 'Empate':
            dif = r['prob_empate'] - r['prob_mercado_empate']
        else:
            dif = r['prob_visitante'] - r['prob_mercado_visitante']
        
        if dif > 0.05:
            oportunidades.append((r, dif))
    
    if oportunidades:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, f'Oportunidades de Valor Detectadas: {len(oportunidades)}', 0, 1)
        pdf.set_font('Arial', '', 9)
        pdf.multi_cell(0, 5, 'Partidos donde el modelo ve significativamente mas probabilidad que el mercado (>5%)')
        pdf.ln(3)
    
    # Predicciones por partido
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Predicciones Detalladas por Partido', 0, 1)
    
    for i, r in enumerate(resultados, 1):
        # Encabezado del partido
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, f"{i}. {r['local']} vs {r['visitante']}", 0, 1)
        
        # Nivel de confianza
        if r['confianza'] >= 0.60:
            nivel_conf = "ALTA"
        elif r['confianza'] >= 0.50:
            nivel_conf = "MEDIA"
        else:
            nivel_conf = "BAJA - EVITAR"
        
        # Predicción
        pdf.set_font('Arial', '', 10)
        
        if r['prediccion'] == 'Local':
            emoji = '[LOCAL]'
            prediccion_texto = f"Victoria {r['local']}"
        elif r['prediccion'] == 'Empate':
            emoji = '[EMPATE]'
            prediccion_texto = "Empate"
        else:
            emoji = '[VISIT]'
            prediccion_texto = f"Victoria {r['visitante']}"
        
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 5, f"   {emoji} {prediccion_texto}", 0, 1)
        pdf.set_font('Arial', '', 10)
        
        # Confianza del modelo
        pdf.cell(0, 5, f"   Confianza del modelo: {r['confianza']:.1%} - Nivel: {nivel_conf}", 0, 1)
        
        # Probabilidades
        pdf.cell(0, 5, f"   Probabilidades: Local {r['prob_local']:.1%} | Empate {r['prob_empate']:.1%} | Visitante {r['prob_visitante']:.1%}", 0, 1)
        
        # Comparación con mercado
        pdf.set_font('Arial', 'I', 9)
        pdf.cell(0, 5, f"   Mercado espera: Local {r['prob_mercado_local']:.1%} | Empate {r['prob_mercado_empate']:.1%} | Visitante {r['prob_mercado_visitante']:.1%}", 0, 1)
        pdf.set_font('Arial', '', 10)
        
        # Análisis de valor
        if r['prediccion'] == 'Local':
            dif = r['prob_local'] - r['prob_mercado_local']
        elif r['prediccion'] == 'Empate':
            dif = r['prob_empate'] - r['prob_mercado_empate']
        else:
            dif = r['prob_visitante'] - r['prob_mercado_visitante']
        
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
        
        # Cuotas
        pdf.cell(0, 5, f"   Cuotas: {r['cuota_h']:.2f} - {r['cuota_d']:.2f} - {r['cuota_a']:.2f}", 0, 1)
        
        # Forma
        pdf.cell(0, 5, f"   Forma reciente: {r['local']} ({r['forma_local']}) | {r['visitante']} ({r['forma_visitante']})", 0, 1)
        
        pdf.ln(3)
        
        # Nueva página cada 3 partidos
        if i % 3 == 0 and i < len(resultados):
            pdf.add_page()
    
    # Página final con recomendaciones
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Recomendaciones Finales', 0, 1)
    
    # NUEVA SECCIÓN: VALUE BETTING
    pdf.set_font('Arial', 'B', 12)
    pdf.set_fill_color(255, 235, 59)  # Fondo amarillo
    pdf.cell(0, 8, 'VALUE BETS DETECTADAS (Edge > 3%)', 0, 1, 'L', True)
    pdf.ln(2)
    
    pdf.set_font('Arial', 'I', 9)
    pdf.multi_cell(0, 4, 
        'Que es un Value Bet? Es cuando tu modelo ve significativamente mas probabilidad '
        'que el mercado (edge > 3%). Solo en estos casos tiene sentido apostar a largo plazo.')
    pdf.ln(2)
    
    # Calcular value bets (edge > 3%)
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
        
        if edge > 0.03:  # 3%
            value_bets.append((r, edge, cuota, prob))
    
    if value_bets:
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, f'SE DETECTARON {len(value_bets)} VALUE BETS:', 0, 1)
        pdf.set_font('Arial', '', 9)
        
        value_bets_sorted = sorted(value_bets, key=lambda x: x[1], reverse=True)
        
        resumen_bets = []
        for i, (r, edge, cuota, prob) in enumerate(value_bets_sorted, 1):
            # Calcular Kelly
            b = cuota - 1
            kelly_full = (b * prob - (1-prob)) / b
            kelly_safe = max(0, kelly_full * 0.25)
            kelly_safe = min(kelly_safe, 0.10)
            stake = BANKROLL * kelly_safe
            # Calcular EV y ROI usando stake real
            ev = stake * (cuota * prob - 1)
            roi = (ev / stake) * 100 if stake > 0 else 0
            resumen_bets.append({'stake': stake, 'ev': ev, 'roi': roi})

            pred_txt = f"{r['local']}" if r['prediccion']=='Local' else f"{r['visitante']}" if r['prediccion']=='Visitante' else 'Empate'

            pdf.set_font('Arial', 'B', 9)
            pdf.cell(0, 5, f"{i}. {r['local']} vs {r['visitante']}", 0, 1)
            pdf.set_font('Arial', '', 9)
            pdf.cell(0, 4, f"   Apostar: {r['prediccion']} ({pred_txt}) | Cuota: {cuota:.2f}", 0, 1)
            pdf.cell(0, 4, f"   Edge: +{edge:.1%} (Modelo ve {edge:.1%} mas que mercado)", 0, 1)
            pdf.cell(0, 4, f"   Expected Value: +{ev:.2f}{MONEDA} (ROI: +{roi:.1f}%)", 0, 1)
            pdf.cell(0, 4, f"   Stake recomendado (Kelly): {stake:.2f}{MONEDA} ({stake/BANKROLL*100:.1f}% bankroll)", 0, 1)
            pdf.cell(0, 4, f"   Probabilidades: Modelo {prob:.1%} vs Mercado {1/cuota:.1%}", 0, 1)
            pdf.ln(2)

        total_stake = sum(b['stake'] for b in resumen_bets)
        total_expected_gain = sum(b['ev'] for b in resumen_bets)
        roi_total = (total_expected_gain / total_stake) * 100 if total_stake > 0 else 0

        pdf.set_font('Arial', 'B', 9)
        pdf.cell(0, 5, 'RESUMEN DE VALUE BETS:', 0, 1)
        pdf.set_font('Arial', '', 9)
        pdf.cell(0, 4, f"Total a invertir: {total_stake:.2f}{MONEDA} ({total_stake/BANKROLL*100:.1f}% del bankroll)", 0, 1)
        pdf.cell(0, 4, f"Ganancia esperada: +{total_expected_gain:.2f}{MONEDA}", 0, 1)
        pdf.cell(0, 4, f"ROI esperado: +{roi_total:.1f}%", 0, 1)
        
    else:
        pdf.set_font('Arial', '', 9)
        pdf.cell(0, 5, 'NO HAY VALUE BETS EN ESTA JORNADA', 0, 1)
        pdf.multi_cell(0, 4, 
            'Ningun partido tiene edge > 3%. Es normal que no todas las jornadas tengan value bets. '
            'La paciencia es clave en el value betting.')
    
    pdf.ln(5)
    
    # NUEVA SECCIÓN: Explicación Confianza vs Edge
    pdf.set_font('Arial', 'B', 11)
    pdf.set_fill_color(200, 230, 255)  # Fondo azul claro
    pdf.cell(0, 7, 'Diferencia entre CONFIANZA y EDGE:', 0, 1, 'L', True)
    pdf.ln(1)
    
    pdf.set_font('Arial', '', 8)
    pdf.multi_cell(0, 4,
        'CONFIANZA = Que tan seguro esta el modelo de su prediccion (ej: 67% Local)\n'
        'EDGE = Cuanta ventaja tienes vs el mercado (ej: Modelo 67% vs Mercado 71% = -4%)\n\n'
        'IMPORTANTE: Puedes tener alta confianza pero NO tener edge (no apostar).\n'
        'O baja confianza pero SI tener edge (apuesta arriesgada pero con valor).\n\n'
        'Lo ideal es: ALTA CONFIANZA + EDGE > 3% = Apuesta perfecta')
    
    pdf.ln(3)
    
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, 'Partidos de Alta Confianza (>=60%):', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    alta_conf = [r for r in resultados if r['confianza'] >= 0.60]
    if alta_conf:
        for r in alta_conf:
            pred_txt = f"{r['local']}" if r['prediccion']=='Local' else f"{r['visitante']}" if r['prediccion']=='Visitante' else 'Empate'
            pdf.cell(0, 5, f"   - {r['local']} vs {r['visitante']}: {r['prediccion']} ({pred_txt}) - {r['confianza']:.1%}", 0, 1)
    else:
        pdf.cell(0, 5, '   Ninguno', 0, 1)
    
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, 'Oportunidades Moderadas (Edge 2-3%):', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    # Filtrar oportunidades que NO son value bets (edge entre 2% y 3%)
    oportunidades_moderadas = [(r, dif) for r, dif in oportunidades if 0.02 < dif <= 0.03]
    
    if oportunidades_moderadas:
        oportunidades_moderadas_sorted = sorted(oportunidades_moderadas, key=lambda x: x[1], reverse=True)
        for r, dif in oportunidades_moderadas_sorted:
            pred_txt = f"{r['local']}" if r['prediccion']=='Local' else f"{r['visitante']}" if r['prediccion']=='Visitante' else 'Empate'
            pdf.cell(0, 5, f"   - {r['local']} vs {r['visitante']}: {r['prediccion']} ({pred_txt})", 0, 1)
            pdf.cell(0, 5, f"     Edge: +{dif:.1%} | Confianza: {r['confianza']:.1%}", 0, 1)
    else:
        pdf.cell(0, 5, '   No hay oportunidades moderadas', 0, 1)
    
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, 'Partidos a Evitar (Baja confianza <50%):', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    baja_conf = [r for r in resultados if r['confianza'] < 0.50]
    if baja_conf:
        for r in baja_conf:
            pdf.cell(0, 5, f"   - {r['local']} vs {r['visitante']} ({r['confianza']:.1%}) - Demasiado impredecible", 0, 1)
    else:
        pdf.cell(0, 5, '   Ninguno - Todos los partidos tienen confianza aceptable', 0, 1)
    
    # Guardar
    pdf.output(nombre_archivo)
    print(f"✅ PDF generado: {nombre_archivo}\n")
    
    return nombre_archivo


# ============================================================================
# GENERAR EXCEL (Alternativa)
# ============================================================================

def generar_excel(resultados, nombre_archivo='predicciones_jornada.xlsx'):
    """Genera un Excel con las predicciones."""
    
    print("\n📊 Generando reporte Excel...")
    
    df = pd.DataFrame(resultados)
    
    # Formatear columnas
    df['Partido'] = df['local'] + ' vs ' + df['visitante']
    df['Prediccion'] = df['prediccion']
    df['Confianza'] = df['confianza'].apply(lambda x: f"{x:.1%}")
    df['Prob Local'] = df['prob_local'].apply(lambda x: f"{x:.1%}")
    df['Prob Empate'] = df['prob_empate'].apply(lambda x: f"{x:.1%}")
    df['Prob Visit'] = df['prob_visitante'].apply(lambda x: f"{x:.1%}")
    df['Cuotas'] = df.apply(lambda x: f"{x['cuota_h']:.2f}-{x['cuota_d']:.2f}-{x['cuota_a']:.2f}", axis=1)
    df['Oportunidad'] = df['diferencia_valor'].apply(lambda x: 'SÍ' if abs(x) > 0.08 else 'No')
    
    df_export = df[['Partido', 'Prediccion', 'Confianza', 'Prob Local', 'Prob Empate', 
                    'Prob Visit', 'Cuotas', 'Oportunidad', 'forma_local', 'forma_visitante']]
    
    df_export.to_excel(nombre_archivo, index=False)
    print(f"✅ Excel generado: {nombre_archivo}\n")
    
    return nombre_archivo


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Función principal."""
    
    print("\n" + "⚽" * 35)
    print("   PREDICTOR DE JORNADA COMPLETA")
    print(f"   Jornada {NUMERO_JORNADA}")
    print("   💰 CON VALUE BETTING PROFESIONAL")
    print("⚽" * 35 + "\n")
    
    # Cargar sistema
    modelo, features, df = cargar_sistema()
    
    if modelo is None or df is None:
        print("❌ No se pudo cargar el sistema")
        return
    
    # Predecir todos los partidos
    print("="*70)
    print(f"🔮 PREDICIENDO {len(partidos_jornada)} PARTIDOS")
    print("="*70 + "\n")
    
    resultados = []
    
    for i, partido in enumerate(partidos_jornada, 1):
        print(f"[{i}/{len(partidos_jornada)}] {partido['local']} vs {partido['visitante']}...", end=' ')
        
        resultado = predecir_partido(partido, modelo, features, df)
        
        if resultado:
            resultados.append(resultado)
            print(f"✅ {resultado['prediccion']} ({resultado['confianza']:.1%})")
        else:
            print("❌ Error")
    
    print(f"\n✅ Predicciones completadas: {len(resultados)}/{len(partidos_jornada)}")
    
    # Mostrar resumen en consola
    print("\n" + "="*70)
    print("📊 RESUMEN DE PREDICCIONES CON ANÁLISIS DE VALOR")
    print("="*70)
    print(f"{'Partido':<40} {'Predicción':<12} {'Confianza':<10} {'Edge':<15}")
    print("-" * 70)
    
    for r in resultados:
        partido = f"{r['local']} vs {r['visitante']}"
        pred = r['prediccion']
        conf = f"{r['confianza']:.1%}"
        
        # El edge ya está calculado correctamente en diferencia_valor
        diferencia = r['diferencia_valor']
        
        # Formato de valor
        if diferencia > 0.08:
            valor_texto = f"+{diferencia:.1%} 🔥"
        elif diferencia > 0.05:
            valor_texto = f"+{diferencia:.1%} ⭐"
        elif diferencia > 0.03:
            valor_texto = f"+{diferencia:.1%} ✓"
        elif diferencia > 0:
            valor_texto = f"+{diferencia:.1%}"
        else:
            valor_texto = f"{diferencia:.1%}"
        
        emoji = "🟢" if pred == 'Local' else "🟡" if pred == 'Empate' else "🔵"
        
        print(f"{emoji} {partido:<38} {pred:<12} {conf:<10} {valor_texto:<15}")
    
    print("-" * 70)
    
    # =========================================================================
    # ANÁLISIS DE EXPECTED VALUE - VALUE BETTING PROFESIONAL
    # =========================================================================
    
    # =========================================================================
    # ANÁLISIS CON 3 CAPAS
    # =========================================================================
    
    print(f"\n💰 VALUE BETTING - SISTEMA DE 3 CAPAS:")
    print("-" * 70)
    print(f"CAPA 1: Factor conservador {FACTOR_CONSERVADOR} ({(1-FACTOR_CONSERVADOR)*100:.0f}% descuento)")
    print(f"CAPA 2: Edge mín {UMBRAL_EDGE_MINIMO:.0%}, Cuota máx {CUOTA_MAXIMA}, Prob mín {PROBABILIDAD_MINIMA:.0%}")
    print(f"CAPA 3: Kelly {KELLY_FRACTION:.0%}, Stake máx {STAKE_MAXIMO_PCT:.1%}%")
    print()
    
    value_bets = []
    partidos_rechazados = []
    
    for r in resultados:
        # Determinar cuota y prob según predicción
        if r['prediccion'] == 'Local':
            cuota = r['cuota_h']
            prob = r['prob_local']
            prob_orig = r.get('prob_local_original', prob)
        elif r['prediccion'] == 'Empate':
            cuota = r['cuota_d']
            prob = r['prob_empate']
            prob_orig = r.get('prob_empate_original', prob)
        else:
            cuota = r['cuota_a']
            prob = r['prob_visitante']
            prob_orig = r.get('prob_visitante_original', prob)
        
        # Aplicar 3 capas
        value_info = calcular_value_bet_con_3_capas(prob, cuota, BANKROLL)
        
        if value_info:
            value_bets.append({
                'partido': f"{r['local']} vs {r['visitante']}",
                'apuesta': r['local'] if r['prediccion']=='Local' else r['visitante'] if r['prediccion']=='Visitante' else 'Empate',
                'tipo': r['prediccion'],
                'prob_original': prob_orig,
                'ajuste': prob - prob_orig,
                **value_info
            })
        else:
            # Guardar rechazados
            prob_mercado = 1/cuota
            edge = prob - prob_mercado
            edge_pct = edge / prob_mercado if prob_mercado > 0 else 0
            
            razones = []
            if edge_pct < UMBRAL_EDGE_MINIMO:
                razones.append(f"Edge {edge_pct:.1%} < {UMBRAL_EDGE_MINIMO:.0%}")
            if cuota > CUOTA_MAXIMA:
                razones.append(f"Cuota {cuota:.2f} > {CUOTA_MAXIMA}")
            if prob < PROBABILIDAD_MINIMA:
                razones.append(f"Prob {prob:.1%} < {PROBABILIDAD_MINIMA:.0%}")
            
            partidos_rechazados.append({
                'partido': f"{r['local']} vs {r['visitante']}",
                'tipo': r['prediccion'],
                'cuota': cuota,
                'prob': prob,
                'edge_pct': edge_pct,
                'razon': ' | '.join(razones) if razones else 'Sin razón'
            })
    
    # Mostrar value bets
    if value_bets:
        value_bets_sorted = sorted(value_bets, key=lambda x: x['ev'], reverse=True)
        
        print(f"🎯 SE DETECTARON {len(value_bets)} VALUE BETS:\n")
        
        total_stake = sum(vb['stake'] for vb in value_bets_sorted)
        # Si tienes el EV absoluto en vb['ev'] (expected value en dólares):
        total_expected_gain = sum(vb['ev'] for vb in value_bets_sorted)
        # Si tienes el ROI individual en vb['roi'] (en porcentaje), puedes ponderar así:
        # total_expected_gain = sum(vb['stake'] * (vb['roi'] / 100) for vb in value_bets_sorted)
        roi_total = (total_expected_gain / total_stake) * 100 if total_stake > 0 else 0

        for i, vb in enumerate(value_bets_sorted, 1):
            print(f"{i}. {vb['partido']}")
            print(f"   Apostar: {vb['tipo']} ({vb['apuesta']}) | Cuota: {vb['cuota']:.2f}")
            print(f"   Edge: +{vb['edge_pct']:.1%} (Modelo ve {vb['edge_pct']:.1%} mas que mercado)")
            print(f"   Expected Value: ${vb['ev']:+.2f} (ROI: {vb['roi']:+.1%})")
            print(f"   Stake recomendado: ${vb['stake']:.2f} ({vb['kelly_fraction']*100:.1f}% bankroll)")
            print(f"   Probabilidades: Modelo {vb['prob_modelo']:.1%} vs Mercado {vb['prob_mercado']:.1%}")
            print(f"   Ajuste CAPA 1: {vb['prob_original']:.1%} → {vb['prob_modelo']:.1%} ({vb['ajuste']:+.1%})")
            print()

        print("="*70)
        print("RESUMEN DE VALUE BETS:")
        print("="*70)
        print(f"Total a invertir: ${total_stake:.2f} ({total_stake/BANKROLL*100:.1f}% del bankroll)")
        print(f"Ganancia esperada: +{total_expected_gain:.2f}$")
        print(f"ROI esperado: +{roi_total:.1f}%")
        print()
    else:
        print("❌ NO SE DETECTARON VALUE BETS")
        print(f"   Ningún partido pasó los filtros de las 3 capas\n")
    
    # Mostrar rechazados
    if partidos_rechazados and len(partidos_rechazados) <= 10:
        print(f"📋 PARTIDOS RECHAZADOS ({len(partidos_rechazados)}):")
        for pr in partidos_rechazados:
            print(f"• {pr['partido']}: {pr['tipo']}")
            print(f"  Cuota: {pr['cuota']:.2f} | Prob: {pr['prob']:.1%} | Edge: {pr['edge_pct']:+.1%}")
            print(f"  Razón: {pr['razon']}")
        print()
    
    print("-" * 70)
    
    # Tabla de confianza
    print(f"\n📊 ANÁLISIS POR NIVEL DE CONFIANZA:")
    print("-" * 70)
    
    alta_confianza = [r for r in resultados if r['confianza'] >= 0.60]
    media_confianza = [r for r in resultados if 0.50 <= r['confianza'] < 0.60]
    baja_confianza = [r for r in resultados if r['confianza'] < 0.50]
    
    if alta_confianza:
        print(f"\n🟢 ALTA CONFIANZA (≥60%) - {len(alta_confianza)} partidos:")
        for r in alta_confianza:
            pred_texto = f"{r['prediccion']} {r['local']}" if r['prediccion'] == 'Local' else f"{r['prediccion']} {r['visitante']}" if r['prediccion'] == 'Visitante' else 'Empate'
            print(f"   • {r['local']} vs {r['visitante']}: {pred_texto} ({r['confianza']:.1%})")
    
    if media_confianza:
        print(f"\n🟡 MEDIA CONFIANZA (50-60%) - {len(media_confianza)} partidos:")
        for r in media_confianza:
            pred_texto = f"{r['prediccion']} {r['local']}" if r['prediccion'] == 'Local' else f"{r['prediccion']} {r['visitante']}" if r['prediccion'] == 'Visitante' else 'Empate'
            print(f"   • {r['local']} vs {r['visitante']}: {pred_texto} ({r['confianza']:.1%})")
    
    if baja_confianza:
        print(f"\n🔴 BAJA CONFIANZA (<50%) - {len(baja_confianza)} partidos - ⚠️ EVITAR:")
        for r in baja_confianza:
            pred_texto = f"{r['prediccion']} {r['local']}" if r['prediccion'] == 'Local' else f"{r['prediccion']} {r['visitante']}" if r['prediccion'] == 'Visitante' else 'Empate'
            print(f"   • {r['local']} vs {r['visitante']}: {pred_texto} ({r['confianza']:.1%})")
    
    # Oportunidades de valor
    print(f"\n💰 OPORTUNIDADES DE VALOR (Modelo ve más que el mercado):")
    print("-" * 70)
    
    oportunidades = []
    for r in resultados:
        dif = r['diferencia_valor']  # Ya es el edge correcto
        
        if dif > 0.05:  # Más de 5% de diferencia
            oportunidades.append((r, dif))
    
    if oportunidades:
        # Ordenar por diferencia
        oportunidades.sort(key=lambda x: x[1], reverse=True)
        
        for r, dif in oportunidades:
            pred_texto = f"{r['prediccion']} {r['local']}" if r['prediccion'] == 'Local' else f"{r['prediccion']} {r['visitante']}" if r['prediccion'] == 'Visitante' else 'Empate'
            emoji_valor = "🔥" if dif > 0.08 else "⭐"
            
            prob_mercado = r['prob_mercado_local'] if r['prediccion']=='Local' else r['prob_mercado_visitante'] if r['prediccion']=='Visitante' else r['prob_mercado_empate']
            
            print(f"   {emoji_valor} {r['local']} vs {r['visitante']}")
            print(f"      → {pred_texto} | Modelo: {r['confianza']:.1%} vs Mercado: {prob_mercado:.1%}")
            print(f"      → EDGE: +{dif:.1%}")
    else:
        print("   No se detectaron oportunidades significativas (>5%)")
    
    print("-" * 70)
    
    # Generar reportes
    print("\n" + "="*70)
    print("📁 GENERANDO REPORTES")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jornada_tag = f"jornada{NUMERO_JORNADA}_{timestamp}"

    # PDF
    pdf_file = None
    if PDF_AVAILABLE:
        pdf_path = os.path.join(RUTA_REPORTES, f'predicciones_{jornada_tag}.pdf')
        pdf_file = generar_pdf(resultados, pdf_path)
    else:
        print("\n💡 Instala fpdf para generar PDF: pip install fpdf2")
    
    # Excel (si openpyxl está instalado)
    excel_file = None
    try:
        excel_path = os.path.join(RUTA_REPORTES, f'predicciones_{jornada_tag}.xlsx')
        excel_file = generar_excel(resultados, excel_path)
    except ImportError:
        print("\n💡 Para generar Excel, instala: pip install openpyxl")
    
    print("="*70)
    print("✅ PROCESO COMPLETADO - VALUE BETTING")
    print("="*70)
    print(f"\nArchivos generados:")
    if pdf_file:
        print(f"   📄 {pdf_file}")
    if excel_file:
        print(f"   📊 {excel_file}")
    
    value_bets = len([r for r in resultados if r['diferencia_valor'] > UMBRAL_EDGE_MINIMO])
    edges_positivos = len([r for r in resultados if r['diferencia_valor'] > 0])
    
    print(f"\n📊 ESTADÍSTICAS DE LA JORNADA:")
    print(f"   Partidos analizados: {len(resultados)}")
    print(f"   Partidos con edge positivo: {edges_positivos}")
    print(f"   🎯 VALUE BETS (edge > {UMBRAL_EDGE_MINIMO:.1%}): {value_bets}")
    print(f"\n💡 Estrategia: Solo apostar en value bets con edge > {UMBRAL_EDGE_MINIMO:.1%}")
    print(f"💰 Bankroll configurado: {BANKROLL}{MONEDA}\n")


if __name__ == "__main__":
    main()