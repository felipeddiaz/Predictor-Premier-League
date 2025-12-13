# -*- coding: utf-8 -*-
"""
Sistema de Expected Value (EV) y Kelly Criterion
¡Lo MÁS IMPORTANTE para apuestas rentables!

EV = (Prob_Ganar × Ganancia) - (Prob_Perder × Pérdida)
Solo apuesta si EV > 0
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

RUTA_DATOS = './datos/procesados/premier_league_con_features.csv'
RUTA_MODELOS = './modelos/'

# ============================================================================
# CÁLCULO DE EXPECTED VALUE (EV)
# ============================================================================

def calcular_ev(prob_modelo, cuota, stake=1.0):
    """
    Calcula el Expected Value de una apuesta.
    
    Args:
        prob_modelo (float): Probabilidad según tu modelo (0-1)
        cuota (float): Cuota ofrecida (formato decimal)
        stake (float): Cantidad apostada (default 1€)
        
    Returns:
        dict: EV, ganancia esperada, y análisis
    """
    
    # Probabilidad de perder
    prob_perder = 1 - prob_modelo
    
    # Ganancia si aciertas (cuota × stake - stake)
    ganancia_si_gana = (cuota * stake) - stake
    
    # Pérdida si fallas (pierdes tu stake)
    perdida_si_pierde = stake
    
    # FÓRMULA DEL EV
    ev = (prob_modelo * ganancia_si_gana) - (prob_perder * perdida_si_pierde)
    
    # ROI (Return on Investment)
    roi = (ev / stake) * 100
    
    # Probabilidad implícita de la casa
    prob_casa = 1 / cuota
    
    # Edge (ventaja sobre la casa)
    edge = prob_modelo - prob_casa
    
    return {
        'ev': ev,
        'roi': roi,
        'edge': edge,
        'prob_modelo': prob_modelo,
        'prob_casa': prob_casa,
        'ganancia_esperada': ganancia_si_gana,
        'apostar': ev > 0
    }


def kelly_criterion(prob_modelo, cuota, kelly_fraction=0.25):
    """
    Calcula el porcentaje óptimo del bankroll a apostar usando Kelly Criterion.
    
    Args:
        prob_modelo (float): Probabilidad según tu modelo
        cuota (float): Cuota ofrecida
        kelly_fraction (float): Fracción de Kelly (0.25 = 25% Kelly, más conservador)
        
    Returns:
        float: Porcentaje del bankroll a apostar (0-1)
    """
    
    # Fórmula de Kelly: f = (bp - q) / b
    # donde:
    # b = cuota - 1 (ganancia neta por unidad apostada)
    # p = probabilidad de ganar
    # q = probabilidad de perder (1-p)
    
    b = cuota - 1
    p = prob_modelo
    q = 1 - p
    
    # Kelly completo
    kelly_full = (b * p - q) / b
    
    # Aplicar fracción de Kelly (más conservador)
    kelly_fraction_result = max(0, kelly_full * kelly_fraction)
    
    # Limitar a 10% máximo (seguridad)
    kelly_safe = min(kelly_fraction_result, 0.10)
    
    return {
        'kelly_full': max(0, kelly_full),
        'kelly_fraction': kelly_fraction_result,
        'kelly_safe': kelly_safe,
        'recomendacion_pct': kelly_safe * 100
    }


# ============================================================================
# ANÁLISIS COMPLETO DE APUESTA
# ============================================================================

def analizar_apuesta(prediccion, cuota, stake=10, bankroll=1000):
    """
    Análisis completo de una apuesta con EV y Kelly.
    
    Args:
        prediccion (dict): Resultado de predicción con probabilidades
        cuota (float): Cuota ofrecida
        stake (float): Apuesta base
        bankroll (float): Capital total disponible
        
    Returns:
        dict: Análisis completo con recomendación
    """
    
    prob = prediccion['confianza']
    
    # Calcular EV
    ev_result = calcular_ev(prob, cuota, stake)
    
    # Calcular Kelly
    kelly_result = kelly_criterion(prob, cuota)
    
    # Recomendación de stake
    stake_recomendado = bankroll * kelly_result['kelly_safe']
    
    # Clasificación de la apuesta
    if ev_result['ev'] <= 0:
        clasificacion = "🔴 NO APOSTAR"
        razon = "EV negativo - Apuesta perdedora a largo plazo"
    elif ev_result['edge'] < 0.03:
        clasificacion = "⚠️ VALOR BAJO"
        razon = "Edge < 3% - No vale la pena el riesgo"
    elif ev_result['edge'] < 0.08:
        clasificacion = "🟡 VALOR MODERADO"
        razon = "Edge 3-8% - Considerar apostar"
    else:
        clasificacion = "🟢 ALTO VALOR"
        razon = "Edge > 8% - Excelente oportunidad"
    
    return {
        **ev_result,
        **kelly_result,
        'stake_recomendado': stake_recomendado,
        'clasificacion': clasificacion,
        'razon': razon,
        'ganancia_esperada_real': ev_result['ev'],
        'roi_anualizado': ev_result['roi'] * 50  # Asumiendo ~50 apuestas/año
    }


# ============================================================================
# SIMULACIÓN DE ROI
# ============================================================================

def simular_roi_historico(resultados_predicciones, strategy='ev_positive', bankroll_inicial=1000):
    """
    Simula qué hubiera pasado si apostaras con tu modelo en el pasado.
    
    Args:
        resultados_predicciones (list): Lista de predicciones vs resultados reales
        strategy (str): Estrategia de apuesta
            - 'ev_positive': Solo apostar con EV > 0
            - 'high_confidence': Solo apostar con confianza > 60%
            - 'value_betting': Solo apostar con edge > 5%
        bankroll_inicial (float): Capital inicial
        
    Returns:
        dict: Resultados de la simulación
    """
    
    bankroll = bankroll_inicial
    bankroll_history = [bankroll]
    apuestas_realizadas = []
    
    for pred in resultados_predicciones:
        # Determinar cuota según predicción y resultado
        if pred['prediccion'] == 'Local':
            cuota = pred['cuota_h']
            prob = pred['prob_local']
            gano = pred['resultado_real'] == 0
        elif pred['prediccion'] == 'Empate':
            cuota = pred['cuota_d']
            prob = pred['prob_empate']
            gano = pred['resultado_real'] == 1
        else:
            cuota = pred['cuota_a']
            prob = pred['prob_visitante']
            gano = pred['resultado_real'] == 2
        
        # Calcular EV
        ev_info = calcular_ev(prob, cuota)
        kelly_info = kelly_criterion(prob, cuota)
        
        # Decidir si apostar según estrategia
        apostar = False
        
        if strategy == 'ev_positive':
            apostar = ev_info['ev'] > 0
        elif strategy == 'high_confidence':
            apostar = prob > 0.60
        elif strategy == 'value_betting':
            apostar = ev_info['edge'] > 0.05
        
        if apostar and bankroll > 0:
            # Calcular stake (Kelly conservador)
            stake = bankroll * kelly_info['kelly_safe']
            stake = max(10, min(stake, bankroll * 0.05))  # Entre 10€ y 5% bankroll
            
            # Resultado de la apuesta
            if gano:
                ganancia = stake * (cuota - 1)
                bankroll += ganancia
            else:
                bankroll -= stake
            
            apuestas_realizadas.append({
                'partido': f"{pred['local']} vs {pred['visitante']}",
                'prediccion': pred['prediccion'],
                'cuota': cuota,
                'stake': stake,
                'gano': gano,
                'ganancia': ganancia if gano else -stake,
                'bankroll': bankroll,
                'ev': ev_info['ev'],
                'edge': ev_info['edge']
            })
            
            bankroll_history.append(bankroll)
    
    # Estadísticas
    if len(apuestas_realizadas) > 0:
        apuestas_ganadoras = sum(1 for a in apuestas_realizadas if a['gano'])
        tasa_acierto = apuestas_ganadoras / len(apuestas_realizadas) * 100
        
        ganancia_total = bankroll - bankroll_inicial
        roi_total = (ganancia_total / bankroll_inicial) * 100
        
        avg_stake = np.mean([a['stake'] for a in apuestas_realizadas])
        avg_cuota = np.mean([a['cuota'] for a in apuestas_realizadas])
        avg_edge = np.mean([a['edge'] for a in apuestas_realizadas])
    else:
        apuestas_ganadoras = 0
        tasa_acierto = 0
        ganancia_total = 0
        roi_total = 0
        avg_stake = 0
        avg_cuota = 0
        avg_edge = 0
    
    return {
        'bankroll_inicial': bankroll_inicial,
        'bankroll_final': bankroll,
        'ganancia_total': ganancia_total,
        'roi_total': roi_total,
        'apuestas_realizadas': len(apuestas_realizadas),
        'apuestas_ganadoras': apuestas_ganadoras,
        'tasa_acierto': tasa_acierto,
        'avg_stake': avg_stake,
        'avg_cuota': avg_cuota,
        'avg_edge': avg_edge,
        'bankroll_history': bankroll_history,
        'detalle_apuestas': apuestas_realizadas
    }


# ============================================================================
# INTEGRACIÓN CON PREDICTOR DE JORNADA
# ============================================================================

def analizar_jornada_con_ev(partidos, modelo, features, df, bankroll=1000):
    """
    Analiza una jornada completa con Expected Value.
    """
    
    from predecir_jornada_completa import predecir_partido, obtener_stats_equipo, transformar_cuotas
    
    print("="*70)
    print("💰 ANÁLISIS DE EXPECTED VALUE - JORNADA COMPLETA")
    print("="*70)
    print(f"Bankroll disponible: {bankroll}€\n")
    
    resultados_ev = []
    
    for partido in partidos:
        # Predecir partido
        pred = predecir_partido(partido, modelo, features, df)
        
        if pred is None:
            continue
        
        # Analizar cada posible resultado
        analisis_opciones = []
        
        # Opción 1: Apostar al Local
        if pred['prediccion'] == 'Local':
            ev_local = analizar_apuesta(
                {'confianza': pred['prob_local']},
                partido['cuota_h'],
                stake=10,
                bankroll=bankroll
            )
            analisis_opciones.append(('Local', pred['local'], ev_local))
        
        # Opción 2: Apostar al Empate
        if pred['prediccion'] == 'Empate':
            ev_empate = analizar_apuesta(
                {'confianza': pred['prob_empate']},
                partido['cuota_d'],
                stake=10,
                bankroll=bankroll
            )
            analisis_opciones.append(('Empate', 'Empate', ev_empate))
        
        # Opción 3: Apostar al Visitante
        if pred['prediccion'] == 'Visitante':
            ev_visitante = analizar_apuesta(
                {'confianza': pred['prob_visitante']},
                partido['cuota_a'],
                stake=10,
                bankroll=bankroll
            )
            analisis_opciones.append(('Visitante', pred['visitante'], ev_visitante))
        
        # Guardar mejor opción
        if analisis_opciones:
            mejor_opcion = analisis_opciones[0]
            
            resultados_ev.append({
                'partido': f"{partido['local']} vs {partido['visitante']}",
                'local': partido['local'],
                'visitante': partido['visitante'],
                'prediccion': mejor_opcion[0],
                'equipo': mejor_opcion[1],
                'ev': mejor_opcion[2]['ev'],
                'roi': mejor_opcion[2]['roi'],
                'edge': mejor_opcion[2]['edge'],
                'clasificacion': mejor_opcion[2]['clasificacion'],
                'stake_recomendado': mejor_opcion[2]['stake_recomendado'],
                'cuota': partido[f"cuota_{'h' if mejor_opcion[0]=='Local' else 'd' if mejor_opcion[0]=='Empate' else 'a'}"],
                'prob_modelo': mejor_opcion[2]['prob_modelo'],
                'prob_casa': mejor_opcion[2]['prob_casa']
            })
    
    # Mostrar resultados
    print("\n📊 ANÁLISIS DE EXPECTED VALUE POR PARTIDO:")
    print("="*70)
    print(f"{'Partido':<40} {'EV':<8} {'Edge':<8} {'Recomendación':<20}")
    print("-"*70)
    
    for r in resultados_ev:
        print(f"{r['partido']:<40} {r['ev']:>6.2f}€ {r['edge']:>6.1%}  {r['clasificacion']}")
    
    # Resumen de apuestas recomendadas
    print("\n" + "="*70)
    print("🎯 APUESTAS RECOMENDADAS (EV Positivo):")
    print("="*70)
    
    apuestas_recomendadas = [r for r in resultados_ev if r['ev'] > 0 and r['edge'] > 0.03]
    
    if apuestas_recomendadas:
        apuestas_recomendadas_sorted = sorted(apuestas_recomendadas, key=lambda x: x['ev'], reverse=True)
        
        total_stake = 0
        ev_total = 0
        
        for i, r in enumerate(apuestas_recomendadas_sorted, 1):
            print(f"\n{i}. {r['partido']}")
            print(f"   Apostar a: {r['prediccion']} ({r['equipo']})")
            print(f"   Cuota: {r['cuota']:.2f}")
            print(f"   Stake recomendado: {r['stake_recomendado']:.2f}€ ({r['stake_recomendado']/bankroll*100:.1f}% del bankroll)")
            print(f"   Expected Value: +{r['ev']:.2f}€ (ROI: {r['roi']:.1f}%)")
            print(f"   Edge vs mercado: +{r['edge']:.1%}")
            print(f"   Probabilidad: Modelo {r['prob_modelo']:.1%} vs Casa {r['prob_casa']:.1%}")
            
            total_stake += r['stake_recomendado']
            ev_total += r['ev']
        
        print(f"\n{'='*70}")
        print(f"RESUMEN:")
        print(f"   Total a apostar: {total_stake:.2f}€ ({total_stake/bankroll*100:.1f}% del bankroll)")
        print(f"   EV Total esperado: +{ev_total:.2f}€")
        print(f"   ROI esperado: {(ev_total/total_stake)*100:.1f}%")
        
    else:
        print("\n   ⚠️  NO HAY APUESTAS RECOMENDADAS")
        print("   Ningún partido tiene EV positivo suficiente (edge > 3%)")
    
    print("="*70)
    
    return resultados_ev


# ============================================================================
# MAIN - EJEMPLO DE USO
# ============================================================================

def main():
    """Ejemplo de uso del sistema EV."""
    
    print("\n" + "💰" * 35)
    print("   SISTEMA DE EXPECTED VALUE (EV)")
    print("💰" * 35 + "\n")
    
    # Ejemplo 1: Cálculo simple de EV
    print("="*70)
    print("EJEMPLO 1: Cálculo de EV Individual")
    print("="*70)
    
    prob_modelo = 0.55  # Tu modelo dice 55%
    cuota = 2.20        # Casa ofrece 2.20
    stake = 10          # Apuestas 10€
    
    ev = calcular_ev(prob_modelo, cuota, stake)
    
    print(f"\nTu modelo: {prob_modelo:.1%}")
    print(f"Cuota ofrecida: {cuota:.2f}")
    print(f"Stake: {stake}€")
    print(f"\nResultado:")
    print(f"   Expected Value: {ev['ev']:+.2f}€")
    print(f"   ROI: {ev['roi']:+.1f}%")
    print(f"   Edge vs casa: {ev['edge']:+.1%}")
    print(f"   ¿Apostar? {'✅ SÍ' if ev['apostar'] else '❌ NO'}")
    
    # Ejemplo 2: Kelly Criterion
    print("\n" + "="*70)
    print("EJEMPLO 2: Kelly Criterion - ¿Cuánto apostar?")
    print("="*70)
    
    kelly = kelly_criterion(prob_modelo, cuota)
    bankroll = 1000
    
    print(f"\nBankroll: {bankroll}€")
    print(f"Kelly completo: {kelly['kelly_full']*100:.1f}% ({bankroll*kelly['kelly_full']:.2f}€)")
    print(f"Kelly 25% (conservador): {kelly['kelly_fraction']*100:.1f}% ({bankroll*kelly['kelly_fraction']:.2f}€)")
    print(f"Kelly seguro (recomendado): {kelly['kelly_safe']*100:.1f}% ({bankroll*kelly['kelly_safe']:.2f}€)")
    
    print("\n💡 Usa siempre Kelly conservador para proteger tu bankroll\n")


if __name__ == "__main__":
    main()