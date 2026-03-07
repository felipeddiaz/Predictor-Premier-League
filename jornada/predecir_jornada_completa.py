"""
predecir_jornada_completa.py — Punto de entrada del predictor.

Ejecuta la prediccion completa de una jornada y genera los reportes.

Para configurar la jornada (partidos + cuotas), edita:
    jornada/jornada_config.py

Para cambiar parametros del sistema (bankroll, umbrales, etc.), edita:
    config.py

Uso:
    python jornada/predecir_jornada_completa.py
"""

from config import BANKROLL_DEFAULT, UMBRAL_EDGE_MINIMO, MONEDA
from core.predictor import Predictor
from jornada.jornada_config import CONFIG_JORNADA

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("   PREDICTOR DE JORNADA COMPLETA")
    print(f"   Jornada {CONFIG_JORNADA.numero}")
    print("   CON VALUE BETTING PROFESIONAL")
    print("=" * 70 + "\n")

    # Inicializar y cargar el sistema
    predictor = Predictor(bankroll=BANKROLL_DEFAULT)
    if not predictor.cargar():
        print("ERROR: No se pudo cargar el sistema. Verifica los modelos.")
        return

    # Predecir todos los partidos de la jornada
    predicciones = predictor.predecir_jornada(CONFIG_JORNADA)

    if not predicciones:
        print("ERROR: No se generaron predicciones.")
        return

    # Mostrar resumen en consola
    _mostrar_resumen_consola(predicciones, predictor)

    # Generar reportes
    print("\n" + "=" * 70)
    print("GENERANDO REPORTES")
    print("=" * 70)

    pdf_path = predictor.generar_reporte(predicciones, CONFIG_JORNADA.numero, formato='pdf')
    excel_path = predictor.generar_reporte(predicciones, CONFIG_JORNADA.numero, formato='excel')

    print("=" * 70)
    print("PROCESO COMPLETADO")
    print("=" * 70)
    if pdf_path:
        print(f"   PDF  : {pdf_path}")
    if excel_path:
        print(f"   Excel: {excel_path}")
    print()


def _mostrar_resumen_consola(predicciones, predictor: Predictor):
    """Imprime el resumen de predicciones y value bets en consola."""
    from config import (
        FACTOR_CONSERVADOR, UMBRAL_EDGE_MINIMO,
        CUOTA_MAXIMA, PROBABILIDAD_MINIMA, KELLY_FRACTION, STAKE_MAXIMO_PCT,
        BANKROLL_DEFAULT, MONEDA
    )

    bankroll = BANKROLL_DEFAULT

    # Tabla de predicciones
    print("=" * 70)
    print("RESUMEN DE PREDICCIONES CON ANALISIS DE VALOR")
    print("=" * 70)
    print(f"{'Partido':<40} {'Prediccion':<12} {'Confianza':<10} {'Edge':<15}")
    print("-" * 70)

    for p in predicciones:
        partido = f"{p.partido.local} vs {p.partido.visitante}"
        dif = p.diferencia_valor

        if dif > 0.08:
            valor_texto = f"+{dif:.1%} (GRAN OPORT.)"
        elif dif > 0.05:
            valor_texto = f"+{dif:.1%} (OPORT.)"
        elif dif > 0.03:
            valor_texto = f"+{dif:.1%} (ok)"
        elif dif > 0:
            valor_texto = f"+{dif:.1%}"
        else:
            valor_texto = f"{dif:.1%}"

        indicador = ">" if p.resultado_predicho == 'Local' else "=" if p.resultado_predicho == 'Empate' else "<"
        print(f"[{indicador}] {partido:<38} {p.resultado_predicho:<12} {p.confianza:.1%}     {valor_texto}")

    print("-" * 70)

    # Value betting — 3 capas
    print(f"\nVALUE BETTING — SISTEMA DE 3 CAPAS:")
    print("-" * 70)
    print(f"CAPA 1: Factor conservador {FACTOR_CONSERVADOR} ({(1-FACTOR_CONSERVADOR)*100:.0f}% descuento)")
    print(f"CAPA 2: Edge min {UMBRAL_EDGE_MINIMO:.0%}, Cuota max {CUOTA_MAXIMA}, Prob min {PROBABILIDAD_MINIMA:.0%}")
    print(f"CAPA 3: Kelly {KELLY_FRACTION:.0%}, Stake max {STAKE_MAXIMO_PCT:.1%}")
    print()

    value_bets = []
    rechazados = []

    for p in predicciones:
        if p.resultado_predicho == 'Local':
            cuota = p.partido.cuota_h
            prob = p.prob_local
            prob_orig = p.prob_local_original
        elif p.resultado_predicho == 'Empate':
            cuota = p.partido.cuota_d
            prob = p.prob_empate
            prob_orig = p.prob_empate_original
        else:
            cuota = p.partido.cuota_a
            prob = p.prob_visitante
            prob_orig = p.prob_visitante_original

        # Eliminar vig para calcular edge real
        from core.sistema_expected_value import eliminar_vig
        fair_h, fair_d, fair_a = eliminar_vig(
            p.partido.cuota_h, p.partido.cuota_d, p.partido.cuota_a
        )
        fair_map = {'Local': fair_h, 'Empate': fair_d, 'Visitante': fair_a}
        prob_fair = fair_map[p.resultado_predicho]

        info = predictor.calcular_value_bet(prob, cuota, prob_fair=prob_fair)

        if info:
            value_bets.append({
                'partido': f"{p.partido.local} vs {p.partido.visitante}",
                'apuesta': p.partido.local if p.resultado_predicho == 'Local'
                           else p.partido.visitante if p.resultado_predicho == 'Visitante'
                           else 'Empate',
                'tipo': p.resultado_predicho,
                'prob_original': prob_orig,
                'ajuste': prob - prob_orig,
                **info,
            })
        else:
            prob_mercado = prob_fair
            edge_pct = (prob - prob_mercado) / prob_mercado if prob_mercado > 0 else 0
            razones = []
            if edge_pct < UMBRAL_EDGE_MINIMO:
                razones.append(f"Edge {edge_pct:.1%} < {UMBRAL_EDGE_MINIMO:.0%}")
            if cuota > CUOTA_MAXIMA:
                razones.append(f"Cuota {cuota:.2f} > {CUOTA_MAXIMA}")
            if prob < PROBABILIDAD_MINIMA:
                razones.append(f"Prob {prob:.1%} < {PROBABILIDAD_MINIMA:.0%}")
            rechazados.append({
                'partido': f"{p.partido.local} vs {p.partido.visitante}",
                'tipo': p.resultado_predicho,
                'cuota': cuota,
                'prob': prob,
                'edge_pct': edge_pct,
                'razon': ' | '.join(razones) or 'Sin razon',
            })

    if value_bets:
        value_bets_sorted = sorted(value_bets, key=lambda x: x['ev'], reverse=True)
        total_stake = sum(vb['stake'] for vb in value_bets_sorted)
        total_ev = sum(vb['ev'] for vb in value_bets_sorted)
        roi_total = (total_ev / total_stake) * 100 if total_stake > 0 else 0

        print(f"SE DETECTARON {len(value_bets)} VALUE BETS:\n")
        for i, vb in enumerate(value_bets_sorted, 1):
            print(f"{i}. {vb['partido']}")
            print(f"   Apostar: {vb['tipo']} ({vb['apuesta']}) | Cuota: {vb['cuota']:.2f}")
            print(f"   Edge: +{vb['edge_pct']:.1%}")
            print(f"   Expected Value: {MONEDA}{vb['ev']:+.2f} (ROI: {vb['roi']:+.1%})")
            print(f"   Stake recomendado: {MONEDA}{vb['stake']:.2f} ({vb['kelly_fraction']*100:.1f}% bankroll)")
            print(f"   Probabilidades: Modelo {vb['prob_modelo']:.1%} vs Mercado {vb['prob_mercado']:.1%}")
            print(f"   Ajuste Capa 1: {vb['prob_original']:.1%} -> {vb['prob_modelo']:.1%} ({vb['ajuste']:+.1%})")
            print()

        print("=" * 70)
        print("RESUMEN DE VALUE BETS:")
        print(f"Total a invertir: {MONEDA}{total_stake:.2f} ({total_stake/bankroll*100:.1f}% del bankroll)")
        print(f"Ganancia esperada: +{total_ev:.2f}{MONEDA}")
        print(f"ROI esperado: +{roi_total:.1f}%")
        print()
    else:
        print("NO SE DETECTARON VALUE BETS")
        print(f"   Ningun partido paso los filtros de las 3 capas\n")

    if rechazados and len(rechazados) <= 10:
        print(f"PARTIDOS RECHAZADOS ({len(rechazados)}):")
        for pr in rechazados:
            print(f"  {pr['partido']}: {pr['tipo']}")
            print(f"  Cuota: {pr['cuota']:.2f} | Prob: {pr['prob']:.1%} | Edge: {pr['edge_pct']:+.1%}")
            print(f"  Razon: {pr['razon']}")
        print()

    # Niveles de confianza
    print("ANALISIS POR NIVEL DE CONFIANZA:")
    print("-" * 70)

    alta = [p for p in predicciones if p.es_alta_confianza()]
    media = [p for p in predicciones if p.es_media_confianza()]
    baja = [p for p in predicciones if p.es_baja_confianza()]

    if alta:
        print(f"\nALTA CONFIANZA (>=60%) — {len(alta)} partidos:")
        for p in alta:
            print(f"   {p.partido.local} vs {p.partido.visitante}: {p.resultado_predicho} ({p.confianza:.1%})")

    if media:
        print(f"\nMEDIA CONFIANZA (50-60%) — {len(media)} partidos:")
        for p in media:
            print(f"   {p.partido.local} vs {p.partido.visitante}: {p.resultado_predicho} ({p.confianza:.1%})")

    if baja:
        print(f"\nBAJA CONFIANZA (<50%) — {len(baja)} partidos — EVITAR:")
        for p in baja:
            print(f"   {p.partido.local} vs {p.partido.visitante}: {p.resultado_predicho} ({p.confianza:.1%})")

    edges_positivos = sum(1 for p in predicciones if p.diferencia_valor > 0)
    vb_count = sum(1 for p in predicciones if p.tiene_edge(UMBRAL_EDGE_MINIMO))

    print(f"\nESTADISTICAS DE LA JORNADA:")
    print(f"   Partidos analizados      : {len(predicciones)}")
    print(f"   Con edge positivo        : {edges_positivos}")
    print(f"   VALUE BETS (edge > {UMBRAL_EDGE_MINIMO:.1%}) : {vb_count}")
    print(f"   Bankroll configurado     : {bankroll}{MONEDA}")
    print()


if __name__ == "__main__":
    main()
