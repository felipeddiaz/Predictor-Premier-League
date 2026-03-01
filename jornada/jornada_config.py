# -*- coding: utf-8 -*-
"""
jornada_config.py — Configuracion de la jornada actual.

INSTRUCCIONES:
  Antes de cada jornada, edita este archivo:
    1. Actualiza NUMERO_JORNADA con el numero correcto.
    2. Reemplaza la lista PARTIDOS_JORNADA con los partidos y cuotas actuales.

  Los nombres de equipo deben coincidir exactamente con los del historico.
  Ejemplos validos: 'Arsenal', 'Man City', 'Man United', 'Nott'm Forest'

Este archivo reemplaza la lista hardcodeada que antes estaba en
predecir_jornada_completa.py. Solo hay que editar aqui, sin tocar
el codigo del predictor.
"""

from core.models import Partido, ConfigJornada

# ============================================================================
# EDITA AQUI: Numero de jornada actual
# ============================================================================

NUMERO_JORNADA = 15

# ============================================================================
# EDITA AQUI: Partidos de la jornada
# Formato: Partido(local='...', visitante='...', cuota_h=X.XX, cuota_d=X.XX, cuota_a=X.XX)
# ============================================================================

PARTIDOS_JORNADA = [
    # Martes 2 de Diciembre
    Partido(local='Bournemouth',  visitante='Everton',       cuota_h=2.30, cuota_d=3.40, cuota_a=3.10),
    Partido(local='Fulham',       visitante='Man City',       cuota_h=5.00, cuota_d=4.25, cuota_a=1.67),
    Partido(local='Newcastle',    visitante='Tottenham',      cuota_h=2.50, cuota_d=3.60, cuota_a=2.70),

    # Miercoles 3 de Diciembre
    Partido(local='Arsenal',      visitante='Brentford',      cuota_h=1.30, cuota_d=5.50, cuota_a=10.00),
    Partido(local='Brighton',     visitante='Aston Villa',    cuota_h=2.60, cuota_d=3.50, cuota_a=2.60),
    Partido(local='Burnley',      visitante='Crystal Palace', cuota_h=2.90, cuota_d=3.30, cuota_a=2.45),
    Partido(local='Wolves',       visitante="Nott'm Forest",  cuota_h=2.35, cuota_d=3.40, cuota_a=3.00),
    Partido(local='Leeds',        visitante='Chelsea',        cuota_h=4.50, cuota_d=4.00, cuota_a=1.75),
    Partido(local='Liverpool',    visitante='Sunderland',     cuota_h=1.25, cuota_d=6.50, cuota_a=11.00),

    # Jueves 4 de Diciembre
    Partido(local='Man United',   visitante='West Ham',       cuota_h=1.65, cuota_d=4.20, cuota_a=5.00),
]

# ============================================================================
# Objeto ConfigJornada listo para usar en predecir_jornada_completa.py
# No editar esta linea.
# ============================================================================

CONFIG_JORNADA = ConfigJornada(numero=NUMERO_JORNADA, partidos=PARTIDOS_JORNADA)
