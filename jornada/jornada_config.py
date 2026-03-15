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
    Partido(local='Crystal Palace', visitante='Leeds', cuota_h=2.51, cuota_d=3.28, cuota_a=2.94),
    Partido(local='Man United', visitante='Aston Villa', cuota_h=1.71, cuota_d=4.16, cuota_a=4.53),
    Partido(local="Nott'm Forest", visitante='Fulham', cuota_h=2.26, cuota_d=3.40, cuota_a=3.26),
    Partido(local='Liverpool', visitante='Tottenham', cuota_h=1.29, cuota_d=6.17, cuota_a=9.10),
    Partido(local='Brentford', visitante='Wolves', cuota_h=1.60, cuota_d=4.20, cuota_a=5.75),
]

# ============================================================================
# Objeto ConfigJornada listo para usar en predecir_jornada_completa.py
# No editar esta linea.
# ============================================================================

CONFIG_JORNADA = ConfigJornada(numero=NUMERO_JORNADA, partidos=PARTIDOS_JORNADA)
