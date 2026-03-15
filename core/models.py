# -*- coding: utf-8 -*-
"""
models.py — Clases de estructura del proyecto.

Define los tipos de datos principales que fluyen por el sistema:
  - Partido: datos de entrada (equipos + cuotas)
  - Prediccion: resultado de predecir un partido
  - ConfigJornada: configuracion de la jornada a predecir

Uso:
    from models import Partido, Prediccion, ConfigJornada
"""

from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# PREDICCION BINARIA — Mercados O/U, tarjetas, corners
# ============================================================================

@dataclass
class PrediccionBinaria:
    """
    Predicciones de mercados binarios para un partido.

    Cada campo es la probabilidad de 'Over' del umbral correspondiente.
    """
    prob_over25: Optional[float] = None       # P(Over 2.5 goles)
    prob_over35_cards: Optional[float] = None  # P(Over 3.5 tarjetas amarillas)
    prob_over95_corners: Optional[float] = None  # P(Over 9.5 corners)


# ============================================================================
# PARTIDO — Datos de entrada de un partido
# ============================================================================

@dataclass
class Partido:
    """
    Representa un partido con sus cuotas de mercado.

    Atributos:
        local      : Nombre del equipo local (debe coincidir con el histórico)
        visitante  : Nombre del equipo visitante
        cuota_h    : Cuota de victoria local (ej: 2.10)
        cuota_d    : Cuota de empate (ej: 3.40)
        cuota_a    : Cuota de victoria visitante (ej: 3.20)

    Campos opcionales — Asian Handicap:
        ah_line    : Handicap de apertura para el local (ej: -1.5, 0.25, 1.0)
                     Negativo = local favorito, Positivo = visitante favorito.
                     Si se omite, las features AH se rellenan con 0 (modelo las ignora).
        ah_cuota_h : Cuota AH apertura para el local  (ej: 1.90)
        ah_cuota_a : Cuota AH apertura para el visitante (ej: 1.90)

    Uso sin AH (minimo):
        p = Partido(local='Arsenal', visitante='Chelsea',
                    cuota_h=1.80, cuota_d=3.60, cuota_a=4.50)

    Uso con AH (mas informacion para el modelo):
        p = Partido(local='Arsenal', visitante='Chelsea',
                    cuota_h=1.80, cuota_d=3.60, cuota_a=4.50,
                    ah_line=-1.5, ah_cuota_h=1.93, ah_cuota_a=1.97)
    """
    local: str
    visitante: str
    cuota_h: float
    cuota_d: float
    cuota_a: float
    # Asian Handicap — opcional
    ah_line:    Optional[float] = None
    ah_cuota_h: Optional[float] = None
    ah_cuota_a: Optional[float] = None

    def __post_init__(self):
        """Valida que los valores sean coherentes."""
        if self.cuota_h <= 1.0:
            raise ValueError(f"cuota_h debe ser > 1.0, recibido: {self.cuota_h}")
        if self.cuota_d <= 1.0:
            raise ValueError(f"cuota_d debe ser > 1.0, recibido: {self.cuota_d}")
        if self.cuota_a <= 1.0:
            raise ValueError(f"cuota_a debe ser > 1.0, recibido: {self.cuota_a}")
        if not self.local.strip():
            raise ValueError("El nombre del equipo local no puede estar vacío")
        if not self.visitante.strip():
            raise ValueError("El nombre del equipo visitante no puede estar vacío")

    def tiene_ah(self) -> bool:
        """True si se proporcionaron datos de Asian Handicap."""
        return self.ah_line is not None and self.ah_cuota_h is not None and self.ah_cuota_a is not None

    def a_dict(self) -> dict:
        """Convierte el partido al formato de diccionario usado internamente."""
        return {
            'local': self.local,
            'visitante': self.visitante,
            'cuota_h': self.cuota_h,
            'cuota_d': self.cuota_d,
            'cuota_a': self.cuota_a,
        }


# ============================================================================
# PREDICCION — Resultado de predecir un partido
# ============================================================================

@dataclass
class Prediccion:
    """
    Resultado completo de la prediccion de un partido.

    Atributos de entrada:
        partido               : El partido que se predijo
    
    Probabilidades del modelo (ajustadas por Capa 1 conservadora):
        prob_local            : Prob. de victoria local
        prob_empate           : Prob. de empate
        prob_visitante        : Prob. de victoria visitante

    Probabilidades originales (sin ajuste):
        prob_local_original   : Prob. local antes del ajuste conservador
        prob_empate_original  : Prob. empate antes del ajuste conservador
        prob_visitante_original: Prob. visitante antes del ajuste conservador

    Resultado predicho:
        resultado_predicho    : 'Local', 'Empate' o 'Visitante'
        confianza             : Probabilidad de la opcion predicha (max de las 3)

    Valor de mercado:
        diferencia_valor      : Edge entre modelo y mercado para la opcion predicha
        prob_mercado_local    : Probabilidad implicita de la cuota local
        prob_mercado_empate   : Probabilidad implicita de la cuota empate
        prob_mercado_visitante: Probabilidad implicita de la cuota visitante

    Forma reciente:
        forma_local           : Ej: '3W-1D-1L'
        forma_visitante       : Ej: '2W-2D-1L'
    """
    partido: Partido
    prob_local: float
    prob_empate: float
    prob_visitante: float
    resultado_predicho: str
    confianza: float
    diferencia_valor: float
    prob_mercado_local: float
    prob_mercado_empate: float
    prob_mercado_visitante: float
    forma_local: str
    forma_visitante: str
    prob_local_original: float = 0.0
    prob_empate_original: float = 0.0
    prob_visitante_original: float = 0.0
    mercados_binarios: Optional[PrediccionBinaria] = None

    def es_alta_confianza(self) -> bool:
        """True si la confianza es >= 60%."""
        return self.confianza >= 0.60

    def es_media_confianza(self) -> bool:
        """True si la confianza esta entre 50% y 60%."""
        return 0.50 <= self.confianza < 0.60

    def es_baja_confianza(self) -> bool:
        """True si la confianza es < 50%."""
        return self.confianza < 0.50

    def tiene_edge(self, umbral: float = 0.03) -> bool:
        """True si el edge supera el umbral indicado (default 3%)."""
        return self.diferencia_valor > umbral

    def a_dict(self) -> dict:
        """
        Convierte la prediccion al formato de diccionario usado por
        generar_pdf, generar_excel y el sistema de value betting.
        Mantiene compatibilidad total con el codigo existente.
        """
        return {
            'local': self.partido.local,
            'visitante': self.partido.visitante,
            'cuota_h': self.partido.cuota_h,
            'cuota_d': self.partido.cuota_d,
            'cuota_a': self.partido.cuota_a,
            'prob_local': self.prob_local,
            'prob_empate': self.prob_empate,
            'prob_visitante': self.prob_visitante,
            'prob_local_original': self.prob_local_original,
            'prob_empate_original': self.prob_empate_original,
            'prob_visitante_original': self.prob_visitante_original,
            'prediccion': self.resultado_predicho,
            'confianza': self.confianza,
            'diferencia_valor': self.diferencia_valor,
            'prob_mercado_local': self.prob_mercado_local,
            'prob_mercado_empate': self.prob_mercado_empate,
            'prob_mercado_visitante': self.prob_mercado_visitante,
            'forma_local': self.forma_local,
            'forma_visitante': self.forma_visitante,
        }

    @classmethod
    def desde_dict(cls, d: dict) -> 'Prediccion':
        """
        Crea una Prediccion desde el diccionario devuelto por predecir_partido().
        Util para convertir resultados existentes al nuevo formato.
        """
        partido = Partido(
            local=d['local'],
            visitante=d['visitante'],
            cuota_h=d['cuota_h'],
            cuota_d=d['cuota_d'],
            cuota_a=d['cuota_a'],
        )
        return cls(
            partido=partido,
            prob_local=d['prob_local'],
            prob_empate=d['prob_empate'],
            prob_visitante=d['prob_visitante'],
            prob_local_original=d.get('prob_local_original', d['prob_local']),
            prob_empate_original=d.get('prob_empate_original', d['prob_empate']),
            prob_visitante_original=d.get('prob_visitante_original', d['prob_visitante']),
            resultado_predicho=d['prediccion'],
            confianza=d['confianza'],
            diferencia_valor=d['diferencia_valor'],
            prob_mercado_local=d['prob_mercado_local'],
            prob_mercado_empate=d['prob_mercado_empate'],
            prob_mercado_visitante=d['prob_mercado_visitante'],
            forma_local=d['forma_local'],
            forma_visitante=d['forma_visitante'],
        )


# ============================================================================
# CONFIGJORNADA — Configuracion de la jornada a predecir
# ============================================================================

@dataclass
class ConfigJornada:
    """
    Configuracion completa de una jornada de prediccion.

    Atributos:
        numero   : Numero de jornada (ej: 15)
        partidos : Lista de objetos Partido a predecir

    Uso directo:
        from models import ConfigJornada, Partido
        from jornada_config import NUMERO_JORNADA, PARTIDOS_JORNADA

        config = ConfigJornada(numero=NUMERO_JORNADA, partidos=PARTIDOS_JORNADA)

    Uso con lista de dicts (compatibilidad con formato anterior):
        config = ConfigJornada.desde_lista_dicts(15, lista_de_dicts)
    """
    numero: int
    partidos: list = field(default_factory=list)

    def __post_init__(self):
        if self.numero < 1 or self.numero > 38:
            raise ValueError(f"Numero de jornada invalido: {self.numero}. Debe estar entre 1 y 38.")
        if not self.partidos:
            raise ValueError("La jornada debe tener al menos un partido.")

    def __len__(self) -> int:
        return len(self.partidos)

    @classmethod
    def desde_lista_dicts(cls, numero: int, lista: list) -> 'ConfigJornada':
        """
        Crea una ConfigJornada desde una lista de diccionarios con formato:
            [{'local': ..., 'visitante': ..., 'cuota_h': ..., ...}, ...]
        """
        partidos = [
            Partido(
                local=d['local'],
                visitante=d['visitante'],
                cuota_h=float(d['cuota_h']),
                cuota_d=float(d['cuota_d']),
                cuota_a=float(d['cuota_a']),
            )
            for d in lista
        ]
        return cls(numero=numero, partidos=partidos)
