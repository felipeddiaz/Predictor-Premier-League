# -*- coding: utf-8 -*-
"""
CLV (Closing Line Value) Tracker

LA métrica que valida si el modelo tiene edge real sobre el mercado.

Para cada apuesta:
  - prob_modelo: lo que el modelo predecía al momento de apostar
  - prob_apertura: probabilidad implícita de la cuota de apertura (sin vig)
  - prob_cierre: probabilidad implícita de la cuota de cierre (sin vig)
  - CLV = prob_cierre - prob_apertura

Si el cierre se movió hacia tu predicción, tienes edge real.
El mercado es eficiente al cierre — si tu modelo consistentemente
"anticipa" el movimiento del cierre, tienes una ventaja estructural.

Uso:
    from core.clv_tracker import CLVTracker

    tracker = CLVTracker()
    tracker.registrar(
        fecha='2025-01-15',
        local='Arsenal', visitante='Chelsea',
        resultado_predicho='Local',
        prob_modelo=0.55,
        cuota_apertura_h=1.90, cuota_apertura_d=3.50, cuota_apertura_a=4.20,
        cuota_cierre_h=1.85, cuota_cierre_d=3.60, cuota_cierre_a=4.30,
        resultado_real=0,  # 0=Local, 1=Empate, 2=Visitante
        stake=50.0,
    )
    tracker.guardar()
    tracker.resumen()
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

from config import RUTA_MODELOS


# Ruta del CSV de tracking
RUTA_CLV_CSV = os.path.join(RUTA_MODELOS, 'clv_tracking.csv')

# Columnas del tracker
COLUMNAS = [
    'fecha',
    'local',
    'visitante',
    'resultado_predicho',   # 'Local', 'Empate', 'Visitante'
    'prob_modelo',          # probabilidad del modelo para el resultado predicho
    'cuota_apertura',       # cuota de apertura del resultado predicho
    'cuota_cierre',         # cuota de cierre del resultado predicho
    'prob_apertura_fair',   # prob implícita apertura (sin vig)
    'prob_cierre_fair',     # prob implícita cierre (sin vig)
    'clv',                  # prob_cierre_fair - prob_apertura_fair
    'edge_vs_apertura',     # prob_modelo - prob_apertura_fair
    'edge_vs_cierre',       # prob_modelo - prob_cierre_fair
    'resultado_real',       # 0, 1, 2
    'acierto',              # bool: resultado_predicho == resultado_real
    'stake',
    'cuota_apostada',       # cuota a la que se apostó (apertura)
    'pnl',                  # profit/loss de la apuesta
    'pnl_acumulado',        # P&L acumulado
    'timestamp',            # momento del registro
]


def _eliminar_vig_trio(cuota_h, cuota_d, cuota_a):
    """Elimina vig de un trio de cuotas. Retorna (fair_h, fair_d, fair_a)."""
    p_h = 1.0 / cuota_h
    p_d = 1.0 / cuota_d
    p_a = 1.0 / cuota_a
    total = p_h + p_d + p_a
    return p_h / total, p_d / total, p_a / total


class CLVTracker:
    """Rastrea el Closing Line Value de cada apuesta para validar edge."""

    def __init__(self, ruta_csv: str = RUTA_CLV_CSV):
        self._ruta = ruta_csv
        if os.path.exists(ruta_csv):
            self._df = pd.read_csv(ruta_csv)
        else:
            self._df = pd.DataFrame(columns=COLUMNAS)

    def registrar(
        self,
        fecha: str,
        local: str,
        visitante: str,
        resultado_predicho: str,
        prob_modelo: float,
        cuota_apertura_h: float,
        cuota_apertura_d: float,
        cuota_apertura_a: float,
        cuota_cierre_h: float,
        cuota_cierre_d: float,
        cuota_cierre_a: float,
        resultado_real: int,
        stake: float,
    ):
        """
        Registra una apuesta con datos de apertura y cierre.

        Args:
            resultado_predicho: 'Local', 'Empate', o 'Visitante'
            resultado_real: 0=Local, 1=Empate, 2=Visitante
        """
        # Probabilities sin vig
        fair_ap_h, fair_ap_d, fair_ap_a = _eliminar_vig_trio(
            cuota_apertura_h, cuota_apertura_d, cuota_apertura_a
        )
        fair_ci_h, fair_ci_d, fair_ci_a = _eliminar_vig_trio(
            cuota_cierre_h, cuota_cierre_d, cuota_cierre_a
        )

        # Mapear al resultado predicho
        idx_map = {'Local': 0, 'Empate': 1, 'Visitante': 2}
        idx = idx_map[resultado_predicho]

        fair_ap = [fair_ap_h, fair_ap_d, fair_ap_a][idx]
        fair_ci = [fair_ci_h, fair_ci_d, fair_ci_a][idx]
        cuota_ap = [cuota_apertura_h, cuota_apertura_d, cuota_apertura_a][idx]
        cuota_ci = [cuota_cierre_h, cuota_cierre_d, cuota_cierre_a][idx]

        clv = fair_ci - fair_ap
        edge_apertura = prob_modelo - fair_ap
        edge_cierre = prob_modelo - fair_ci
        acierto = (idx == resultado_real)

        if acierto:
            pnl = stake * (cuota_ap - 1)
        else:
            pnl = -stake

        # P&L acumulado
        pnl_previo = self._df['pnl_acumulado'].iloc[-1] if len(self._df) > 0 else 0.0
        pnl_acum = pnl_previo + pnl

        fila = {
            'fecha': fecha,
            'local': local,
            'visitante': visitante,
            'resultado_predicho': resultado_predicho,
            'prob_modelo': round(prob_modelo, 4),
            'cuota_apertura': round(cuota_ap, 2),
            'cuota_cierre': round(cuota_ci, 2),
            'prob_apertura_fair': round(fair_ap, 4),
            'prob_cierre_fair': round(fair_ci, 4),
            'clv': round(clv, 4),
            'edge_vs_apertura': round(edge_apertura, 4),
            'edge_vs_cierre': round(edge_cierre, 4),
            'resultado_real': resultado_real,
            'acierto': acierto,
            'stake': round(stake, 2),
            'cuota_apostada': round(cuota_ap, 2),
            'pnl': round(pnl, 2),
            'pnl_acumulado': round(pnl_acum, 2),
            'timestamp': datetime.now().isoformat(),
        }

        self._df = pd.concat([self._df, pd.DataFrame([fila])], ignore_index=True)
        return fila

    def registrar_batch_historico(self, df_historico: pd.DataFrame):
        """
        Registra apuestas históricas desde un DataFrame con cuotas de apertura y cierre.

        El DataFrame debe tener columnas:
            Date, HomeTeam, AwayTeam, FTR_numeric,
            B365H, B365D, B365A (apertura),
            B365CH, B365CD, B365CA (cierre)

        Y las columnas de probabilidades del modelo (generadas por predict_proba):
            prob_0, prob_1, prob_2
        """
        registros = 0
        for _, row in df_historico.iterrows():
            # Determinar resultado predicho (argmax de probs)
            probs = [row['prob_0'], row['prob_1'], row['prob_2']]
            idx_pred = int(np.argmax(probs))
            resultado_predicho = ['Local', 'Empate', 'Visitante'][idx_pred]

            # Verificar que hay cuotas de cierre
            cierre_cols = ['B365CH', 'B365CD', 'B365CA']
            if not all(c in row.index and pd.notna(row[c]) and row[c] > 1 for c in cierre_cols):
                continue

            self.registrar(
                fecha=str(row.get('Date', '')),
                local=row['HomeTeam'],
                visitante=row['AwayTeam'],
                resultado_predicho=resultado_predicho,
                prob_modelo=probs[idx_pred],
                cuota_apertura_h=row['B365H'],
                cuota_apertura_d=row['B365D'],
                cuota_apertura_a=row['B365A'],
                cuota_cierre_h=row['B365CH'],
                cuota_cierre_d=row['B365CD'],
                cuota_cierre_a=row['B365CA'],
                resultado_real=int(row['FTR_numeric']),
                stake=10.0,
            )
            registros += 1

        print(f"   CLV Tracker: {registros} apuestas registradas")
        return registros

    def guardar(self):
        """Guarda el tracker a CSV."""
        self._df.to_csv(self._ruta, index=False)
        print(f"   CLV tracking guardado: {self._ruta} ({len(self._df)} registros)")

    def resumen(self):
        """Imprime resumen del CLV tracking."""
        if len(self._df) == 0:
            print("   CLV Tracker: sin registros")
            return

        df = self._df
        n = len(df)

        print("\n" + "=" * 70)
        print("CLV TRACKER — RESUMEN")
        print("=" * 70)

        # CLV stats
        clv_mean = df['clv'].mean()
        clv_positive = (df['clv'] > 0).sum()
        clv_pct_positive = clv_positive / n * 100

        print(f"\n   Total apuestas:         {n}")
        print(f"   CLV promedio:           {clv_mean:+.4f} ({'POSITIVO' if clv_mean > 0 else 'NEGATIVO'})")
        print(f"   CLV positivo:           {clv_positive}/{n} ({clv_pct_positive:.1f}%)")

        # Edge stats
        edge_ap_mean = df['edge_vs_apertura'].mean()
        edge_ci_mean = df['edge_vs_cierre'].mean()
        print(f"\n   Edge vs apertura (avg): {edge_ap_mean:+.4f}")
        print(f"   Edge vs cierre (avg):   {edge_ci_mean:+.4f}")

        # P&L stats
        aciertos = int(df['acierto'].sum())
        hit_rate = aciertos / n * 100
        pnl_total = df['pnl'].sum()
        roi = pnl_total / df['stake'].sum() * 100 if df['stake'].sum() > 0 else 0

        print(f"\n   Hit rate:               {aciertos}/{n} ({hit_rate:.1f}%)")
        print(f"   P&L total:              {pnl_total:+.2f}")
        print(f"   ROI:                    {roi:+.1f}%")

        # Interpretación
        print(f"\n   INTERPRETACION:")
        if clv_mean > 0.01:
            print(f"   CLV > 0 indica que el mercado se mueve HACIA tu prediccion.")
            print(f"   Esto valida que el modelo tiene edge estructural.")
        elif clv_mean > 0:
            print(f"   CLV marginalmente positivo. Edge posible pero no concluyente.")
            print(f"   Necesitas mas datos para confirmar.")
        else:
            print(f"   CLV <= 0 sugiere que el modelo NO anticipa movimientos del mercado.")
            print(f"   El edge aparente puede ser ruido o artefacto de las cuotas.")

        print("=" * 70)

    @property
    def datos(self) -> pd.DataFrame:
        """Retorna el DataFrame de tracking."""
        return self._df.copy()
