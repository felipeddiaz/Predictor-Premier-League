# -*- coding: utf-8 -*-
"""Entrena modelo binario Over/Under 9.5 corners."""

from config import ARCHIVO_MODELO_CORNERS, ARCHIVO_FEATURES_CORNERS
from pipeline.mercados_binarios import entrenar_mercado_binario


FEATURES_CORNERS = [
    'HT_CornersFor5',
    'HT_CornersTotal5',
    'Corner_Dominance_Diff',
    'AT_Fouls5',
    'HT_AvgShotsTarget', 'AT_AvgShotsTarget', 'HT_AvgGoals', 'AT_AvgGoals',
    'AT_xG_Avg',
    'HT_Elo', 'Elo_Diff',
    'HT_Position', 'AT_Position', 'Position_Diff', 'HT_Pressure',
    'HT_Form_W', 'AT_Form_W', 'HT_Form_L', 'AT_Form_L',
]


def main():
    entrenar_mercado_binario(
        nombre_mercado='CORNERS',
        target_col='TARGET_CORNERS95',
        features=FEATURES_CORNERS,
        over_col=None,
        under_col=None,
        archivo_modelo=ARCHIVO_MODELO_CORNERS,
        archivo_features=ARCHIVO_FEATURES_CORNERS,
        target_builder=lambda df: ((df['HC'].fillna(0) + df['AC'].fillna(0)) > 9.5).astype(int),
    )


if __name__ == '__main__':
    main()
