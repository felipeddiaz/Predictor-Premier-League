# -*- coding: utf-8 -*-
"""Entrena modelo binario Over/Under 9.5 corners."""

from config import ARCHIVO_MODELO_CORNERS, ARCHIVO_FEATURES_CORNERS
from pipeline.mercados_binarios import entrenar_mercado_binario


FEATURES_CORNERS = [
    'HT_CornersFor5', 'AT_CornersFor5', 'HT_CornersAgainst5', 'AT_CornersAgainst5',
    'HT_CornersTotal5', 'AT_CornersTotal5', 'H2H_Corners_Avg',
    'HT_Corner_Dominance5', 'AT_Corner_Dominance5', 'Corner_Dominance_Diff',
    'HT_Fouls5', 'AT_Fouls5',
    'HT_Shots5', 'AT_Shots5', 'HT_ShotsTarget5', 'AT_ShotsTarget5',
    'HT_AvgShotsTarget', 'AT_AvgShotsTarget', 'HT_AvgGoals', 'AT_AvgGoals',
    'HT_xG_Avg', 'AT_xG_Avg', 'xG_Total',
    'HT_Elo', 'AT_Elo', 'Elo_Diff',
    'HT_Position', 'AT_Position', 'Position_Diff', 'HT_Pressure', 'AT_Pressure',
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
