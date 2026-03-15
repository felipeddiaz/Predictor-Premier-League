# -*- coding: utf-8 -*-
"""Entrena modelo binario Over/Under 2.5 goles."""

from config import ARCHIVO_MODELO_OU, ARCHIVO_FEATURES_OU
from pipeline.mercados_binarios import entrenar_mercado_binario


FEATURES_OU = [
    'HT_AvgGoals', 'AT_AvgGoals', 'HT_AvgShotsTarget', 'AT_AvgShotsTarget',
    'HT_xG_Avg', 'AT_xG_Avg', 'HT_xGA_Avg', 'AT_xGA_Avg', 'xG_Diff', 'xG_Total',
    'H2H_Total_Goals_Avg', 'H2H_BTTS_Rate',
    'HT_BTTS_Rate5', 'AT_BTTS_Rate5',
    'AT_Shots5', 'HT_ShotsTarget5', 'AT_ShotsTarget5',
    'HT_xG_Residual5',
    'AT_Days_Rest',
    'HT_Elo', 'Elo_Diff',
]


def main():
    entrenar_mercado_binario(
        nombre_mercado='OVER_UNDER',
        target_col='TARGET_OVER25',
        features=FEATURES_OU,
        over_col='B365>2.5',
        under_col='B365<2.5',
        archivo_modelo=ARCHIVO_MODELO_OU,
        archivo_features=ARCHIVO_FEATURES_OU,
        target_builder=lambda df: ((df['FTHG'].fillna(0) + df['FTAG'].fillna(0)) > 2.5).astype(int),
    )


if __name__ == '__main__':
    main()
