# -*- coding: utf-8 -*-
"""Entrena modelo binario Over/Under 3.5 tarjetas amarillas."""

from config import ARCHIVO_MODELO_TARJETAS, ARCHIVO_FEATURES_TARJETAS
from pipeline.mercados_binarios import entrenar_mercado_binario


FEATURES_TARJETAS = [
    'HT_YellowAvg5', 'AT_YellowAvg5', 'HT_YellowFor5',
    'HT_RedRate5', 'AT_RedRate5',
    'Ref_Yellow_Avg', 'Ref_Cards_Total_Avg', 'H2H_Yellow_Avg',
    'Ref_Aggressiveness_Interaction',
    'HT_Form_D', 'HT_Form_L', 'AT_Form_W', 'AT_Form_D', 'AT_Form_L',
    'HT_Days_Rest', 'AT_Days_Rest', 'Rest_Diff',
]


def main():
    entrenar_mercado_binario(
        nombre_mercado='TARJETAS',
        target_col='TARGET_YELLOW35',
        features=FEATURES_TARJETAS,
        over_col=None,
        under_col=None,
        archivo_modelo=ARCHIVO_MODELO_TARJETAS,
        archivo_features=ARCHIVO_FEATURES_TARJETAS,
        target_builder=lambda df: ((df['HY'].fillna(0) + df['AY'].fillna(0)) > 3.5).astype(int),
    )


if __name__ == '__main__':
    main()
