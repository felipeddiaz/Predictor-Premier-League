# agregar_features_derivadas.py
"""
Agrega features de rendimiento derivadas al dataset procesado.
Ejecutar una sola vez después de 01_preparar_datos.py si las columnas
Goal_Diff, Form_Diff y Shots_Diff no están presentes.
"""
import pandas as pd

from config import ARCHIVO_FEATURES

if __name__ == '__main__':
    df = pd.read_csv(ARCHIVO_FEATURES)

    # Solo agregar las 3 de rendimiento que faltan
    df['Goal_Diff'] = df['HT_AvgGoals'] - df['AT_AvgGoals']
    df['Form_Diff'] = df['HT_Form_W'] - df['AT_Form_W']
    df['Shots_Diff'] = df['HT_AvgShotsTarget'] - df['AT_AvgShotsTarget']

    df.to_csv(ARCHIVO_FEATURES, index=False)

    print("✅ Agregadas: Goal_Diff, Form_Diff, Shots_Diff")
    print("🚀 Ejecuta: python 03_entrenar_con_cuotas.py")
