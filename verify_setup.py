"""
verify_setup.py
Verifica que el proyecto use un único CSV consolidado con H2H y features derivadas.
Comprueba:
 - existencia de ./datos/procesados/premier_league_con_features.csv
 - presencia de columnas H2H_
 - presencia de columnas derivadas (Goal_Diff, Form_Diff, Shots_Diff)
 - imprime resumen de columnas y primeras filas
"""

import os
import pandas as pd

CSV_PATH = os.path.join('datos', 'procesados', 'premier_league_con_features.csv')


def main():
    print('\nVerificando setup del proyecto...')
    print('CSV objetivo:', CSV_PATH)

    if not os.path.exists(CSV_PATH):
        print('❌ Archivo no encontrado. Asegúrate de haber renombrado el CSV H2H a premier_league_con_features.csv')
        return 1

    print('✅ Archivo encontrado')

    df = pd.read_csv(CSV_PATH)
    cols = df.columns.tolist()

    h2h_cols = [c for c in cols if c.startswith('H2H_')]
    derivadas = [c for c in ['Goal_Diff', 'Form_Diff', 'Shots_Diff'] if c in cols]

    print(f"\nColumnas totales: {len(cols)}")
    print(f"Columnas H2H detectadas: {len(h2h_cols)} -> {h2h_cols[:20]}")
    print(f"Features derivadas presentes: {derivadas}")

    # Muestra primeras filas de las columnas relevantes
    mostrar = h2h_cols[:5] + derivadas[:3]
    mostrar = [c for c in mostrar if c in cols]

    if mostrar:
        print('\nPrimeras filas (columnas clave):')
        print(df[mostrar].head(5).to_string(index=False))

    print('\n✅ Verificación completada')
    return 0


if __name__ == '__main__':
    exit(main())