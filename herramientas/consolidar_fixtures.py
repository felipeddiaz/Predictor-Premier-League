"""
Consolidador de fixtures: lee todos los CSVs individuales de
datos/raw/historicos/ y produce datos/raw/fbref_fixtures.csv

Uso:
    python herramientas/consolidar_fixtures.py [--dry-run]

Opciones:
    --dry-run   Solo muestra estadisticas sin sobreescribir el CSV de salida.

Estructura esperada de cada archivo fuente:
    datos/raw/historicos/{comp}_{season}.csv
    Columnas requeridas: Season, Date, Team, Comp, Venue, Opponent

Columnas del CSV de salida (orden canonico):
    Season, Date, Team, Comp, Venue, Opponent
"""
import os
import sys
import glob
import pandas as pd

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC    = os.path.join(ROOT, 'datos', 'raw', 'historicos')
DEST   = os.path.join(ROOT, 'datos', 'raw', 'fbref_fixtures.csv')

REQUIRED_COLS = ['Season', 'Date', 'Team', 'Comp', 'Venue', 'Opponent']
OUTPUT_COLS   = REQUIRED_COLS  # same order

# Orden de temporadas para el CSV de salida
SEASON_ORDER = [
    '2016-17', '2017-18', '2018-19', '2019-20', '2020-21',
    '2021-22', '2022-23', '2023-24', '2024-25', '2025-26',
]

# Orden de competiciones dentro de cada temporada
COMP_ORDER = ['Champions Lg', 'Europa Lg', 'FA Cup', 'EFL Cup']


def load_all():
    pattern = os.path.join(SRC, '*.csv')
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"ERROR: No se encontraron CSVs en {SRC}", file=sys.stderr)
        sys.exit(1)

    frames = []
    for fpath in files:
        fname = os.path.basename(fpath)
        df    = pd.read_csv(fpath)

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            print(f"  ADVERTENCIA {fname}: faltan columnas {missing} — omitido")
            continue

        frames.append(df[REQUIRED_COLS])
        print(f"  {fname}: {len(df)} filas")

    return pd.concat(frames, ignore_index=True)


def sort_df(df):
    season_cat = pd.CategoricalDtype(categories=SEASON_ORDER, ordered=True)
    comp_cat   = pd.CategoricalDtype(categories=COMP_ORDER,   ordered=True)

    df = df.copy()
    df['Season'] = df['Season'].astype(season_cat)
    df['Comp']   = df['Comp'].astype(comp_cat)
    df = df.sort_values(['Season', 'Comp', 'Date']).reset_index(drop=True)
    df['Season'] = df['Season'].astype(str)
    df['Comp']   = df['Comp'].astype(str)
    return df


def main():
    dry_run = '--dry-run' in sys.argv

    print(f"Leyendo archivos de {SRC} ...\n")
    combined = load_all()

    print(f"\nTotal filas cargadas: {len(combined)}")
    print("\nDesglose por temporada y competicion:")
    print(combined.groupby(['Season', 'Comp']).size().to_string())

    combined = sort_df(combined)

    if dry_run:
        print("\n[--dry-run] No se sobreescribe el CSV de salida.")
        return

    # Backup del CSV existente antes de sobreescribir
    if os.path.exists(DEST):
        backup = DEST + '.consolidador_backup'
        import shutil
        shutil.copy2(DEST, backup)
        print(f"\nBackup creado: {backup}")

    combined.to_csv(DEST, index=False)
    print(f"\nEscrito: {DEST}  ({len(combined)} filas, {len(combined.columns)} columnas)")
    print("Listo.")


if __name__ == '__main__':
    main()
