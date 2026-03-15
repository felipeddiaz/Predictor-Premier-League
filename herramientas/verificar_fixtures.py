"""
Verifica que el consolidador reproduce exactamente fbref_fixtures.csv.
Ejecuta consolidar_fixtures en memoria y compara con el CSV existente.
"""
import os
import sys
import glob
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, 'datos', 'raw', 'historicos')
DEST = os.path.join(ROOT, 'datos', 'raw', 'fbref_fixtures.csv')

REQUIRED_COLS = ['Season', 'Date', 'Team', 'Comp', 'Venue', 'Opponent']


def load_historicos():
    frames = []
    for fpath in sorted(glob.glob(os.path.join(SRC, '*.csv'))):
        df = pd.read_csv(fpath)
        frames.append(df[REQUIRED_COLS])
    return pd.concat(frames, ignore_index=True)


def canonical(df):
    """Sort deterministically for comparison."""
    return (
        df[REQUIRED_COLS]
        .sort_values(REQUIRED_COLS)
        .reset_index(drop=True)
    )


def main():
    original    = pd.read_csv(DEST)
    consolidated = load_historicos()

    print(f"Original:     {len(original)} filas")
    print(f"Consolidado:  {len(consolidated)} filas")

    o = canonical(original)
    c = canonical(consolidated)

    if o.equals(c):
        print("\nOK: Los datasets son identicos.")
        sys.exit(0)

    # Find differences
    merged = o.merge(c, on=REQUIRED_COLS, how='outer', indicator=True)
    only_orig = merged[merged['_merge'] == 'left_only']
    only_cons = merged[merged['_merge'] == 'right_only']

    print(f"\nDIFERENCIAS:")
    if not only_orig.empty:
        print(f"  Solo en original ({len(only_orig)} filas):")
        print(only_orig.to_string())
    if not only_cons.empty:
        print(f"  Solo en consolidado ({len(only_cons)} filas):")
        print(only_cons.to_string())

    sys.exit(1)


if __name__ == '__main__':
    main()
