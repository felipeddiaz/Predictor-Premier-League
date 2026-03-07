"""
Exporta fbref_fixtures.csv a archivos individuales por temporada y competicion
en datos/raw/historicos/
"""
import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, 'datos', 'raw', 'fbref_fixtures.csv')
DEST = os.path.join(ROOT, 'datos', 'raw', 'historicos')

COMP_SLUG = {
    'Champions Lg': 'champions_lg',
    'Europa Lg':    'europa_lg',
    'FA Cup':       'fa_cup',
    'EFL Cup':      'efl_cup',
}

def main():
    df = pd.read_csv(SRC)
    print(f"Leido {SRC}: {len(df)} filas")

    os.makedirs(DEST, exist_ok=True)

    written = []
    for (season, comp), group in df.groupby(['Season', 'Comp']):
        slug  = COMP_SLUG.get(comp, comp.lower().replace(' ', '_'))
        fname = f"{slug}_{season}.csv"
        fpath = os.path.join(DEST, fname)
        group.to_csv(fpath, index=False)
        written.append((fname, len(group)))

    print(f"\nEscritos {len(written)} archivos en {DEST}:")
    for fname, n in sorted(written):
        print(f"  {fname}  ({n} filas)")

    total = sum(n for _, n in written)
    print(f"\nTotal filas exportadas: {total}")
    print("Listo.")

if __name__ == '__main__':
    main()
