# -*- coding: utf-8 -*-
"""
completar_fixtures_historicos.py
=================================
Descarga los fixtures históricos de UEFA Champions League y Europa League
(temporadas 2016-17 a 2021-22) desde el repositorio público openfootball/
champions-league en GitHub, filtra los partidos de equipos de la Premier
League y hace merge con datos/raw/fbref_fixtures.csv.

Fuente:
    https://github.com/openfootball/champions-league
    Licencia: CC0 (dominio público)

Formato de los archivos .txt:
    Wed Oct/19
      20.45  Arsenal FC (ENG)   v PFC Ludogorets (BUL)   6-0 (2-0)

Uso:
    python herramientas/completar_fixtures_historicos.py
    python herramientas/completar_fixtures_historicos.py --dry-run
    python herramientas/completar_fixtures_historicos.py --seasons 2018 2019
"""

import argparse
import os
import re
import sys
import time
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
_DIR_SCRIPT  = os.path.dirname(os.path.abspath(__file__))
_RAIZ        = os.path.dirname(_DIR_SCRIPT)
_CSV_DESTINO = os.path.join(_RAIZ, 'datos', 'raw', 'fbref_fixtures.csv')

# ---------------------------------------------------------------------------
# Fuentes de datos — openfootball/champions-league en GitHub
# ---------------------------------------------------------------------------
_BASE_RAW = (
    'https://raw.githubusercontent.com/openfootball/champions-league/master'
)

# Temporadas a descargar para rellenar el gap 2016-2021
# Clave: año de inicio (ej. 2016 → temporada 2016-17)
# Valor: archivos disponibles en ese directorio
_TEMPORADAS_ARCHIVOS = {
    2016: {'cl': 'cl.txt',  'el': None},    # solo UCL disponible en repo
    2017: {'cl': 'cl.txt',  'el': None},    # solo UCL disponible en repo
    2018: {'cl': 'cl.txt',  'el': None},    # solo UCL disponible en repo
    2019: {'cl': 'cl.txt',  'el': 'el.txt'},
    2020: {'cl': 'cl.txt',  'el': 'el.txt'},
    2021: {'cl': 'cl.txt',  'el': 'el.txt'},
}

_COMP_MAP = {
    'cl': 'Champions Lg',
    'el': 'Europa Lg',
}

# ---------------------------------------------------------------------------
# Equipos de PL con sus variantes en openfootball → nombre canónico
# ---------------------------------------------------------------------------
_TEAM_MAP = {
    # Nombres exactos tal como aparecen en openfootball (sin sufijo país)
    'Arsenal FC':               'Arsenal',
    'Arsenal':                  'Arsenal',
    'Aston Villa':              'Aston Villa',
    'Chelsea FC':               'Chelsea',
    'Chelsea':                  'Chelsea',
    'Everton FC':               'Everton',
    'Everton':                  'Everton',
    'Leeds United':             'Leeds',
    'Leicester City':           'Leicester',
    'Liverpool FC':             'Liverpool',
    'Liverpool':                'Liverpool',
    'Manchester City':          'Man City',
    'Manchester United':        'Man United',
    'Newcastle United':         'Newcastle',
    "Nott'm Forest":            "Nott'm Forest",
    'Nottingham Forest':        "Nott'm Forest",
    'Southampton FC':           'Southampton',
    'Southampton':              'Southampton',
    'Tottenham Hotspur':        'Tottenham',
    'Tottenham':                'Tottenham',
    'West Ham United':          'West Ham',
    'West Ham':                 'West Ham',
    'Wolverhampton Wanderers':  'Wolves',
    'Wolves':                   'Wolves',
    'Brighton & Hove Albion':   'Brighton',
    'Brighton':                 'Brighton',
    'Brentford FC':             'Brentford',
    'Brentford':                'Brentford',
    'Fulham FC':                'Fulham',
    'Fulham':                   'Fulham',
    'Crystal Palace':           'Crystal Palace',
    'Burnley FC':               'Burnley',
    'Burnley':                  'Burnley',
    'Watford FC':               'Watford',
    'Watford':                  'Watford',
    'Bournemouth':              'Bournemouth',
    'AFC Bournemouth':          'Bournemouth',
    'Norwich City':             'Norwich',
    'Sheffield United':         'Sheffield United',
    'Sheffield Utd':            'Sheffield United',
    'Huddersfield Town':        'Huddersfield',
    'Cardiff City':             'Cardiff',
    'Swansea City':             'Swansea',
    'Stoke City':               'Stoke',
    'West Bromwich Albion':     'West Brom',
    'Hull City':                'Hull',
    'Sunderland AFC':           'Sunderland',
    'Sunderland':               'Sunderland',
    'Middlesbrough':            'Middlesbrough',
    'Ipswich Town':             'Ipswich',
    'Luton Town':               'Luton',
    'Queens Park Rangers':      'QPR',
}

_EQUIPOS_PL = set(_TEAM_MAP.values())

# Regex principal para parsear una línea de partido:
#   HH.MM  Equipo Local (XXX)   v  Equipo Visitante (XXX)   N-N (N-N)
# También acepta líneas sin hora (cuando el partido comparte fecha con otro)
_RE_MATCH = re.compile(
    r'^\s*(?:\d{1,2}\.\d{2}\s+)?'      # hora opcional (ej. "20.45  ")
    r'(.+?)\s+v\s+(.+?)'               # "Equipo Local  v  Equipo Visit."
    r'\s+\d+[-–]\d+'                   # resultado "N-N"
    r'(?:\s+a\.e\.t\.)?'               # posible "a.e.t."
    r'(?:\s*\(\d+[-–]\d+(?:,\s*\d+[-–]\d+)?\))?'  # resultado parcial "(N-N)"
    r'\s*$'
)

# Regex para extraer el nombre del equipo sin el sufijo "(ENG)", "(ESP)", etc.
_RE_EQUIPO = re.compile(r'^(.*?)\s*\([A-Z]{2,3}\)\s*$')

# Regex para detectar una línea de fecha
# Formatos: "Wed Sep/13 2016", "Tue Oct/27", "Sat May/29 2021"
_RE_FECHA = re.compile(
    r'^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+'
    r'(\w{3})/(\d{1,2})(?:\s+(\d{4}))?\s*$'
)

_MES_MAP = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
}


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------

def limpiar_nombre(raw: str) -> str:
    """Elimina el sufijo de país '(ENG)' y espacios extra."""
    raw = raw.strip()
    m = _RE_EQUIPO.match(raw)
    return m.group(1).strip() if m else raw


def normalizar_equipo(nombre: str) -> str | None:
    """
    Devuelve el nombre canónico si el equipo es de PL, None si no lo es.
    """
    return _TEAM_MAP.get(nombre)


def temporada_str(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"


def descargar_texto(url: str, sleep_s: float = 1.0) -> str | None:
    """Descarga un archivo de texto desde una URL."""
    time.sleep(sleep_s)
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.text
    except requests.exceptions.RequestException as e:
        print(f"    ERROR al descargar {url}: {e}")
        return None


def parsear_txt(texto: str, comp_nombre: str, season_year: int) -> list[dict]:
    """
    Parsea el contenido de un archivo .txt de openfootball y extrae
    los partidos donde al menos un equipo es de PL.

    Args:
        texto:        Contenido del archivo .txt.
        comp_nombre:  Nombre canónico de la competición ('Champions Lg', etc.).
        season_year:  Año de inicio de la temporada (ej. 2016).

    Returns:
        Lista de dicts con columnas: Season, Date, Team, Comp, Venue, Opponent.
    """
    filas = []
    fecha_actual = None
    anio_actual  = season_year  # el año va avanzando a lo largo del archivo

    for linea in texto.splitlines():
        # ¿Es una línea de fecha?
        m_fecha = _RE_FECHA.match(linea)
        if m_fecha:
            mes_str, dia_str, anio_str = m_fecha.groups()
            mes = _MES_MAP.get(mes_str, 0)
            dia = int(dia_str)
            if anio_str:
                anio_actual = int(anio_str)
            # Heurística: si el mes es enero-julio y el año no cambió aún,
            # probablemente ya estamos en el año siguiente de la temporada
            elif mes <= 7 and anio_actual == season_year:
                anio_actual = season_year + 1
            try:
                fecha_actual = pd.Timestamp(year=anio_actual, month=mes, day=dia)
            except Exception:
                fecha_actual = None
            continue

        # ¿Es una línea de partido?
        if fecha_actual is None:
            continue

        m_partido = _RE_MATCH.match(linea)
        if not m_partido:
            continue

        home_raw = limpiar_nombre(m_partido.group(1))
        away_raw = limpiar_nombre(m_partido.group(2))

        home_canon = normalizar_equipo(home_raw)
        away_canon = normalizar_equipo(away_raw)

        # Solo nos interesan partidos donde al menos un equipo es de PL
        if home_canon is None and away_canon is None:
            continue

        season_label = temporada_str(season_year)

        # Fila para el equipo local (si es de PL)
        if home_canon:
            filas.append({
                'Season':   season_label,
                'Date':     fecha_actual,
                'Team':     home_canon,
                'Comp':     comp_nombre,
                'Venue':    'Home',
                'Opponent': away_canon or away_raw,
            })

        # Fila para el equipo visitante (si es de PL)
        if away_canon:
            filas.append({
                'Season':   season_label,
                'Date':     fecha_actual,
                'Team':     away_canon,
                'Comp':     comp_nombre,
                'Venue':    'Away',
                'Opponent': home_canon or home_raw,
            })

    return filas


def descargar_todo(temporadas: list[int], dry_run: bool = False,
                   sleep_s: float = 1.0) -> pd.DataFrame:
    """
    Descarga y parsea todos los archivos para las temporadas indicadas.
    """
    todas_las_filas = []
    total = sum(
        1 + (1 if _TEMPORADAS_ARCHIVOS[y]['el'] else 0)
        for y in temporadas
        if y in _TEMPORADAS_ARCHIVOS
    )
    n = 0

    for year in sorted(temporadas):
        if year not in _TEMPORADAS_ARCHIVOS:
            print(f"  AVISO: temporada {year} no está configurada, se omite.")
            continue

        archivos = _TEMPORADAS_ARCHIVOS[year]
        season_label = temporada_str(year)
        dir_temp = f"{year}-{str(year + 1)[-2:]}"

        for tipo, filename in archivos.items():
            if filename is None:
                continue
            n += 1
            comp = _COMP_MAP[tipo]
            url  = f"{_BASE_RAW}/{dir_temp}/{filename}"
            print(f"  [{n:2d}/{total}] {comp} {season_label} ...", end='', flush=True)

            if dry_run:
                print(f" [DRY RUN] {url}")
                continue

            texto = descargar_texto(url, sleep_s)
            if texto is None:
                print(f" no encontrado (404)")
                continue

            filas = parsear_txt(texto, comp, year)
            todas_las_filas.extend(filas)
            print(f" {len(filas)} filas PL")

    if not todas_las_filas:
        return pd.DataFrame()

    df = pd.DataFrame(todas_las_filas)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def merge_con_existente(df_nuevo: pd.DataFrame,
                        csv_destino: str) -> pd.DataFrame:
    """
    Carga el CSV existente y hace merge eliminando duplicados.
    Prioriza los datos de api-football (más completos) sobre openfootball.
    """
    if os.path.exists(csv_destino):
        df_existente = pd.read_csv(csv_destino, parse_dates=['Date'])
        print(f"\n  CSV existente: {len(df_existente)} filas "
              f"({df_existente['Season'].nunique()} temporadas)")

        # Concatenar — los duplicados exactos (Season+Date+Team+Comp) se eliminan
        df_combined = pd.concat([df_existente, df_nuevo], ignore_index=True)
        antes = len(df_combined)
        df_combined = df_combined.drop_duplicates(
            subset=['Season', 'Date', 'Team', 'Comp']
        )
        eliminados = antes - len(df_combined)
        if eliminados:
            print(f"  Eliminados {eliminados} duplicados")
    else:
        df_combined = df_nuevo

    df_combined = df_combined.sort_values(['Team', 'Date']).reset_index(drop=True)
    return df_combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Descarga fixtures históricos de UCL/UEL (2016-2021) desde '
            'openfootball/champions-league y hace merge con fbref_fixtures.csv'
        )
    )
    parser.add_argument(
        '--seasons', nargs='+', type=int,
        default=list(_TEMPORADAS_ARCHIVOS.keys()),
        metavar='YEAR',
        help='Años de inicio de temporada a descargar (default: 2016-2021)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Muestra URLs sin descargar ni modificar el CSV',
    )
    parser.add_argument(
        '--csv', default=_CSV_DESTINO,
        help=f'Ruta del CSV de salida (default: {_CSV_DESTINO})',
    )
    parser.add_argument(
        '--sleep', type=float, default=1.0, metavar='SECONDS',
        help='Segundos entre requests (default: 1.0)',
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  FIXTURES HISTÓRICOS UCL/UEL — openfootball → PL")
    print("=" * 60)
    print(f"  Temporadas: {[temporada_str(y) for y in sorted(args.seasons)]}")
    print(f"  Destino:    {args.csv}")
    print(f"  Modo:       {'DRY-RUN' if args.dry_run else 'REAL'}\n")

    print("[1/3] Descargando y parseando archivos...")
    df_nuevo = descargar_todo(args.seasons, args.dry_run, args.sleep)

    if args.dry_run:
        print("\n[DRY RUN] Finalizado sin cambios.")
        return

    if df_nuevo.empty:
        print("\nNo se descargó ningún partido. Verifica la conexión.")
        sys.exit(1)

    print(f"\n  Partidos descargados: {len(df_nuevo)}")
    print(f"  Equipos: {sorted(df_nuevo['Team'].unique())}")
    print(f"  Por competición:")
    for comp, cnt in df_nuevo['Comp'].value_counts().items():
        print(f"    {comp}: {cnt}")

    print("\n[2/3] Mergeando con CSV existente...")
    df_final = merge_con_existente(df_nuevo, args.csv)

    print("\n[3/3] Guardando CSV...")
    print(f"\n  Resumen por temporada y competición:")
    resumen = df_final.groupby(['Season', 'Comp']).size().reset_index(name='n')
    for _, row in resumen.iterrows():
        print(f"    {row['Season']}  {row['Comp']:<15} {row['n']:>4} filas")

    print(f"\n  Total filas: {len(df_final)}")
    print(f"  Equipos:     {df_final['Team'].nunique()}")
    print(f"  Rango:       {df_final['Date'].min().date()} → "
          f"{df_final['Date'].max().date()}")

    # Backup antes de sobreescribir
    if os.path.exists(args.csv):
        backup = args.csv + '.backup_historicos'
        import shutil
        shutil.copy2(args.csv, backup)
        print(f"\n  Backup guardado en: {backup}")

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    df_final.to_csv(args.csv, index=False)
    print(f"  CSV guardado en:    {args.csv}")

    print("\nListo.")
    print("  Siguiente paso: ejecuta pipeline/01_preparar_datos.py "
          "para regenerar el dataset con las features de descanso actualizadas.")


if __name__ == '__main__':
    main()
