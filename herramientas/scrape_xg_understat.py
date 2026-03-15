# -*- coding: utf-8 -*-
"""
scrape_xg_understat.py
======================
Descarga los datos de xG (Expected Goals) de la temporada 2025/26 de la
Premier League desde Understat.com y actualiza el archivo
datos/raw/final_matches_xg.csv con las nuevas filas.

Fuente: https://understat.com/league/EPL/2025
  - Endpoint JSON: https://understat.com/getLeagueData/EPL/{year}
    donde year=2025 corresponde a la temporada 2025/26.

Estructura del CSV resultante (compatible con utils.py → merge_xg_data):
  date, time, comp, round, day, venue, result, gf, ga, opponent,
  xg, xga, poss, attendance, captain, formation, opp formation,
  referee, match report, notes, sh, sot, dist, fk, pk, pkatt,
  team, season, opponent.1

Uso:
    python herramientas/scrape_xg_understat.py

    Flags opcionales:
      --season YYYY   Temporada a descargar (default: 2025, es decir 2025/26)
      --dry-run       Muestra resumen sin modificar el CSV
      --csv RUTA      Ruta alternativa al CSV de salida

Requisitos (ya disponibles en el venv del proyecto):
    pip install requests
"""

import argparse
import os
import sys
import time
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Rutas por defecto (relativas a este archivo → raíz del proyecto)
# ---------------------------------------------------------------------------
_DIR_SCRIPT  = os.path.dirname(os.path.abspath(__file__))
_RAIZ        = os.path.dirname(_DIR_SCRIPT)
_CSV_DESTINO = os.path.join(_RAIZ, 'datos', 'raw', 'final_matches_xg.csv')

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
UNDERSTAT_URL = 'https://understat.com/getLeagueData/EPL/{year}'

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/131.0.0.0 Safari/537.36'
    ),
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'X-Requested-With': 'XMLHttpRequest',
    'Referer': 'https://understat.com/league/EPL/',
}

# Temporada label para el CSV (mismo convenio que los datos históricos:
# 2017 = temporada 2016/17, ..., 2026 = temporada 2025/26)
SEASON_LABEL_MAP = {
    2016: 2017,
    2017: 2018,
    2018: 2019,
    2019: 2020,
    2020: 2021,
    2021: 2022,
    2022: 2023,
    2023: 2024,
    2024: 2025,
    2025: 2026,
}

# Mapeo de nombres de Understat → nombre canónico del dataset (coincide con
# el _XG_TEAM_MAP de utils.py, más los nombres específicos de Understat)
TEAM_NAME_MAP = {
    # Understat usa nombres completos en su mayoría
    'Manchester City':        'Man City',
    'Manchester United':      'Man United',
    'Newcastle United':       'Newcastle',
    'Nottingham Forest':      "Nott'm Forest",
    'Wolverhampton Wanderers':'Wolves',
    'Tottenham':              'Tottenham',
    'Brighton':               'Brighton',
    'Leeds':                  'Leeds',
    'Leicester':              'Leicester',
    'Luton':                  'Luton',
    'West Ham':               'West Ham',
    'Norwich':                'Norwich',
    # Understat 25/26 specific
    'Wolverhampton Wanderers':'Wolves',
    'Bournemouth':            'Bournemouth',
    'Brentford':              'Brentford',
    'Burnley':                'Burnley',
    'Sunderland':             'Sunderland',
}

# Columnas canónicas del CSV (deben coincidir con el CSV histórico)
COLUMNAS_CSV = [
    'date', 'time', 'comp', 'round', 'day', 'venue', 'result',
    'gf', 'ga', 'opponent', 'xg', 'xga', 'poss', 'attendance',
    'captain', 'formation', 'opp formation', 'referee',
    'match report', 'notes', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt',
    'team', 'season', 'opponent.1',
]

# Días de la semana en inglés (para la columna 'day')
_DIAS = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}

# Resultado inverso: si el local ganó (W) el visitante perdió (L) y viceversa
_RESULT_INV = {'W': 'L', 'L': 'W', 'D': 'D'}


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------

def normalizar_equipo(nombre: str) -> str:
    """Devuelve el nombre canónico del equipo."""
    return TEAM_NAME_MAP.get(nombre, nombre)


def descargar_partidos(year: int) -> list[dict]:
    """
    Llama al endpoint de Understat y devuelve la lista de partidos jugados
    con datos de xG para la temporada indicada.

    Args:
        year: Año de inicio de la temporada (ej. 2025 para 2025/26).

    Returns:
        Lista de dicts con los campos del partido (solo isResult=True y xG no nulos).
    """
    url = UNDERSTAT_URL.format(year=year)
    print(f"   Descargando: {url}")

    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except requests.exceptions.Timeout:
        print("   ERROR: Timeout al conectar con Understat (30 s)")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"   ERROR HTTP: {e}")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("   ERROR: Sin conexión a Internet o DNS no resuelto")
        sys.exit(1)

    try:
        data = r.json()
    except ValueError:
        print("   ERROR: La respuesta no es JSON válido")
        print(f"   Respuesta (primeros 300 chars): {r.text[:300]}")
        sys.exit(1)

    partidos_raw = data.get('dates', [])
    jugados = [
        m for m in partidos_raw
        if m.get('isResult') is True
        and m.get('xG', {}).get('h') is not None
        and m.get('xG', {}).get('a') is not None
    ]

    print(f"   Partidos en la temporada: {len(partidos_raw)}")
    print(f"   Partidos jugados con xG:  {len(jugados)}")
    return jugados


def convertir_a_filas(partidos: list[dict], season_label: int) -> pd.DataFrame:
    """
    Transforma la lista de partidos de Understat al formato del CSV histórico.
    Genera DOS filas por partido (una para el equipo local, otra para el visitante),
    exactamente igual que el formato FBRef original.

    Columnas clave para utils.merge_xg_data:
      date, venue (Home/Away), xg, xga, team, opponent, season

    Args:
        partidos:     Lista de dicts de Understat (solo jugados con xG).
        season_label: Entero que identifica la temporada (ej. 2026 para 2025/26).

    Returns:
        DataFrame con las filas en formato canónico.
    """
    filas = []

    for m in partidos:
        # ---- Extraer campos básicos ----
        home_name = normalizar_equipo(m['h']['title'])
        away_name = normalizar_equipo(m['a']['title'])
        home_xg   = float(m['xG']['h'])
        away_xg   = float(m['xG']['a'])
        home_gf   = int(m['goals']['h']) if m['goals']['h'] is not None else ''
        away_gf   = int(m['goals']['a']) if m['goals']['a'] is not None else ''

        # ---- Fecha y hora ----
        dt = pd.to_datetime(m['datetime'])
        fecha_str = dt.strftime('%d/%m/%Y')
        hora_str  = dt.strftime('%H:%M')
        dia_str   = _DIAS[dt.weekday()]

        # ---- Resultado desde perspectiva de cada equipo ----
        # Understat no incluye el resultado directamente en getLeagueData
        # (la clave 'result' no siempre está presente), lo deducimos de los goles
        if home_gf != '' and away_gf != '':
            if home_gf > away_gf:
                res_local    = 'W'
                res_visitante = 'L'
            elif home_gf < away_gf:
                res_local    = 'L'
                res_visitante = 'W'
            else:
                res_local    = 'D'
                res_visitante = 'D'
        else:
            res_local = res_visitante = ''

        # ---- Fila del equipo LOCAL ----
        filas.append({
            'date':          fecha_str,
            'time':          hora_str,
            'comp':          'Premier League',
            'round':         '',
            'day':           dia_str,
            'venue':         'Home',
            'result':        res_local,
            'gf':            home_gf,
            'ga':            away_gf,
            'opponent':      away_name,
            'xg':            home_xg,
            'xga':           away_xg,
            'poss':          '',
            'attendance':    '',
            'captain':       '',
            'formation':     '',
            'opp formation': '',
            'referee':       '',
            'match report':  '',
            'notes':         '',
            'sh':            '',
            'sot':           '',
            'dist':          '',
            'fk':            '',
            'pk':            '',
            'pkatt':         '',
            'team':          home_name,
            'season':        season_label,
            'opponent.1':    away_name,
        })

        # ---- Fila del equipo VISITANTE ----
        filas.append({
            'date':          fecha_str,
            'time':          hora_str,
            'comp':          'Premier League',
            'round':         '',
            'day':           dia_str,
            'venue':         'Away',
            'result':        res_visitante,
            'gf':            away_gf,
            'ga':            home_gf,
            'opponent':      home_name,
            'xg':            away_xg,
            'xga':           home_xg,
            'poss':          '',
            'attendance':    '',
            'captain':       '',
            'formation':     '',
            'opp formation': '',
            'referee':       '',
            'match report':  '',
            'notes':         '',
            'sh':            '',
            'sot':           '',
            'dist':          '',
            'fk':            '',
            'pk':            '',
            'pkatt':         '',
            'team':          away_name,
            'season':        season_label,
            'opponent.1':    home_name,
        })

    df = pd.DataFrame(filas, columns=COLUMNAS_CSV)
    return df


def actualizar_csv(df_nuevo: pd.DataFrame, ruta_csv: str, season_label: int,
                   dry_run: bool = False) -> None:
    """
    Carga el CSV histórico existente, elimina cualquier fila previa de la
    temporada indicada y concatena las filas nuevas. Guarda el resultado.

    Args:
        df_nuevo:     DataFrame con las nuevas filas de xG.
        ruta_csv:     Ruta al archivo CSV de destino.
        season_label: Etiqueta de temporada a reemplazar (ej. 2026).
        dry_run:      Si True, solo imprime el resumen sin guardar.
    """
    if os.path.exists(ruta_csv):
        df_existente = pd.read_csv(ruta_csv)
        filas_antes = len(df_existente)
        # Eliminar filas de la temporada que se va a reemplazar
        df_existente = df_existente[df_existente['season'] != season_label]
        filas_eliminadas = filas_antes - len(df_existente)
        if filas_eliminadas:
            print(f"   Eliminadas {filas_eliminadas} filas previas de season={season_label}")
        df_final = pd.concat([df_existente, df_nuevo], ignore_index=True)
    else:
        print(f"   CSV no encontrado — se creará nuevo en: {ruta_csv}")
        df_final = df_nuevo

    print(f"\n   Resumen del CSV actualizado:")
    print(f"   {'Temporada':<12} {'Filas':>8}")
    print(f"   {'-'*22}")
    for season, cnt in df_final.groupby('season').size().items():
        marca = " <-- NUEVO" if season == season_label else ""
        print(f"   {season:<12} {cnt:>8}{marca}")
    print(f"   {'-'*22}")
    print(f"   {'TOTAL':<12} {len(df_final):>8}")

    if dry_run:
        print("\n   [DRY-RUN] No se guardaron cambios.")
        return

    os.makedirs(os.path.dirname(ruta_csv), exist_ok=True)
    # Crear backup antes de sobreescribir
    backup = ruta_csv + '.backup_scrape'
    if os.path.exists(ruta_csv):
        import shutil
        shutil.copy2(ruta_csv, backup)
        print(f"\n   Backup guardado en: {backup}")

    df_final.to_csv(ruta_csv, index=False)
    print(f"   CSV guardado en:    {ruta_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Descarga xG de Understat.com y actualiza datos/raw/final_matches_xg.csv'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python herramientas/scrape_xg_understat.py
  python herramientas/scrape_xg_understat.py --season 2024
  python herramientas/scrape_xg_understat.py --dry-run
  python herramientas/scrape_xg_understat.py --season 2025 --csv mis_datos/xg.csv
        """
    )
    parser.add_argument(
        '--season', type=int, default=2025,
        help='Año de inicio de la temporada (default: 2025 → temporada 2025/26)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Muestra resumen sin modificar el CSV'
    )
    parser.add_argument(
        '--csv', type=str, default=_CSV_DESTINO,
        help='Ruta al CSV de destino (default: datos/raw/final_matches_xg.csv)'
    )
    args = parser.parse_args()

    season_label = SEASON_LABEL_MAP.get(args.season)
    if season_label is None:
        # Si el año no está en el mapa, usar year+1 como convención
        season_label = args.season + 1
        print(f"   Aviso: year={args.season} no está en SEASON_LABEL_MAP, "
              f"usando season_label={season_label}")

    print("=" * 60)
    print("  SCRAPER xG -- Understat.com -> Premier League")
    print("=" * 60)
    print(f"  Temporada:   {args.season}/{str(args.season + 1)[-2:]} "
          f"(season label en CSV: {season_label})")
    print(f"  Destino CSV: {args.csv}")
    print(f"  Modo:        {'DRY-RUN (sin cambios)' if args.dry_run else 'REAL (actualiza CSV)'}")
    print()

    print("[1/3] Descargando partidos de Understat...")
    partidos = descargar_partidos(args.season)

    print("\n[2/3] Transformando al formato del CSV historico...")
    df_nuevo = convertir_a_filas(partidos, season_label)
    print(f"   Filas generadas: {len(df_nuevo)}  "
          f"({len(partidos)} partidos x 2 equipos)")

    # Verificación rápida
    equipos_unicos = sorted(df_nuevo['team'].unique())
    print(f"   Equipos detectados ({len(equipos_unicos)}): "
          f"{', '.join(equipos_unicos)}")

    print("\n[3/3] Actualizando CSV...")
    actualizar_csv(df_nuevo, args.csv, season_label, dry_run=args.dry_run)

    print("\nListo.")
    if not args.dry_run:
        print("  Siguiente paso: ejecuta pipeline/01_preparar_datos.py "
              "para regenerar el dataset con los nuevos xG.")


if __name__ == '__main__':
    main()
