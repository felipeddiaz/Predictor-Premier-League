# -*- coding: utf-8 -*-
"""
descargar_fixtures_europeos.py
==============================
Descarga los fixtures de Champions League, Europa League, FA Cup y
Carabao Cup (EFL Cup) desde api-football.com y construye/actualiza el
archivo datos/raw/fbref_fixtures.csv, que luego usa utils.py para
calcular las features de días de descanso (HT_Days_Rest, AT_Days_Rest,
HT_Had_Europa, etc.).

Fuente: https://www.api-football.com  (plan gratuito: 100 req/día)

Uso:
    python herramientas/descargar_fixtures_europeos.py --api-key TU_KEY

    Flags opcionales:
      --api-key KEY       API key de api-football.com (o variable de entorno
                          API_FOOTBALL_KEY)
      --seasons 2020 2021 Temporadas a descargar (default: 2016 a 2025)
      --dry-run           Muestra cuántas requests haría sin ejecutarlas
      --csv RUTA          Ruta alternativa al CSV de salida

Plan de requests (free tier 100/día):
    4 competiciones × 10 temporadas = 40 requests en total.
    Con --sleep 2 (2s entre requests) tarda ~1.5 minutos.

Competiciones descargadas:
    UCL  — Champions League    (id: 2)
    UEL  — Europa League       (id: 3)
    FAC  — FA Cup              (id: 45)
    ELC  — Carabao Cup         (id: 48)

Estructura del CSV resultante (compatible con utils.agregar_features_descanso):
    Season, Date, Team, Comp, Venue, Opponent
    2024-25, 2024-10-01, Arsenal, Champions Lg, Home, PSG
    2024-25, 2024-10-05, Arsenal, Premier League, Away, Man City

Requisitos:
    pip install requests pandas
"""

import argparse
import os
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
# Configuración de competiciones
# ---------------------------------------------------------------------------
# ID de api-football.com → nombre canónico en el CSV
COMPETICIONES = {
    2:  'Champions Lg',   # UEFA Champions League
    3:  'Europa Lg',      # UEFA Europa League
    45: 'FA Cup',         # FA Cup
    48: 'EFL Cup',        # Carabao Cup / League Cup
}

# Temporadas disponibles en el dataset PL (año de inicio de temporada)
TEMPORADAS_DEFAULT = list(range(2016, 2026))  # 2016-17 a 2025-26

# ---------------------------------------------------------------------------
# Normalización de nombres de equipo al formato canónico del proyecto
# (mismo sistema que utils._XG_TEAM_MAP)
# ---------------------------------------------------------------------------
_TEAM_MAP = {
    # api-football → nombre canónico del proyecto
    'Manchester City':          'Man City',
    'Manchester United':        'Man United',
    'Newcastle United':         'Newcastle',
    'Tottenham Hotspur':        'Tottenham',
    'West Ham United':          'West Ham',
    'Wolverhampton Wanderers':  'Wolves',
    'Brighton & Hove Albion':   'Brighton',
    'Nottingham Forest':        "Nott'm Forest",
    'Sheffield Utd':            'Sheffield United',
    'Sheffield United':         'Sheffield United',
    'Leicester City':           'Leicester',
    'Leeds United':             'Leeds',
    'Luton Town':               'Luton',
    'Norwich City':             'Norwich',
    'West Bromwich Albion':     'West Brom',
    'Ipswich Town':             'Ipswich',
    'Huddersfield Town':        'Huddersfield',
    'Cardiff City':             'Cardiff',
    'Middlesbrough':            'Middlesbrough',
    'Stoke City':               'Stoke',
    'Swansea City':             'Swansea',
    'Hull City':                'Hull',
    'Sunderland AFC':           'Sunderland',
    'Burnley FC':               'Burnley',
    'Bournemouth':              'Bournemouth',
    'AFC Bournemouth':          'Bournemouth',
    'Brentford FC':             'Brentford',
    'Fulham FC':                'Fulham',
    'Crystal Palace':           'Crystal Palace',
    'Everton FC':               'Everton',
    'Watford FC':               'Watford',
    'Aston Villa':              'Aston Villa',
    'Arsenal FC':               'Arsenal',
    'Chelsea FC':               'Chelsea',
    'Liverpool FC':             'Liverpool',
    'Southampton FC':           'Southampton',
}

# Equipos de PL que nos interesan (solo guardamos partidos de estos equipos)
_EQUIPOS_PL = {
    'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
    'Burnley', 'Cardiff', 'Chelsea', 'Crystal Palace', 'Everton',
    'Fulham', 'Huddersfield', 'Hull', 'Ipswich', 'Leeds', 'Leicester',
    'Liverpool', 'Luton', 'Man City', 'Man United', 'Middlesbrough',
    'Newcastle', "Nott'm Forest", 'Sheffield United', 'Southampton',
    'Stoke', 'Sunderland', 'Swansea', 'Tottenham', 'Watford',
    'West Brom', 'West Ham', 'Wolves',
}


def normalizar_equipo(nombre: str) -> str:
    """Normaliza el nombre de un equipo al formato canónico del proyecto."""
    return _TEAM_MAP.get(nombre, nombre)


def temporada_str(year: int) -> str:
    """Convierte año de inicio (2024) → string de temporada '2024-25'."""
    return f"{year}-{str(year + 1)[-2:]}"


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
BASE_URL = 'https://v3.football.api-sports.io'


def get_fixtures(api_key: str, league_id: int, season: int,
                 sleep_s: float = 2.0) -> list[dict]:
    """
    Descarga todos los fixtures de una liga y temporada.

    Args:
        api_key:   API key de api-football.com
        league_id: ID de la competición
        season:    Año de inicio de la temporada (ej: 2024 para 2024-25)
        sleep_s:   Segundos de espera antes de la request (rate limiting)

    Returns:
        Lista de dicts con los fixtures de la respuesta JSON.
    """
    time.sleep(sleep_s)
    headers = {
        'x-rapidapi-key': api_key,
        'x-rapidapi-host': 'v3.football.api-sports.io',
    }
    params = {'league': league_id, 'season': season}
    url = f'{BASE_URL}/fixtures'

    resp = requests.get(url, headers=headers, params=params, timeout=20)
    resp.raise_for_status()

    data = resp.json()
    errores = data.get('errors', {})
    if errores:
        raise ValueError(f'API error: {errores}')

    fixtures = data.get('response', [])
    return fixtures


def fixture_a_filas(fixtures: list[dict], comp_nombre: str,
                    season: int) -> list[dict]:
    """
    Convierte la lista de fixtures de la API en filas para el CSV.

    Cada partido genera DOS filas: una por equipo local, otra por visitante.
    Solo se guardan filas de equipos que estén en _EQUIPOS_PL.

    Args:
        fixtures:    Lista de dicts devuelta por get_fixtures()
        comp_nombre: Nombre canónico de la competición ('Champions Lg', etc.)
        season:      Año de inicio de la temporada

    Returns:
        Lista de dicts listos para DataFrame.
    """
    filas = []
    for f in fixtures:
        # Fecha del partido (puede ser null si no está programado)
        fixture_info = f.get('fixture', {})
        fecha_str = fixture_info.get('date', '')
        if not fecha_str:
            continue
        # Convertir a date (solo la fecha, sin hora)
        try:
            fecha = pd.to_datetime(fecha_str).date()
        except Exception:
            continue

        home_raw = f.get('teams', {}).get('home', {}).get('name', '')
        away_raw = f.get('teams', {}).get('away', {}).get('name', '')
        home = normalizar_equipo(home_raw)
        away = normalizar_equipo(away_raw)
        temp_str = temporada_str(season)

        for team, venue, opponent in [(home, 'Home', away),
                                      (away, 'Away', home)]:
            if team in _EQUIPOS_PL:
                filas.append({
                    'Season': temp_str,
                    'Date':   str(fecha),
                    'Team':   team,
                    'Comp':   comp_nombre,
                    'Venue':  venue,
                    'Opponent': opponent,
                })

    return filas


# ---------------------------------------------------------------------------
# Lógica principal
# ---------------------------------------------------------------------------

def descargar_todo(api_key: str, temporadas: list[int],
                   dry_run: bool = False, sleep_s: float = 2.0,
                   csv_destino: str = _CSV_DESTINO) -> pd.DataFrame:
    """
    Descarga fixtures de todas las competiciones y temporadas configuradas.

    Args:
        api_key:      API key de api-football.com
        temporadas:   Lista de años de inicio de temporada a descargar
        dry_run:      Si True, no hace ninguna request real
        sleep_s:      Segundos entre requests
        csv_destino:  Ruta del CSV de salida

    Returns:
        DataFrame con todos los fixtures descargados.
    """
    total_requests = len(COMPETICIONES) * len(temporadas)
    print(f"\nPlan de descarga:")
    print(f"  Competiciones : {list(COMPETICIONES.values())}")
    print(f"  Temporadas    : {[temporada_str(t) for t in temporadas]}")
    print(f"  Total requests: {total_requests} (free tier: 100/día)")

    if dry_run:
        print("\n[DRY RUN] Sin requests reales. Saliendo.")
        return pd.DataFrame()

    todas_las_filas = []
    n = 0

    for comp_id, comp_nombre in COMPETICIONES.items():
        for season in temporadas:
            n += 1
            temp_str = temporada_str(season)
            print(f"  [{n:2d}/{total_requests}] {comp_nombre} {temp_str} ...",
                  end='', flush=True)

            try:
                fixtures = get_fixtures(api_key, comp_id, season, sleep_s)
                filas = fixture_a_filas(fixtures, comp_nombre, season)
                todas_las_filas.extend(filas)
                print(f" {len(fixtures)} partidos, {len(filas)} filas PL")

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    print(f" RATE LIMIT — esperar y reintentar")
                    time.sleep(60)
                else:
                    print(f" ERROR HTTP {e.response.status_code}: {e}")
            except Exception as e:
                print(f" ERROR: {e}")

    if not todas_las_filas:
        print("\nNo se descargó ningún fixture.")
        return pd.DataFrame()

    df_nuevo = pd.DataFrame(todas_las_filas)
    df_nuevo['Date'] = pd.to_datetime(df_nuevo['Date'])
    df_nuevo = df_nuevo.sort_values(['Team', 'Date']).reset_index(drop=True)

    # Merge con CSV existente (si existe) eliminando duplicados
    if os.path.exists(csv_destino):
        print(f"\nMergeando con CSV existente: {csv_destino}")
        df_existente = pd.read_csv(csv_destino, parse_dates=['Date'])
        df_combined = pd.concat([df_existente, df_nuevo], ignore_index=True)
        df_combined = df_combined.drop_duplicates(
            subset=['Season', 'Date', 'Team', 'Comp']
        ).sort_values(['Team', 'Date']).reset_index(drop=True)
    else:
        df_combined = df_nuevo

    # Guardar
    os.makedirs(os.path.dirname(csv_destino), exist_ok=True)
    df_combined.to_csv(csv_destino, index=False)
    print(f"\nGuardado: {csv_destino}")
    print(f"  Total filas: {len(df_combined)}")
    print(f"  Equipos:     {df_combined['Team'].nunique()}")
    print(f"  Rango:       {df_combined['Date'].min().date()} → "
          f"{df_combined['Date'].max().date()}")

    return df_combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Descarga fixtures europeos y de copas desde api-football.com'
    )
    parser.add_argument(
        '--api-key',
        default=os.environ.get('API_FOOTBALL_KEY', ''),
        help='API key de api-football.com (o variable de entorno API_FOOTBALL_KEY)',
    )
    parser.add_argument(
        '--seasons',
        nargs='+',
        type=int,
        default=TEMPORADAS_DEFAULT,
        metavar='YEAR',
        help='Años de inicio de temporada a descargar (default: 2016 a 2025)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Muestra el plan sin hacer requests reales',
    )
    parser.add_argument(
        '--csv',
        default=_CSV_DESTINO,
        help=f'Ruta del CSV de salida (default: {_CSV_DESTINO})',
    )
    parser.add_argument(
        '--sleep',
        type=float,
        default=2.0,
        metavar='SECONDS',
        help='Segundos de espera entre requests (default: 2.0)',
    )
    args = parser.parse_args()

    if not args.api_key and not args.dry_run:
        print("ERROR: Se requiere --api-key o la variable de entorno API_FOOTBALL_KEY")
        print("  Regístrate gratis en https://dashboard.api-football.com/register")
        print("  El plan gratuito incluye 100 requests/día y todas las competiciones.")
        sys.exit(1)

    descargar_todo(
        api_key=args.api_key,
        temporadas=sorted(args.seasons),
        dry_run=args.dry_run,
        sleep_s=args.sleep,
        csv_destino=args.csv,
    )


if __name__ == '__main__':
    main()
