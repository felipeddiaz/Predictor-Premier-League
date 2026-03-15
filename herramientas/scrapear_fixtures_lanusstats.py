# -*- coding: utf-8 -*-
"""
Scrapea fixtures de copas/europa usando el stack de LanusStats + SofaScore.

Motivacion:
- API-Football free no permite temporada 2025-26 en este entorno.
- Este scraper usa endpoints publicos de SofaScore mediante navegador headless
  (undetected-chromedriver) para evitar el bloqueo 403.

Salida:
- Actualiza `datos/raw/fbref_fixtures.csv` en el formato canonico:
  Season, Date, Team, Comp, Venue, Opponent

Uso recomendado:
    python herramientas/scrapear_fixtures_lanusstats.py --auto

Opcional:
    python herramientas/scrapear_fixtures_lanusstats.py --start-date 2025-03-17 --end-date 2026-03-15
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime, timedelta

import pandas as pd
import undetected_chromedriver as uc
from bs4 import BeautifulSoup


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FIXTURES = os.path.join(ROOT, "datos", "raw", "fbref_fixtures.csv")

TARGET_COMP_MAP = {
    "UEFA Champions League": "Champions Lg",
    "UEFA Europa League": "Europa Lg",
    "FA Cup": "FA Cup",
    "EFL Cup": "EFL Cup",
}

TEAM_MAP = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton & Hove Albion": "Brighton",
    "Nottingham Forest": "Nott'm Forest",
    "Sheffield Utd": "Sheffield United",
    "Sheffield United": "Sheffield United",
    "Leicester City": "Leicester",
    "Leeds United": "Leeds",
    "Luton Town": "Luton",
    "Norwich City": "Norwich",
    "West Bromwich Albion": "West Brom",
    "Ipswich Town": "Ipswich",
    "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Fulham FC": "Fulham",
    "Chelsea FC": "Chelsea",
    "Liverpool FC": "Liverpool",
    "Arsenal FC": "Arsenal",
    "Everton FC": "Everton",
    "Burnley FC": "Burnley",
}

EQUIPOS_PL = {
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Burnley", "Cardiff", "Chelsea", "Crystal Palace", "Everton",
    "Fulham", "Huddersfield", "Hull", "Ipswich", "Leeds", "Leicester",
    "Liverpool", "Luton", "Man City", "Man United", "Middlesbrough",
    "Newcastle", "Nott'm Forest", "Sheffield United", "Southampton",
    "Stoke", "Sunderland", "Swansea", "Tottenham", "Watford",
    "West Brom", "West Ham", "Wolves",
}


def norm_team(name: str) -> str:
    return TEAM_MAP.get(name, name)


def season_from_date(d: date) -> str:
    y = d.year if d.month >= 8 else d.year - 1
    return f"{y}-{str(y + 1)[-2:]}"


def get_driver(version_main: int = 145):
    opts = uc.ChromeOptions()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    return uc.Chrome(options=opts, version_main=version_main)


def fetch_scheduled_events(driver, day: date) -> list[dict]:
    url = f"https://www.sofascore.com/api/v1/sport/football/scheduled-events/{day.isoformat()}"
    driver.get(url)
    time.sleep(1.8)
    txt = BeautifulSoup(driver.page_source, "html.parser").text
    data = json.loads(txt)
    return data.get("events", [])


def evento_a_filas(evento: dict) -> list[dict]:
    ut_name = (
        evento.get("tournament", {})
        .get("uniqueTournament", {})
        .get("name", "")
    )
    comp = TARGET_COMP_MAP.get(ut_name)
    if not comp:
        return []

    ts = evento.get("startTimestamp")
    if not ts:
        return []

    d = datetime.utcfromtimestamp(ts).date()
    season = season_from_date(d)

    home_raw = evento.get("homeTeam", {}).get("name", "")
    away_raw = evento.get("awayTeam", {}).get("name", "")
    home = norm_team(home_raw)
    away = norm_team(away_raw)

    filas = []
    if home in EQUIPOS_PL:
        filas.append({
            "Season": season,
            "Date": d.isoformat(),
            "Team": home,
            "Comp": comp,
            "Venue": "Home",
            "Opponent": away,
        })
    if away in EQUIPOS_PL:
        filas.append({
            "Season": season,
            "Date": d.isoformat(),
            "Team": away,
            "Comp": comp,
            "Venue": "Away",
            "Opponent": home,
        })
    return filas


def infer_auto_start() -> date:
    if not os.path.exists(CSV_FIXTURES):
        return date(2025, 1, 1)
    df = pd.read_csv(CSV_FIXTURES)
    if df.empty:
        return date(2025, 1, 1)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format="mixed", dayfirst=True)
    latest = (
        df[df["Comp"].isin(["Champions Lg", "Europa Lg", "FA Cup", "EFL Cup"])]
        ["Date"].max()
    )
    if pd.isna(latest):
        return date(2025, 1, 1)
    return (latest.date() + timedelta(days=1))


def main():
    parser = argparse.ArgumentParser(description="Scraper de fixtures con stack LanusStats/SofaScore")
    parser.add_argument("--auto", action="store_true", help="Inicia desde el dia siguiente al ultimo fixture guardado")
    parser.add_argument("--start-date", default="", help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=date.today().isoformat(), help="YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="No escribe CSV")
    parser.add_argument("--chrome-major", type=int, default=145, help="Major version de Chrome para uc")
    args = parser.parse_args()

    if args.auto:
        start = infer_auto_start()
    elif args.start_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    else:
        raise ValueError("Usa --auto o --start-date")

    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    if end < start:
        raise ValueError("end-date no puede ser menor a start-date")

    print(f"Rango scraping: {start} -> {end}")

    driver = get_driver(version_main=args.chrome_major)
    all_rows = []
    try:
        d = start
        total_days = (end - start).days + 1
        i = 0
        while d <= end:
            i += 1
            try:
                events = fetch_scheduled_events(driver, d)
                day_rows = []
                for e in events:
                    day_rows.extend(evento_a_filas(e))
                all_rows.extend(day_rows)
                if i % 7 == 0 or day_rows:
                    print(f"[{i:03d}/{total_days}] {d}: events={len(events)} rows={len(day_rows)}")
            except Exception as ex:
                print(f"[{i:03d}/{total_days}] {d}: ERROR {ex}")
            d += timedelta(days=1)
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    if not all_rows:
        print("No se encontraron nuevas filas de equipos PL en copas/europa.")
        return

    new_df = pd.DataFrame(all_rows)
    new_df["Date"] = pd.to_datetime(new_df["Date"], errors="coerce")
    new_df = new_df.dropna(subset=["Date"]).copy()

    if os.path.exists(CSV_FIXTURES):
        base = pd.read_csv(CSV_FIXTURES)
        base["Date"] = pd.to_datetime(base["Date"], errors="coerce", format="mixed", dayfirst=True)
        combined = pd.concat([base, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.drop_duplicates(subset=["Season", "Date", "Team", "Comp", "Venue", "Opponent"])
    combined = combined.sort_values(["Team", "Date"]).reset_index(drop=True)

    print("\nResumen nuevas filas:")
    print(new_df.groupby(["Season", "Comp"]).size().to_string())
    print(f"\nNuevas filas (sin dedupe): {len(new_df)}")
    print(f"Total combinado: {len(combined)}")
    print(f"Max fecha final: {combined['Date'].max().date()}")

    if args.dry_run:
        print("\n[DRY RUN] No se escribio CSV")
        return

    backup = CSV_FIXTURES + ".backup_lanusstats"
    if os.path.exists(CSV_FIXTURES):
        import shutil
        shutil.copy2(CSV_FIXTURES, backup)
        print(f"Backup: {backup}")

    combined.to_csv(CSV_FIXTURES, index=False)
    print(f"Escrito: {CSV_FIXTURES}")


if __name__ == "__main__":
    main()
