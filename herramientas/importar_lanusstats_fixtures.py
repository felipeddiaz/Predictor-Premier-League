# -*- coding: utf-8 -*-
"""
Importa fixtures desde un CSV exportado (por ejemplo LanusStats)
y lo convierte al formato historico del proyecto:

    Season,Date,Team,Comp,Venue,Opponent

Uso rapido:
    python herramientas/importar_lanusstats_fixtures.py \
      --input datos/raw/lanus/fixtures_2025_26.csv \
      --output datos/raw/historicos/champions_lg_2025-26.csv

Si el CSV trae varias competiciones, puedes usar --comp-col para respetarlas
o --comp para fijar un nombre unico.
"""

import argparse
import os
import pandas as pd


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


def _pick_column(df: pd.DataFrame, preferred: list[str]) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in preferred:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    raise ValueError(f"No se encontro columna entre: {preferred}")


def _season_from_date(d: pd.Timestamp) -> str:
    if pd.isna(d):
        return ""
    y = d.year if d.month >= 8 else d.year - 1
    return f"{y}-{str(y + 1)[-2:]}"


def main():
    parser = argparse.ArgumentParser(description="Convierte CSV de LanusStats a fixtures historicos")
    parser.add_argument("--input", required=True, help="CSV de entrada (LanusStats export)")
    parser.add_argument("--output", required=True, help="CSV de salida en datos/raw/historicos/")
    parser.add_argument("--comp", default="", help="Nombre de competicion fijo (ej: Champions Lg)")
    parser.add_argument("--date-col", default="", help="Nombre columna fecha")
    parser.add_argument("--home-col", default="", help="Nombre columna local")
    parser.add_argument("--away-col", default="", help="Nombre columna visitante")
    parser.add_argument("--comp-col", default="", help="Nombre columna competicion (si existe)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"No existe input: {args.input}")

    df = pd.read_csv(args.input)
    if df.empty:
        raise ValueError("El CSV de entrada esta vacio")

    date_col = args.date_col or _pick_column(df, ["Date", "Fecha", "match_date"])
    home_col = args.home_col or _pick_column(df, ["HomeTeam", "Local", "Home", "home_team"])
    away_col = args.away_col or _pick_column(df, ["AwayTeam", "Visitante", "Away", "away_team"])

    comp_col = None
    if args.comp_col:
        if args.comp_col not in df.columns:
            raise ValueError(f"comp-col no existe: {args.comp_col}")
        comp_col = args.comp_col
    elif not args.comp:
        for c in ["Comp", "Competition", "Torneo", "League"]:
            if c in df.columns:
                comp_col = c
                break

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce", format="mixed", dayfirst=True)
    work = work.dropna(subset=[date_col, home_col, away_col]).copy()

    if work.empty:
        raise ValueError("No hay filas validas tras parsear fecha/equipos")

    def norm_team(x: str) -> str:
        x = str(x).strip()
        return TEAM_MAP.get(x, x)

    work[home_col] = work[home_col].map(norm_team)
    work[away_col] = work[away_col].map(norm_team)

    if args.comp:
        comp_vals = pd.Series([args.comp] * len(work), index=work.index)
    elif comp_col:
        comp_vals = work[comp_col].astype(str).str.strip()
    else:
        comp_vals = pd.Series(["Unknown"] * len(work), index=work.index)

    home_rows = pd.DataFrame({
        "Season": work[date_col].map(_season_from_date),
        "Date": work[date_col].dt.strftime("%Y-%m-%d"),
        "Team": work[home_col].astype(str),
        "Comp": comp_vals,
        "Venue": "Home",
        "Opponent": work[away_col].astype(str),
    })

    away_rows = pd.DataFrame({
        "Season": work[date_col].map(_season_from_date),
        "Date": work[date_col].dt.strftime("%Y-%m-%d"),
        "Team": work[away_col].astype(str),
        "Comp": comp_vals,
        "Venue": "Away",
        "Opponent": work[home_col].astype(str),
    })

    out = pd.concat([home_rows, away_rows], ignore_index=True)
    out = out.drop_duplicates(subset=["Season", "Date", "Team", "Comp", "Venue", "Opponent"])
    out = out.sort_values(["Season", "Comp", "Date", "Team", "Venue"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Input filas: {len(df)}")
    print(f"Output filas: {len(out)}")
    print(f"Competiciones: {sorted(out['Comp'].dropna().unique())}")
    print(f"Rango fechas: {out['Date'].min()} -> {out['Date'].max()}")
    print(f"Guardado en: {args.output}")


if __name__ == "__main__":
    main()
