# -*- coding: utf-8 -*-
"""
Analizar qué partidos faltan de xG
"""

import pandas as pd

# Cargar datos integrados
df = pd.read_csv('./datos/procesados/premier_league_con_xg_integrado.csv')

# Filtrar partidos SIN xG
sin_xg = df[df['Home_xG'].isna()].copy()

print("="*70)
print(f"PARTIDOS SIN xG: {len(sin_xg)}/{len(df)}")
print("="*70)

# Convertir fechas (formato YYYY-MM-DD o DD/MM/YYYY automático)
df['Date_parsed'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')
sin_xg['Date_parsed'] = pd.to_datetime(sin_xg['Date'], format='mixed', dayfirst=True, errors='coerce')

# Análisis por año
df['Year'] = df['Date_parsed'].dt.year
sin_xg['Year'] = sin_xg['Date_parsed'].dt.year

print("\n📊 DISTRIBUCIÓN POR AÑO:")
print("-"*50)
for year in sorted(sin_xg['Year'].dropna().unique()):
    total_year = len(df[df['Year'] == year])
    sin_year = len(sin_xg[sin_xg['Year'] == year])
    pct = sin_year / total_year * 100 if total_year > 0 else 0
    print(f"   {int(year)}: {sin_year}/{total_year} sin xG ({pct:.1f}%)")

# Primeros y últimos
print("\n📅 RANGO DE PARTIDOS SIN xG:")
print("-"*50)
print(f"   Primer partido sin xG: {sin_xg['Date'].min()}")
print(f"   Último partido sin xG: {sin_xg['Date'].max()}")

# Ver primeros 20 partidos sin xG
print("\n📋 PRIMEROS 20 PARTIDOS SIN xG:")
print("-"*50)
cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
print(sin_xg[cols].head(20).to_string(index=False))

# Ver últimos 20 partidos sin xG
print("\n📋 ÚLTIMOS 20 PARTIDOS SIN xG:")
print("-"*50)
print(sin_xg[cols].tail(20).to_string(index=False))

# Análisis por temporada (Sep-Mayo)
sin_xg['Month'] = sin_xg['Date_parsed'].dt.month
sin_xg['Temporada'] = sin_xg['Year'].apply(
    lambda y: f"{int(y)-1}/{int(y)}" if y == y else None
)
sin_xg.loc[sin_xg['Month'] >= 8, 'Temporada'] = sin_xg.loc[sin_xg['Month'] >= 8, 'Year'].apply(
    lambda y: f"{int(y)}/{int(y)+1}" if y == y else None
)

print("\n📊 DISTRIBUCIÓN POR TEMPORADA:")
print("-"*50)
for temp in sorted(sin_xg['Temporada'].dropna().unique()):
    count = len(sin_xg[sin_xg['Temporada'] == temp])
    print(f"   {temp}: {count} partidos sin xG")

# Exportar lista completa
sin_xg_export = sin_xg[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].copy()
sin_xg_export.to_csv('./datos/procesados/partidos_sin_xg.csv', index=False)

print(f"\n💾 Lista completa guardada en: ./datos/procesados/partidos_sin_xg.csv")

# Resumen
print("\n" + "="*70)
print("RESUMEN:")
print("="*70)
print(f"✅ Total partidos: {len(df)}")
print(f"✅ Con xG: {df['Home_xG'].notna().sum()} ({df['Home_xG'].notna().sum()/len(df)*100:.1f}%)")
print(f"⚠️  Sin xG: {len(sin_xg)} ({len(sin_xg)/len(df)*100:.1f}%)")

# Ver rango de fechas con xG
con_xg = df[df['Home_xG'].notna()].copy()
con_xg['Date_parsed'] = pd.to_datetime(con_xg['Date'], format='mixed', dayfirst=True, errors='coerce')

print(f"\n📊 RANGO DE FECHAS CON xG:")
print(f"   Desde: {con_xg['Date_parsed'].min()}")
print(f"   Hasta: {con_xg['Date_parsed'].max()}")

print(f"\n📊 RANGO DE FECHAS SIN xG:")
print(f"   Desde: {sin_xg['Date_parsed'].min()}")
print(f"   Hasta: {sin_xg['Date_parsed'].max()}")

# Ver equipos más afectados
print(f"\n📊 EQUIPOS MÁS AFECTADOS (sin xG):")
print("-"*50)
equipos_sin = pd.concat([sin_xg['HomeTeam'], sin_xg['AwayTeam']]).value_counts().head(10)
for equipo, count in equipos_sin.items():
    print(f"   {equipo}: {count} partidos sin xG")