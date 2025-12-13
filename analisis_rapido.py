# -*- coding: utf-8 -*-
"""
Análisis Rápido del CSV - Pégalo aquí y me dices qué ves
"""

import pandas as pd
import numpy as np

# Carga tu CSV
df = pd.read_csv('./datos/procesados/premier_league_con_features.csv')

print("="*70)
print("📊 ANÁLISIS RÁPIDO")
print("="*70)

# 1. Información básica
print(f"\n1. DATOS GENERALES:")
print(f"   Total partidos: {len(df)}")
print(f"   Columnas: {len(df.columns)}")
print(f"   Fechas: {df['Date'].min()} → {df['Date'].max()}")

# 2. Distribución de resultados
print(f"\n2. DISTRIBUCIÓN DE RESULTADOS:")
for clase in [0, 1, 2]:
    nombre = ['Local', 'Empate', 'Visitante'][clase]
    count = (df['FTR_numeric'] == clase).sum()
    pct = count / len(df) * 100
    print(f"   {nombre}: {count:4d} ({pct:5.1f}%)")

# 3. Features disponibles
print(f"\n3. FEATURES DISPONIBLES:")
features_forma = ['HT_AvgGoals', 'AT_AvgGoals', 'HT_AvgShotsTarget', 'AT_AvgShotsTarget',
                  'HT_Form_W', 'HT_Form_D', 'HT_Form_L', 'AT_Form_W', 'AT_Form_D', 'AT_Form_L']
cuotas = ['B365H', 'B365D', 'B365A', 'B365CH', 'B365CD', 'B365CA']

for feat in features_forma:
    if feat in df.columns:
        nulos = df[feat].isnull().sum()
        ceros = (df[feat] == 0).sum()
        pct_ceros = ceros / len(df) * 100
        print(f"   ✓ {feat:20s} - Nulos: {nulos:4d}, Ceros: {ceros:4d} ({pct_ceros:5.1f}%)")
    else:
        print(f"   ✗ {feat:20s} - NO EXISTE")

print(f"\n   CUOTAS:")
for cuota in cuotas:
    if cuota in df.columns:
        nulos = df[cuota].isnull().sum()
        print(f"   ✓ {cuota:10s} - Nulos: {nulos:4d}")
    else:
        print(f"   ✗ {cuota:10s} - NO EXISTE")

# 4. Temporadas
print(f"\n4. PARTIDOS POR TEMPORADA:")
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
for year, count in df.groupby('Year').size().items():
    print(f"   {year}: {count:3d} partidos")

# 5. Problema de ceros
print(f"\n5. PROBLEMA DE CEROS (features sin historial):")
for feat in features_forma:
    if feat in df.columns:
        pct_ceros = (df[feat] == 0).sum() / len(df) * 100
        if pct_ceros > 20:
            print(f"   ⚠️  {feat}: {pct_ceros:.1f}% son ceros")

# 6. Muestra de datos
print(f"\n6. MUESTRA DE LOS PRIMEROS 3 PARTIDOS:")
columnas_ver = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 
                'HT_AvgGoals', 'AT_AvgGoals', 'HT_Form_W', 'AT_Form_W']
columnas_disponibles = [c for c in columnas_ver if c in df.columns]
print(df[columnas_disponibles].head(3).to_string())

# 7. Estadísticas de features
print(f"\n7. ESTADÍSTICAS DE FEATURES CLAVE:")
for feat in ['HT_AvgGoals', 'AT_AvgGoals', 'HT_Form_W', 'AT_Form_W']:
    if feat in df.columns:
        print(f"\n   {feat}:")
        print(f"      Media: {df[feat].mean():.2f}")
        print(f"      Min: {df[feat].min():.2f}")
        print(f"      Max: {df[feat].max():.2f}")
        print(f"      Std: {df[feat].std():.2f}")

print("\n" + "="*70)
print("COPIA TODO ESTO Y PÉGALO EN EL CHAT")
print("="*70)