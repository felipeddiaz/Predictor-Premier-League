import pandas as pd

# Simular el proceso
df_raw = pd.read_csv('./datos/raw/final_matches_xg.csv')

MAPEO = {
    'Manchester United': 'Man United',
    'Nottingham Forest': "Nott'm Forest",
}

# Aplicar mapeo
df_raw['team'] = df_raw['team'].replace(MAPEO)

# Ver qué quedó
print("Equipos en CSV después del mapeo:")
print(df_raw['team'].unique())

# Buscar Man United
man_utd = df_raw[df_raw['team'] == 'Man United']
print(f"\nPartidos de Man United después del mapeo: {len(man_utd)}")

# Ver si Manchester United sigue existiendo
man_utd_original = df_raw[df_raw['team'] == 'Manchester United']
print(f"Partidos de 'Manchester United' (original): {len(man_utd_original)}")