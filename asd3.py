import pandas as pd

# Cargar datos integrados
df = pd.read_csv('./datos/procesados/premier_league_con_xg_integrado.csv')

print(f"📊 ANTES de rellenar:")
print(f"   Home_xG nulos: {df['Home_xG'].isna().sum()}")
print(f"   Away_xG nulos: {df['Away_xG'].isna().sum()}")
print(f"   Home_xGA nulos: {df['Home_xGA'].isna().sum()}")
print(f"   Away_xGA nulos: {df['Away_xGA'].isna().sum()}")

# ============================================================================
# RELLENAR VALORES FALTANTES CON LA LÓGICA CORRECTA
# ============================================================================

# Rellena Home_Poss si falta y Away_Poss está presente
df.loc[df['Home_Poss'].isnull() & df['Away_Poss'].notnull(), 'Home_Poss'] = \
    100 - df.loc[df['Home_Poss'].isnull() & df['Away_Poss'].notnull(), 'Away_Poss']

# Rellena Away_Poss si falta y Home_Poss está presente
df.loc[df['Away_Poss'].isnull() & df['Home_Poss'].notnull(), 'Away_Poss'] = \
    100 - df.loc[df['Away_Poss'].isnull() & df['Home_Poss'].notnull(), 'Home_Poss']

# Si Home_xG está vacío pero Away_xGA tiene valor → copiar
df.loc[df['Home_xG'].isna() & df['Away_xGA'].notna(), 'Home_xG'] = \
    df.loc[df['Home_xG'].isna() & df['Away_xGA'].notna(), 'Away_xGA']

# Si Away_xG está vacío pero Home_xGA tiene valor → copiar
df.loc[df['Away_xG'].isna() & df['Home_xGA'].notna(), 'Away_xG'] = \
    df.loc[df['Away_xG'].isna() & df['Home_xGA'].notna(), 'Home_xGA']

# Si Home_xGA está vacío pero Away_xG tiene valor → copiar
df.loc[df['Home_xGA'].isna() & df['Away_xG'].notna(), 'Home_xGA'] = \
    df.loc[df['Home_xGA'].isna() & df['Away_xG'].notna(), 'Away_xG']

# Si Away_xGA está vacío pero Home_xG tiene valor → copiar
df.loc[df['Away_xGA'].isna() & df['Home_xG'].notna(), 'Away_xGA'] = \
    df.loc[df['Away_xGA'].isna() & df['Home_xG'].notna(), 'Home_xG']

print(f"\n📊 DESPUÉS de rellenar:")
print(f"   Home_xG nulos: {df['Home_xG'].isna().sum()}")
print(f"   Away_xG nulos: {df['Away_xG'].isna().sum()}")
print(f"   Home_xGA nulos: {df['Home_xGA'].isna().sum()}")
print(f"   Away_xGA nulos: {df['Away_xGA'].isna().sum()}")

# Guardar
df.to_csv('./datos/procesados/premier_league_con_xg_integrado.csv', index=False)

print(f"\n✅ Guardado con valores rellenados")