# Predictor de Partidos de la Premier League

Este proyecto implementa un predictor de resultados de partidos de la Premier League usando técnicas de aprendizaje automático.

## Estructura del Proyecto

```
mi_predictor_pl/
├── 01_preparar_datos.py      # Script para preparación de datos
├── 02_entrenar_modelo.py     # Script para entrenamiento del modelo
├── 03_predecir_partidos.py   # Script para realizar predicciones
├── requirements.txt          # Dependencias del proyecto
├── README.md                 # Este archivo
│
├── datos/
│   ├── temporadas/          # Archivos CSV de temporadas
│   └── procesados/          # Datos procesados para entrenamiento
│
└── modelos/                 # Modelos entrenados
```

# ⚽ Premier League Match Predictor

Sistema de predicción de resultados usando Machine Learning.

## 🚀 Instalación Rápida
```bash
# 1. Crear estructura
mkdir mi_predictor_pl
cd mi_predictor_pl
mkdir -p datos/temporadas modelos

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Colocar CSVs en datos/temporadas/

# 4. Ejecutar pipeline
python 01_preparar_datos.py
python 02_entrenar_modelo.py
python 03_predecir_partidos.py
```

## 📋 Requisitos de Datos

Los CSVs deben tener estas columnas:
- `Date`, `HomeTeam`, `AwayTeam`
- `FTHG`, `FTAG`, `FTR` (goles y resultado)
- `HS`, `AS`, `HST`, `AST` (tiros)
- `B365H`, `B365D`, `B365A` (cuotas - opcional)

## 🎯 Uso

### Predecir un partido:
```bash
python 03_predecir_partidos.py
# Luego sigue las instrucciones en pantalla
```

### Ejemplo de salida:
```
⚽ Man City vs Arsenal
═══════════════════════════════════════

📊 PROBABILIDADES:
   🏠 Man City: 52.3%
   🤝 Empate: 24.1%
   ✈️  Arsenal: 23.6%

🎯 PREDICCIÓN: Victoria Man City
   Confianza: 52.3%

📈 FORMA RECIENTE:
   Man City: 4W-1D-0L
   Arsenal: 3W-1D-1L
```

## ⚠️ Importante

- Este proyecto es **educativo**
- NO garantiza ganancias en apuestas
- La precisión típica es 50-55% (mejor que azar)