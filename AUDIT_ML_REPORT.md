# Auditoría ML — Predictor Premier League

**Fecha:** 2026-03-01
**Alcance:** Sistema completo (pipeline, features, modelo, value betting)
**Objetivo:** Elevar el sistema a nivel semi-profesional / research-grade

---

## RESUMEN EJECUTIVO

El sistema muestra un diseño sólido en su estructura de pipeline y una buena disciplina contra data leakage explícito. Sin embargo, presenta **dependencia crítica del mercado** (>50% de la importancia del modelo proviene de señales de cuotas), **riesgos de leakage implícito** en features de cierre, **métricas desalineadas** con el objetivo final (ROI), y un **sistema de value betting con parámetros arbitrarios** no optimizados empíricamente.

El modelo actual **NO predice fútbol — replica al mercado con ruido**. Esto no es necesariamente malo, pero debe reconocerse y abordarse de forma diferente a un modelo predictivo puro.

**Veredicto: F1=0.5102 es un resultado razonable para 3-class, pero insuficiente como evidencia de edge real contra el mercado.**

---

## 1. DIAGNÓSTICO: DEPENDENCIA DEL MERCADO (PRIORIDAD: CRÍTICA)

### 1.1 Análisis de Feature Importance

Las 6 features más importantes del modelo son **todas señales de mercado**:

| Rank | Feature | Importancia | Tipo |
|------|---------|------------|------|
| 1 | Pinnacle_Sharp_H | 0.0783 | Cuota Pinnacle |
| 2 | Pinnacle_Sharp_A | 0.0680 | Cuota Pinnacle |
| 3 | Prob_A | 0.0626 | Cuota Bet365 |
| 4 | AH_Edge_Home | 0.0485 | Asian Handicap |
| 5 | Pinnacle_Conf | 0.0269 | Cuota Pinnacle |
| 6 | Prob_Spread | 0.0231 | Cuota Bet365 |

**Las señales de mercado representan ~52% de la importancia acumulada.**
Las features puras de rendimiento (forma, xG, tabla) representan menos del 25%.

### 1.2 Diagnóstico

Esto significa que el modelo XGBoost ha aprendido **una función de aproximación a las probabilidades implícitas del mercado**, no una función independiente de predicción deportiva. El F1=0.5102 es en gran parte un reflejo de que las cuotas Pinnacle son predictivas (lo cual es trivialmente cierto — Pinnacle opera un mercado eficiente con ~2% de margen).

**Pregunta clave que falta responder: ¿El modelo supera a la propia closing line de Pinnacle?**

Si la respuesta es no, no existe edge explotable. Si la respuesta es sí, el edge probablemente viene del arbitraje de discrepancias entre mercados (Bet365 vs Pinnacle vs AH), no de información estructural deportiva.

### 1.3 Riesgo de Leakage Implícito

Las features `Pinnacle_Sharp_H/A` y `Pinnacle_Conf` se calculan a partir de **cuotas de cierre** (`PSCH`, `PSCA`). En `utils.py:agregar_features_pinnacle_move()` (línea ~670+):

- `Pinnacle_Move_H/D/A` = cierre - apertura
- `Pinnacle_Sharp_H/A` = probabilidades implícitas del **cierre**

Las cuotas de cierre incorporan **toda la información del mercado hasta minutos antes del partido**, incluyendo:
- Noticias de lesiones de última hora
- Alineaciones confirmadas
- Condiciones climáticas
- Movimiento de dinero institucional

**Usar closing odds como feature es leakage temporal indirecto**: en producción, el modelo recibe cuotas de apertura (cuando se decide apostar), pero fue entrenado con información de cierre (cuando ya no se puede apostar al mismo precio). El modelo aprende a "predecir" usando información que no estará disponible en el momento de la decisión.

Evidencia adicional en `predictor.py:474`:
```python
'Prob_Move_H': 0.0,
'Prob_Move_D': 0.0,
'Prob_Move_A': 0.0,
```
En predicción en vivo, los movimientos de mercado se fijan en 0 porque **no hay cierre disponible**. Esto confirma la inconsistencia train/serve.

### 1.4 Acciones Recomendadas

**ALTA PRIORIDAD:**
1. **Separar el análisis en dos modelos:**
   - **Modelo A (Estructural):** Solo features deportivas (forma, xG, tabla, H2H). Sin ninguna cuota.
   - **Modelo B (Híbrido):** Features deportivas + cuotas de **apertura** solamente.
   - **Nunca usar closing odds como feature.** Usarlas solo como benchmark (CLV).

2. **Medir CLV (Closing Line Value):**
   - Si modelo B predice P(H)=0.55 y apuestas a cuota de apertura 2.00, la métrica real es: ¿La closing line se movió hacia tu predicción?
   - CLV positivo consistente es la **única prueba creíble de edge**.

3. **Eliminar** `Pinnacle_Sharp_H`, `Pinnacle_Sharp_A`, `Pinnacle_Conf`, `Pinnacle_Move_*` del modelo.
   Reemplazar con `Pinnacle_Open_H/A` si se quiere usar Pinnacle como señal de apertura.

---

## 2. EVALUACIÓN DEL MODELO (PRIORIDAD: ALTA)

### 2.1 F1-Weighted como Métrica Principal — Diagnóstico

F1-weighted (0.5102) es una métrica razonable para clasificación desbalanceada, pero es **la métrica incorrecta para el objetivo real del sistema (ROI)**. Razones:

1. **F1 no penaliza la calibración**: Un modelo puede tener F1=0.50 pero estar terriblemente calibrado (P(H)=0.90 cuando la realidad es 0.55). Para value betting, la **calibración de probabilidades** es más importante que la accuracy de clasificación.

2. **F1 trata todos los errores igual**: En betting, confundir un favorito con un underdog tiene implicaciones de ROI completamente diferentes.

3. **F1 no considera las cuotas**: Un modelo puede ser peor en F1 pero mejor en ROI si acierta selectivamente en apuestas de alto valor.

### 2.2 Métricas que Faltan (IMPRESCINDIBLES)

| Métrica | Qué mide | Por qué importa |
|---------|----------|-----------------|
| **Log Loss** | Calidad probabilística | Penaliza predicciones sobreconfiadas |
| **Brier Score** | Error cuadrático de probabilidad | Estándar en meteorología y apuestas |
| **Calibration Curve** | Fiabilidad de probabilidades | ¿Cuando dices 60%, ocurre ~60%? |
| **ROI out-of-sample** | Rentabilidad real | El único resultado que importa |
| **CLV** | Ventaja sobre closing line | Prueba estándar en apuestas profesionales |
| **Yield** | ROI por unidad apostada | Comparable entre estrategias |
| **Max Drawdown** | Pérdida máxima consecutiva | Viabilidad de la estrategia |

### 2.3 Calibración

El sistema usa `CalibratedClassifierCV(method='sigmoid', cv=3)` — Platt Scaling con 3-fold CV. Problemas:

1. **3 folds CV en datos temporales**: `CalibratedClassifierCV` usa CV estándar (shuffle implícito dentro de la calibración). Esto introduce leakage temporal en la calibración. Debería ser `cv=TimeSeriesSplit(n_splits=3)` o calibración manual sobre un holdout temporal.

2. **Se guarda el modelo NO calibrado**: En `02_entrenar_modelo.py:594`, se ejecuta `guardar_modelo_final(modelo_final, ...)` donde `modelo_final` es el modelo **antes de calibrar** (línea 579-581). El `modelo_calibrado` se retorna pero no se guarda como modelo principal. Solo las métricas reportan "calibrado" pero el artefacto guardado no lo es.

   ```python
   # Línea 579-581:
   modelo_final, pred_final, mejorado = optimizar_modelo_adicional(...)
   # Línea 584:
   modelo_calibrado, probs_calibradas = calibrar_modelo(modelo_final, ...)
   # Línea 594 — GUARDA modelo_final, NO modelo_calibrado:
   guardar_modelo_final(modelo_final, features, nombre_final)
   ```

   **Esto es un bug**: si XGBoost gana, se guarda sin calibrar pero se reportan métricas como "calibrado".

### 2.4 Acciones Recomendadas

1. Implementar Log Loss y Brier Score como métricas primarias de optimización.
2. Añadir calibration curve (`sklearn.calibration.calibration_curve`) al pipeline de evaluación.
3. Corregir el bug de calibración: guardar `modelo_calibrado` si mejora el rendimiento.
4. Usar `CalibratedClassifierCV(cv=TimeSeriesSplit(n_splits=3))` en vez de cv=3 estándar.
5. Implementar ROI out-of-sample en `02_entrenar_modelo.py`.

---

## 3. VALUE BETTING — ANÁLISIS DETALLADO (PRIORIDAD: ALTA)

### 3.1 Capa 1: Shrinkage hacia Uniforme

```python
# predictor.py:459
ajustadas = 0.60 * probabilidades + 0.40 * (1/3, 1/3, 1/3)
```

**Problema:** El factor 0.60/0.40 es **completamente arbitrario**. No hay justificación teórica ni empírica para este valor.

**Análisis:**
- Con shrinkage 40%, una predicción modelo de P(H)=0.70 se convierte en P(H)=0.555.
- El edge percibido se reduce un 40%, lo cual es excesivamente conservador para un modelo ya regularizado (XGBoost con gamma=0.98, reg_alpha=0.71).
- El XGBoost ya produce probabilidades conservadoras por diseño (learning_rate=0.00566 es extremadamente bajo).

**Mejora propuesta:**
El shrinkage óptimo debería calibrarse empíricamente:
```python
# Buscar alpha óptimo vía grid search sobre validation set:
for alpha in np.arange(0.5, 1.0, 0.05):
    probs_adj = alpha * probs_model + (1 - alpha) * uniform
    brier = brier_score_loss(y_val, probs_adj)  # minimizar
```

Alternativamente, usar **Temperature Scaling** (un solo parámetro T que se optimiza por log loss):
```
probs_calibrated = softmax(logits / T)
```

### 3.2 Capa 2: Filtros de Calidad

```python
UMBRAL_EDGE_MINIMO = 0.10       # Edge mínimo 10%
CUOTA_MAXIMA = 5.0              # No underdogs extremos
PROBABILIDAD_MINIMA = 0.35      # Mínimo 35% de confianza
```

**Problemas:**
1. **Edge 10% sobre la implícita es extremadamente alto.** En mercados eficientes, edges de 2-5% ya son excepcionales. Un filtro de 10% elimina prácticamente todas las apuestas legítimas y solo deja outliers que probablemente son errores del modelo o ruido estadístico.

2. **El edge se calcula sobre probabilidades sin vig removal:**
   ```python
   # sistema_expected_value.py:53
   prob_casa = 1 / cuota  # INCLUYE el margen de la casa!
   ```
   Las probabilidades implícitas sin eliminar el overround (~5% para Bet365) inflan artificialmente el edge calculado. Un edge "real" de 5% se ve como 10% si no se normaliza.

3. **CUOTA_MAXIMA = 5.0** es razonable pero estático. Debería basarse en el Brier Score del modelo por rango de cuota (el modelo puede ser bueno para cuotas 1.5-3.0 pero terrible para 3.0-5.0).

### 3.3 Capa 3: Kelly Criterion

```python
KELLY_FRACTION = 0.25
STAKE_MAXIMO_PCT = 0.025
```

El Kelly fraccionario al 25% es conservador pero razonable. Sin embargo:

1. **La probabilidad usada en Kelly es la ajustada** (post-shrinkage), que ya fue rebajada un 40%. Esto produce un doble conservadurismo: el Kelly del 25% sobre una probabilidad que ya fue shrunk.

2. **No hay diversificación**: Si se apuesta a 5 partidos simultáneos, cada uno con Kelly independiente, la exposición total puede ser 12.5% del bankroll. Kelly asume apuestas secuenciales; para apuestas simultáneas se necesita **Kelly simultáneo** (Thorp, 2006):
   ```
   f_simultáneo = f_kelly / N_apuestas_simultáneas * factor_correlación
   ```

3. **No hay simulación Monte Carlo**: No se ha verificado si la estrategia sobrevive drawdowns realistas con el tamaño de bankroll propuesto.

### 3.4 Pregunta Fundamental

**¿El edge es contra apertura o contra cierre?**

En `sistema_expected_value.py`, el EV se calcula contra las cuotas que el usuario introduce en `jornada_config.py`. Si estas son cuotas de apertura, el edge puede ser real. Si son cuotas de cierre, no hay edge explotable (no puedes apostar al cierre después de que ocurrió).

El sistema no distingue ni documenta esto. Para value betting profesional, la **única métrica que importa es el CLV**: ¿Tus apuestas a precio de apertura cierran a menor cuota (mayor probabilidad implícita)? Si sí, tienes edge. Si no, estás apostando a ruido.

### 3.5 Acciones Recomendadas

1. **Eliminar vig antes de calcular edge**: normalizar probabilidades implícitas.
2. **Calibrar alpha empíricamente** en vez de usar 0.60 fijo.
3. **Reducir UMBRAL_EDGE_MINIMO a 3-5%** después de corregir el cálculo de vig.
4. **Implementar CLV tracker**: comparar prob_modelo vs closing_prob para cada apuesta histórica.
5. **Simular bankroll con Monte Carlo**: 10,000 simulaciones con distribución de resultados observada.
6. **Ajustar Kelly para apuestas simultáneas.**

---

## 4. INGENIERÍA DE FEATURES (PRIORIDAD: MEDIA-ALTA)

### 4.1 Colinealidad entre Señales de Mercado

Las siguientes features son altamente correlacionadas (r > 0.85):
- `Pinnacle_Sharp_H` ↔ `Prob_H` (ambas son probabilidades implícitas del mercado)
- `Pinnacle_Sharp_A` ↔ `Prob_A`
- `AH_Implied_Home` ↔ `Prob_H` (normalizada)
- `B365H` ↔ `Prob_H` (transformación lineal)

Para XGBoost, la colinealidad **no afecta la accuracy** pero sí:
- **Infla la importancia percibida** del grupo de mercado (se reparte entre features correlacionadas).
- **Reduce interpretabilidad**: no puedes distinguir qué mercado aporta información marginal.
- **Dificulta la ablation**: eliminar una feature correlacionada no reduce el rendimiento, dando la falsa impresión de que "no era importante".

### 4.2 Features que Faltan (Alto Impacto Potencial)

| Feature | Descripción | Justificación |
|---------|-------------|---------------|
| **Días desde último partido** | Fatiga / recuperación | Equipos con <3 días entre partidos rinden ~15% menos (Scoppa, 2015) |
| **Strength of Schedule** | Dificultad del fixture reciente | Un equipo con 3W-2D contra top 6 ≠ 3W-2D contra bottom 6 |
| **Decay exponencial** | Ponderar partidos recientes más | `weight = exp(-lambda * days_ago)` en vez de ventana fija de 5 |
| **Fase de temporada** | Apertura vs cierre | Equipos relegados rinden diferente en las últimas 10 jornadas |
| **Home/Away splits** | Forma separada por sede | Un equipo puede ser fuerte en casa pero débil fuera |
| **Interacciones explícitas** | xG_Diff × Position_Diff | Combinaciones no lineales que el árbol puede no descubrir |
| **Promoted/Relegated flag** | Equipo recién ascendido | Los recién ascendidos tienen patrones diferentes (menos H2H) |
| **Manager tenure** | Meses desde cambio de DT | Efecto "new manager bounce" está bien documentado |
| **Elo delta** | Cambio de Elo en últimos 30 días | Tendencia de fuerza relativa (descartaste Elo pero no probaste delta) |

### 4.3 Rolling Window Fija (window=5)

El uso de ventana fija de 5 partidos es subóptimo:

1. **5 partidos son ~3 semanas**: demasiado poca memoria para captar tendencias de largo plazo.
2. **No hay multi-escala**: sería mejor tener rolling_5, rolling_10, rolling_20 para captar tendencias cortas, medias y largas.
3. **No hay decay**: el partido de hace 5 semanas se pondera igual que el de la semana pasada.

**Mejora propuesta:**
```python
# Exponential weighted moving average (EWMA)
df['HT_xG_EWMA'] = df.groupby('HomeTeam')['Home_xG'].transform(
    lambda x: x.shift(1).ewm(halflife=5).mean()
)
```
EWMA tiene la ventaja de no tener un cutoff abrupto y pondera más los resultados recientes.

### 4.4 NaN Handling

En `02_entrenar_modelo.py:118`:
```python
X = df[features].fillna(0)
```

Rellenar con 0 es problemático para features donde 0 tiene significado semántico:
- `AH_Line = 0` significa "partido parejo", pero NaN significa "no hay datos AH".
- `Position_Diff = 0` significa "misma posición", pero NaN puede significar "inicio de temporada".

**Mejora:** Usar `fillna(median)` o `-999` para features numéricas donde 0 sea ambiguo, o usar XGBoost con `missing=np.nan` (lo soporta nativamente).

### 4.5 Config Inconsistency: FEATURES_ASIAN_HANDICAP

En `config.py:199-207`, la lista `FEATURES_ASIAN_HANDICAP` incluye:
```python
'AH_Move', 'AH_Magnitude', 'AH_Home_Favored', 'AH_Close_Match', 'AH_Big_Favorite'
```

Pero en `utils.py:agregar_features_asian_handicap()`, las features generadas son:
```python
'AH_Line', 'AH_Line_Move', 'AH_Implied_Home', 'AH_Implied_Away',
'AH_Edge_Home', 'AH_Market_Conf', 'AH_Close_Move_H', 'AH_Close_Move_A'
```

**Los nombres no coinciden.** `AH_Move`, `AH_Magnitude`, `AH_Home_Favored`, `AH_Close_Match`, `AH_Big_Favorite` no se generan en `utils.py`. El filtro `[f for f in ALL_FEATURES if f in df.columns]` (línea 97 de 02) los descarta silenciosamente, pero esto representa 5 features fantasma en la configuración.

---

## 5. VALIDACIÓN Y ROBUSTEZ (PRIORIDAD: ALTA)

### 5.1 Split Temporal — Análisis

El split 80/20 temporal (`shuffle=False`) es correcto conceptualmente, pero presenta problemas:

1. **Un solo split no es robusto**: Los últimos 20% (~739 partidos = ~2 temporadas) pueden tener distribución atípica. El rendimiento del modelo puede variar significativamente según dónde caiga el corte.

2. **No hay walk-forward en producción**: En `02_entrenar_modelo.py`, el modelo se entrena una vez y se evalúa una vez. El Optuna usa walk-forward (3 folds) para buscar hiperparámetros, pero el modelo final se entrena sobre el 80% completo y se evalúa en el 20%. No hay validación de que el modelo funciona consistentemente a través de múltiples temporadas.

3. **El Optuna optimiza sobre train set walk-forward, pero evalúa en test set**: Si el modelo fue optimizado para generalizar bien en el walk-forward del train, pero el test set tiene distribución diferente, los hiperparámetros pueden estar sobreajustados al train.

### 5.2 Validación que Falta

**Walk-Forward Real (Season-by-Season Backtest):**
```
Train: 2016-2020  →  Test: 2020-21  →  ROI?
Train: 2016-2021  →  Test: 2021-22  →  ROI?
Train: 2016-2022  →  Test: 2022-23  →  ROI?
Train: 2016-2023  →  Test: 2023-24  →  ROI?
Train: 2016-2024  →  Test: 2024-25  →  ROI?
Train: 2016-2025  →  Test: 2025-26  →  ROI? (parcial)
```

Esto revela:
- Si el modelo es consistente a través de temporadas.
- Si el edge se degrada con el tiempo (señal de sobreajuste).
- Varianza real del ROI (no solo una estimación puntual).

**Bootstrap Temporal:**
```python
# Muestrear con reemplazo bloques de partidos (por jornada)
for i in range(1000):
    sample = resample(test_predictions, replace=True)
    roi_bootstrap[i] = calcular_roi(sample)
ci_95 = np.percentile(roi_bootstrap, [2.5, 97.5])
```

### 5.3 TimeSeriesSplit en Calibración

`CalibratedClassifierCV(cv=3)` usa KFold estándar, que **mezcla datos temporales**. Un fold de calibración puede entrenar con datos de 2024 y calibrar con datos de 2020, violando la causalidad temporal. Debe usarse `cv=TimeSeriesSplit(n_splits=3)`.

### 5.4 Test Completamente Fuera de Muestra

La temporada 2025-26 está en curso. Los partidos jugados después del último re-entrenamiento son **verdadero OOS**. El sistema debería:
1. Registrar todas las predicciones antes de los partidos.
2. Comparar con resultados reales.
3. Calcular ROI real (no simulado).

Sin tracking en vivo, no hay forma de distinguir entre skill y sobreajuste al dataset histórico.

---

## 6. OPTUNA — CRÍTICA DE OPTIMIZACIÓN (PRIORIDAD: MEDIA)

### 6.1 Configuración Actual

```python
# optimizar_xgb_optuna.py
N_TRIALS = 200
WF_N_FOLDS = 3
WF_VAL_SIZE = 0.15
```

### 6.2 Problemas Detectados

1. **200 trials para 13 hiperparámetros**: El espacio de búsqueda tiene 10 parámetros de modelo + 3 pesos de clase = 13 dimensiones. Con 200 trials, Optuna explora ~15 puntos por dimensión (en promedio). Para TPE, esto es borderline suficiente. **Recomendación: 500-800 trials** con early pruning habilitado.

2. **No hay pruning**: El script no usa `optuna.pruners.MedianPruner` ni `HyperbandPruner`. Esto significa que cada trial completo se ejecuta aunque los primeros folds ya muestren rendimiento inferior al mediano. Con pruning, se pueden explorar 500+ trials en el mismo tiempo que 200 sin pruning.

3. **Métrica de optimización: F1-weighted**: Debería ser **log loss** o **Brier score** para alinear con el objetivo de calibración probabilística.

4. **No se penaliza la varianza entre folds**: La función objetivo retorna `np.average(fold_scores, weights=fold_weights)`. Un trial con scores [0.55, 0.45, 0.50] tiene el mismo valor que [0.50, 0.50, 0.50], pero el segundo es más robusto. Propuesta:
   ```python
   score = np.average(fold_scores, weights=fold_weights)
   penalty = np.std(fold_scores) * 0.5
   return score - penalty  # penalizar varianza
   ```

5. **early_stopping no se usa**: XGBoost con `n_estimators=800` y `learning_rate=0.01` es lento. Usar `early_stopping_rounds=50` con eval_set reduciría tiempo y evitaría sobreajuste:
   ```python
   m.fit(X_tr, y_tr, sample_weight=sw,
         eval_set=[(X_val, y_val)],
         early_stopping_rounds=50, verbose=False)
   ```

6. **Espacio de búsqueda:**
   - `min_child_weight: [1, 8]` es bajo. Para 3000 muestras de train, valores de 10-30 son más apropiados (el config actual usa 20, fuera del rango de Optuna).
   - `learning_rate: [0.01, 0.15]` — el modelo actual usa 0.00566 (fuera del rango de búsqueda). El Optuna actual no puede redescubrir este valor.

### 6.3 Inconsistencia de Splits

- `optimizar_xgb_optuna.py`: usa `TEST_SIZE` de config (que es 0.20, no 0.15 como dice el docstring "85/15").
- `02_entrenar_modelo.py`: también usa `TEST_SIZE = 0.20`.
- El docstring del optimizador dice "85/15" pero el código dice `test_size=TEST_SIZE` que es 0.20.

  Esto es una inconsistencia de documentación, no de código, pero causa confusión.

### 6.4 Acciones Recomendadas

1. Aumentar a 500+ trials con `MedianPruner`.
2. Optimizar **log loss** en vez de F1.
3. Implementar early stopping con eval_set.
4. Penalizar varianza entre folds.
5. Corregir rango de `min_child_weight` a [5, 30].
6. Corregir rango de `learning_rate` a [0.003, 0.15].
7. Corregir documentación del split.

---

## 7. BUGS Y PROBLEMAS DE CÓDIGO (PRIORIDAD: ALTA)

### 7.1 Bug: Modelo Guardado No Es el Calibrado

En `02_entrenar_modelo.py`:
```python
# Línea 579: obtiene modelo_final (NO calibrado)
modelo_final, pred_final, mejorado = optimizar_modelo_adicional(...)

# Línea 584: calibra
modelo_calibrado, probs_calibradas = calibrar_modelo(modelo_final, ...)

# Línea 594: ¡GUARDA EL NO CALIBRADO!
guardar_modelo_final(modelo_final, features, nombre_final)

# Línea 597-598: ¡REPORTA MÉTRICAS DEL NO CALIBRADO TAMBIÉN!
acc_final = mejor_modelo['accuracy']
f1_final = mejor_modelo['f1_score']
```

**Fix:** Decidir si se quiere calibrar o no. Si sí, guardar `modelo_calibrado` y reportar sus métricas. Si no, eliminar el paso de calibración.

### 7.2 Bug: Features Fantasma en Config

`config.py:199-207` define `FEATURES_ASIAN_HANDICAP` con nombres que no coinciden con las features generadas en `utils.py`. El código funciona porque el filtro `[f for f in ALL_FEATURES if f in df.columns]` los ignora silenciosamente, pero:
- `ALL_FEATURES` tiene más features listadas de las que realmente existen.
- Si alguien cambia las features de AH en utils.py sin actualizar config.py, puede perder features nuevas.

### 7.3 CRÍTICO: Discrepancia entre features.pkl y ALL_FEATURES en Config

**Verificación empírica de `modelos/features.pkl` revela 55 features que incluyen:**

- **6 features Pinnacle**: `Pinnacle_Move_H/D/A`, `Pinnacle_Sharp_H/A`, `Pinnacle_Conf`
- **4 features Referee**: `Ref_Away_Yellow`, `Ref_Yellow_Avg`, `Ref_Home_WinRate`, `Ref_Goals_Avg`
- **8 features Forma/Momentum**: `HT_HomeGoals5`, `AT_GoalsFor5`, `HT_GoalsFor5`, `HT_HomeWinRate5`, `Momentum_Diff`, `HT_Streak`, `HT_Pressure`, `HT_GoalsFor5`

**NINGUNA de estas está en `ALL_FEATURES` de `config.py`.**

Esto significa que el **modelo actual fue entrenado con una versión anterior del código** que incluía estas features. Si se re-ejecuta `02_entrenar_modelo.py` hoy, generará un modelo con un **set de features completamente diferente** (solo las de `ALL_FEATURES`, que excluye Pinnacle, Referee, y Forma/Momentum avanzadas).

**Consecuencias:**
1. El modelo `.pkl` y el código actual **no son reproducibles**: re-ejecutar el pipeline produce un modelo diferente.
2. Las métricas reportadas (F1=0.5102) corresponden a un modelo con Pinnacle — sin ellas, el rendimiento será significativamente peor.
3. En predicción en vivo (`predictor.py`), el modelo espera 55 features incluyendo Pinnacle, pero si no se generan esas features para la entrada, reciben 0/NaN, degradando las predicciones.

**Fix requerido:** Actualizar `ALL_FEATURES` en config.py para incluir las categorías faltantes:
```python
FEATURES_PINNACLE = [
    'Pinnacle_Move_H', 'Pinnacle_Move_D', 'Pinnacle_Move_A',
    'Pinnacle_Sharp_H', 'Pinnacle_Sharp_A', 'Pinnacle_Conf',
]
FEATURES_REFEREE = [
    'Ref_Home_WinRate', 'Ref_Goals_Avg', 'Ref_Yellow_Avg', 'Ref_Away_Yellow',
]
FEATURES_FORMA_MOMENTUM = [
    'HT_HomeWinRate5', 'HT_HomeGoals5', 'HT_GoalsFor5',
    'AT_GoalsFor5', 'HT_Streak', 'Momentum_Diff', 'HT_Pressure',
]
```
Y añadirlas a `ALL_FEATURES`.

### 7.4 AH Features en features.pkl vs Config

Las features AH en el modelo real son:
```
AHh, AHCh, AH_Line_Move, AH_Implied_Home, AH_Edge_Home, AH_Market_Conf, AH_Close_Move_H
```

Pero `FEATURES_ASIAN_HANDICAP` en config.py lista:
```
AHh, AHCh, AH_Move, AH_Magnitude, AH_Home_Favored, AH_Close_Match, AH_Big_Favorite
```

**Los nombres son completamente diferentes.** El modelo usa features que no coinciden con la configuración.

---

## 8. EXPERIMENTOS AVANZADOS (PRIORIDAD: MEDIA — Research Mode)

### 8.1 Modelo Dual (Recomendado como Siguiente Paso)

```
Modelo A: Estructural (sin cuotas)
    Features: forma, xG, tabla, H2H, fatiga, SoS
    Objetivo: predecir P(H), P(D), P(A)

Modelo B: Señal de Mercado
    Features: solo cuotas de apertura normalizadas
    Objetivo: predecir P(H), P(D), P(A)

Meta-modelo (Stacking):
    Input: probabilidades de A + probabilidades de B
    Objetivo: minimizar log loss

Edge = Meta-modelo probabilidad - Closing line probabilidad
```

Si el meta-modelo supera consistentemente al Modelo B solo, entonces el Modelo A **aporta información marginal** no capturada por el mercado. Eso sería evidencia real de edge.

### 8.2 Poisson + ML Hybrid

En vez de clasificar directamente H/D/A:
1. Predecir goles esperados del local (lambda_H) y visitante (lambda_A) con regresión.
2. Derivar P(H), P(D), P(A) via distribución Poisson bivariada.
3. Comparar con las probabilidades del mercado.

**Ventaja**: más interpretable, permite predecir mercados de goles (over/under), y produce distribuciones completas.

### 8.3 Predecir Ineficiencia del Mercado Directamente

En vez de predecir el resultado del partido:
```python
target = (closing_prob - opening_prob)  # ¿El mercado se movió?
# O:
target = (resultado_real != prediccion_opening)  # ¿El mercado se equivocó?
```

Esto convierte el problema en **detección de anomalías del mercado**, que es realmente lo que el value bettor necesita.

### 8.4 SHAP Stability Analysis

Ejecutar SHAP values por temporada y comparar:
- ¿Las features más importantes cambian entre temporadas?
- ¿Hay concept drift significativo?
- Si las importancias de mercado son estables pero las deportivas cambian, el modelo depende del mercado.

### 8.5 Train Against Closing Odds

```python
# En vez de predecir FTR (H/D/A):
target = closing_implied_probabilities  # Vector [p_h, p_d, p_a]
# Modelo de regresión multitarget
# Edge = modelo - closing (si predices mejor que el cierre a partir de la apertura)
```

---

## 9. RESUMEN DE PRIORIDADES

| # | Acción | Prioridad | Impacto | Esfuerzo |
|---|--------|-----------|---------|----------|
| 1 | Eliminar closing odds como features | **CRÍTICA** | Alto | Bajo |
| 2 | Implementar CLV tracking | **CRÍTICA** | Alto | Medio |
| 3 | Corregir bug de calibración (modelo guardado) | **ALTA** | Medio | Bajo |
| 4 | Añadir Log Loss + Brier Score como métricas | **ALTA** | Alto | Bajo |
| 5 | Walk-forward season-by-season backtest | **ALTA** | Alto | Medio |
| 6 | Normalizar probabilidades (eliminar vig) | **ALTA** | Medio | Bajo |
| 7 | Calibrar shrinkage empíricamente | **ALTA** | Medio | Medio |
| 8 | Modelo dual (con/sin cuotas) | **MEDIA-ALTA** | Alto | Alto |
| 9 | Corregir FEATURES_ASIAN_HANDICAP en config | **MEDIA** | Bajo | Bajo |
| 10 | Optuna: 500 trials + pruning + log loss | **MEDIA** | Medio | Medio |
| 11 | Añadir features dinámicas (fatiga, SoS, decay) | **MEDIA** | Medio | Alto |
| 12 | Simulación Monte Carlo de bankroll | **MEDIA** | Medio | Medio |
| 13 | Poisson hybrid model | **BAJA** | Alto | Alto |
| 14 | SHAP stability analysis | **BAJA** | Medio | Medio |

---

## 10. CONCLUSIÓN

El sistema está **bien construido como pipeline** pero **no está validado como sistema de apuestas rentable**. La dependencia del mercado enmascara la ausencia de edge real. El F1=0.5102 es un resultado decente para 3-class prediction pero no constituye evidencia de alpha.

**Antes de cualquier apuesta real, se necesita:**
1. Demostrar CLV positivo consistente.
2. Demostrar ROI positivo en walk-forward season-by-season.
3. Simular la estrategia con Monte Carlo y verificar supervivencia del bankroll.
4. Verificar que el modelo supera un baseline de "apostar siempre al favorito del mercado".

Sin estas pruebas, el sistema es un excelente proyecto educativo pero no una herramienta de apuestas confiable.

---

*Auditoría realizada con enfoque en teoría de mercados eficientes, calibración probabilística y robustez estadística.*
