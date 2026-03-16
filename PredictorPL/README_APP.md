# 🎯 PredictorPL - Aplicaciones Gradio

Este directorio contiene las interfaces Gradio para el sistema de predicción de Premier League.

## 📋 Versiones Disponibles

### 1. **app_demo.py** ⭐ RECOMENDADO PARA HUGGING FACE
Versión de demostración con predicciones de ejemplo pre-configuradas.

**Ventajas:**
- ✅ No requiere entrenamiento de modelos
- ✅ Interfaz funcional completa
- ✅ Muestra todas las funcionalidades de la UI
- ✅ Ideal para Hugging Face Spaces (sin problemas de dependencias)
- ✅ Predicciones realistas basadas en datos históricos

**Cómo ejecutar:**
```bash
cd PredictorPL
python3 app_demo.py
# Accede a: http://localhost:7860
```

**Predicciones incluidas:**
- Arsenal vs Chelsea
- Manchester City vs Liverpool
- Manchester United vs Tottenham
- Brighton vs Newcastle

---

### 2. **app_simplificado.py**
Versión que intenta usar los modelos reales con partidos de `jornada_config.py`.

**Ventajas:**
- Usa datos reales del predictor
- Predicciones en tiempo real
- Base para versión completa

**Limitaciones:**
- ⚠️ Requiere que los modelos estén correctamente entrenados
- ⚠️ Actualmente hay un problema de feature mismatch (en investigación)

**Cómo ejecutar:**
```bash
cd ..  # Vuelve al directorio raíz
python3 -m pip install -r requirements.txt
cd PredictorPL
python3 app_simplificado.py
```

---

### 3. **app.py**
Versión completa con entrada manual de equipos y cuotas.

**Estado:** En desarrollo (espera solución del feature mismatch)

---

## 🐛 Problemas Conocidos

### Feature Mismatch en Modelos Reales
El modelo entrenado espera 35 features específicos, pero la función `_obtener_stats_equipo()` está generando un conjunto diferente.

**Indicadores:**
```
ValueError: The feature names should match those that were passed during fit.
```

**Soluciones en progreso:**
- Investigar si es un problema de incompatibilidad de versiones
- Verificar si los datos históricos están completos
- Considerar usar un modelo alternativo desde `/modelos/archive/`

---

## 🚀 Desplegar en Hugging Face Spaces

### Opción A: Usar app_demo.py (RECOMENDADO)
1. Crea un nuevo Space en [huggingface.co/spaces](https://huggingface.co/spaces)
2. Selecciona **Gradio** como SDK
3. Sube los archivos o conecta el repositorio GitHub
4. Apunta a `PredictorPL/app_demo.py` como entrada
5. Actualiza `requirements.txt` con la dependencia de Gradio

### Opción B: Resolver Feature Mismatch
Cuando se resuelva el problema de features, usa `app.py` o `app_simplificado.py` para predicciones en tiempo real.

---

## 📦 Dependencias

Mínimas (para app_demo.py):
```
gradio>=4.0.0
```

Completas (para modelos reales):
```
gradio>=4.0.0
pandas>=1.3.0,<3.0.0
numpy>=1.21.0,<2.0.0
scikit-learn>=1.0.0,<2.0.0
xgboost>=1.6.0,<3.0.0
lightgbm>=3.3.0,<5.0.0
joblib>=1.1.0,<2.0.0
```

---

## 🎨 Características de la UI

Todas las versiones incluyen:
- 📊 Predicción del resultado (Local, Empate, Visitante)
- 💪 Confianza del modelo
- 📈 Probabilidades detalladas
- 💰 Análisis de valor (Value Betting)
- 📝 Forma reciente de equipos
- ⚡ Stats avanzados (xG, Elo, etc.)
- 📚 Información sobre el modelo
- ⚠️ Disclaimers responsables

---

## 🔄 Roadmap

- [ ] Resolver problema de feature mismatch
- [ ] Agregar predicciones de mercados binarios (O/U goles)
- [ ] Integración en tiempo real con datos de jornada actual
- [ ] Comparación de modelos
- [ ] Histórico de predicciones
- [ ] Análisis de performance
- [ ] Dark mode
- [ ] Multiidioma

---

## 📞 Soporte

Para problemas:
1. Revisa los logs en la consola
2. Verifica que todas las dependencias estén instaladas
3. Comprueba que los archivos de modelo existan en `/modelos/`

---

**Última actualización:** Marzo 2026
**Versión:** 1.0
