# Testing Guide - PL Predictor

## 📋 Resumen de Tests

El proyecto incluye tests automáticos para validar:
- ✅ Endpoints API funcionan correctamente
- ✅ Validación de inputs (odds, nombres de equipos, etc.)
- ✅ Integración con el modelo ML
- ✅ Manejo de errores
- ✅ Utilidades (EV, vigorish, etc.)

## 🚀 Ejecutar Tests

### Todos los tests
```bash
pytest
```

### Tests específicos
```bash
# Tests del endpoint de predicciones
pytest tests/api/test_main.py -v

# Tests de routers adicionales
pytest tests/api/test_routers.py -v

# Test específico
pytest tests/api/test_main.py::TestHealthEndpoints::test_health_check -v

# Tests con cobertura
pytest --cov=api tests/
```

### Con output detallado
```bash
pytest -v -s                    # Verbose + mostrar prints
pytest --tb=short               # Traceback corto
pytest --lf                     # Solo últimos tests fallidos
pytest --ff                     # Primero fallidos, luego otros
```

## 📊 Estructura de Tests

```
tests/
├── conftest.py              # Configuración global y fixtures
├── api/
│   ├── test_main.py        # Tests de endpoints principales
│   └── test_routers.py     # Tests de routers adicionales
└── data/                   # (Futura) Datos de test
```

## 🧪 Categorías de Tests

### Health Endpoints (`TestHealthEndpoints`)
- `test_root_endpoint` - API info
- `test_health_check` - Health status
- `test_status_endpoint` - Status detallado

### Prediction Endpoints (`TestPredictionEndpoints`)
- `test_predict_match_basic` - Predicción simple
- `test_predict_match_with_ah` - Con Asian Handicap
- `test_predict_match_detail` - Predicción detallada (con binarios)
- `test_invalid_teams` - Validación de nombres
- `test_invalid_odds` - Validación de cuotas
- `test_missing_required_fields` - Campos requeridos

### Schema Validation (`TestSchemaValidation`)
- `test_match_input_schema_valid` - Schema válido
- `test_match_input_schema_with_ah` - Con AH
- `test_match_input_schema_invalid_odds` - Validación de cuotas

### Routers (`TestGameweekRouter`, etc.)
- Gameweek predictions
- Team statistics
- Prediction history
- Utilities (EV, odds validation)

## 🔧 Configuración de Tests

El archivo `pytest.ini` define:
```ini
[pytest]
asyncio_mode = auto           # Manejo automático de async
testpaths = tests             # Directorio de tests
python_files = test_*.py      # Patrón de archivos
addopts = -v --tb=short      # Opciones por defecto
```

## 📚 Fixtures Disponibles

En `tests/conftest.py`:

```python
@pytest.fixture
def valid_match_data():
    # Datos de partido válidos

@pytest.fixture
def invalid_match_data():
    # Datos inválidos para testing de errores

@pytest.fixture
def sample_gameweek_matches():
    # 3 partidos de ejemplo para jornada
```

## ✅ Checklist de Validación

Antes de hacer deploy, verificar:

- [ ] Todos los tests pasan: `pytest`
- [ ] Sin errores de imports: `python -c "from api.main import app"`
- [ ] API inicia correctamente: `python -m uvicorn api.main:app`
- [ ] Documentación API disponible: `http://localhost:8000/docs`
- [ ] Schema OpenAPI válido: `http://localhost:8000/openapi.json`
- [ ] Health check funciona: `curl http://localhost:8000/health`

## 🐛 Debugging Tests

### Ver logs detallados
```bash
pytest -vv -s --tb=long
```

### Ejecutar un test específico con debugger
```bash
pytest tests/api/test_main.py::TestPredictionEndpoints::test_predict_match_basic -vv -s --pdb
```

### Generar reporte HTML
```bash
pytest --html=report.html --self-contained-html
```

## 📈 Cobertura de Código

```bash
# Instalar coverage
pip install coverage pytest-cov

# Ejecutar tests con cobertura
pytest --cov=api --cov-report=html

# Ver reporte
open htmlcov/index.html
```

## 🔄 CI/CD Integration

Para GitHub Actions o similar:

```yaml
- name: Run tests
  run: |
    pip install -r requeriments-api.txt
    pytest --cov=api --cov-report=xml
```

## ⚠️ Limitaciones Actuales

1. **Predictor no cargado**: Si los modelos ML no están disponibles, algunos tests retornarán 503
2. **Base de datos**: History tracking (accuracy) requiere implementar base de datos
3. **Datos históricos**: Algunos endpoints dependen de datos que deben estar en lugar

## 🎯 Próximos Tests a Implementar

- [ ] Tests de integración end-to-end
- [ ] Tests de carga (load testing)
- [ ] Tests de caché y performance
- [ ] Tests de base de datos (cuando se agregue)
- [ ] Tests de autenticación (si se agrega)

## 📖 Recursos Útiles

- [Pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/advanced/testing-dependencies/)
- [Pydantic Testing](https://docs.pydantic.dev/latest/concepts/json/)
