# PL Predictor - Web Interface Setup

## 📁 Estructura del Proyecto

```
Predictor-Premier-League/
├── api/                          # Backend API (FastAPI)
│   ├── main.py                  # Aplicación principal
│   └── schemas.py               # Modelos Pydantic
├── frontend/                     # Frontend (React + Vite)
│   ├── src/
│   │   ├── main.jsx             # Punto de entrada React
│   │   ├── App.jsx              # Componente principal
│   │   └── App.css              # Estilos globales
│   ├── public/                  # Archivos estáticos
│   ├── index.html               # HTML principal
│   ├── package.json
│   ├── vite.config.js
│   └── Dockerfile
├── web/                         # HTML estático (legacy)
├── docker-compose.yml           # Orquestación de servicios
├── Dockerfile.api               # Imagen Docker para API
└── requeriments-api.txt        # Dependencias Python para API
```

## 🚀 Inicio Rápido

### Con Docker (Recomendado)

```bash
# Iniciar todos los servicios
docker-compose up

# El frontend estará disponible en: http://localhost:3000
# El API estará disponible en: http://localhost:8000
# Docs del API: http://localhost:8000/docs
```

### Sin Docker

#### Backend API
```bash
# Instalar dependencias
pip install -r requirements.txt
pip install -r requeriments-api.txt

# Iniciar servidor
python -m uvicorn api.main:app --reload
# API en: http://localhost:8000
```

#### Frontend
```bash
# Instalar dependencias
cd frontend
npm install

# Iniciar servidor de desarrollo
npm run dev
# Frontend en: http://localhost:3000
```

## 📦 Stack Tecnológico

### Backend
- **FastAPI**: Framework web moderno y rápido
- **Uvicorn**: Servidor ASGI
- **Pydantic**: Validación de datos

### Frontend
- **React 18**: Librería UI
- **Vite**: Build tool y dev server rápido
- **CSS**: Estilos nativos

### DevOps
- **Docker**: Containerización
- **Docker Compose**: Orquestación multi-contenedor

## 🔌 API Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Información de la API |
| GET | `/health` | Health check |
| GET | `/predictions` | Obtener todas las predicciones |
| GET | `/predictions/{match_id}` | Obtener predicción específica |

## 🛠️ Desarrollo

### Estructura Frontend
- Componentes funcionales con React Hooks
- Comunicación con API mediante `fetch`
- Proxy automático de `/api` a `http://localhost:8000`

### Integración con Modelo ML
El endpoint `/api/predictions` necesita ser integrado con el modelo existente en `core/` y `pipeline/`.

## 📝 Próximos Pasos

1. **Integración del modelo ML**: Conectar predictor existente con `/api/predictions`
2. **Autenticación**: Agregar JWT o similar si es necesario
3. **Base de datos**: Considerar agregación de persistencia
4. **Tests**: Tests unitarios y E2E
5. **Deploy**: CI/CD pipeline

## 📚 Recursos

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [React Docs](https://react.dev/)
- [Vite Docs](https://vitejs.dev/)
- [Docker Docs](https://docs.docker.com/)
