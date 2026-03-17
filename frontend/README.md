# 🎯 Premier League Predictor - Frontend

Frontend React/Vite profesional, escalable y bien estructurado para predicciones de la Premier League.

## 📋 Estructura del Proyecto

```
frontend/
├── src/
│   ├── components/                    # Componentes reutilizables
│   │   ├── Header/                   # Encabezado principal
│   │   │   ├── Header.jsx
│   │   │   └── Header.css
│   │   ├── PredictionForm/           # Formulario de predicción
│   │   │   ├── PredictionForm.jsx
│   │   │   └── PredictionForm.css
│   │   ├── ResultDisplay/            # Muestra resultados
│   │   │   ├── ResultDisplay.jsx
│   │   │   └── ResultDisplay.css
│   │   ├── Loading/                  # Spinner de carga
│   │   │   ├── Loading.jsx
│   │   │   └── Loading.css
│   │   └── ErrorMessage/             # Mensajes de error
│   │       ├── ErrorMessage.jsx
│   │       └── ErrorMessage.css
│   │
│   ├── pages/                        # Páginas principales
│   │   ├── Home.jsx                 # Página de inicio
│   │   └── Home.css
│   │
│   ├── hooks/                        # Custom hooks reutilizables
│   │   ├── useAPI.js                # Hook para llamadas API
│   │   └── usePrediction.js         # Hook especializado para predicciones
│   │
│   ├── services/                     # Servicios (API, etc)
│   │   └── api.js                   # Configuración y endpoints
│   │
│   ├── App.jsx                       # Componente raíz
│   ├── App.css                       # Estilos globales
│   ├── main.jsx                      # Punto de entrada React
│   └── index.css                     # Reset CSS
│
├── public/                           # Assets estáticos
├── index.html                        # HTML raíz
├── package.json                      # Dependencias
├── vite.config.js                    # Configuración Vite
├── .env.production                   # Variables producción
└── .gitignore

```

## 🚀 Instalación y Desarrollo

### 1. Instalar dependencias
```bash
npm install
```

### 2. Variables de entorno
```bash
# Desarrollo (por defecto apunta a localhost:8000)
VITE_API_URL=http://localhost:8000

# Producción
VITE_API_URL=https://tu-api.run.app
```

### 3. Ejecutar en desarrollo
```bash
npm run dev
```
Abre http://localhost:5173

### 4. Build para producción
```bash
npm run build
```

## 🏗️ Arquitectura

### Componentes
- **Header**: Muestra estado de la API, marca e información
- **PredictionForm**: Formulario para seleccionar equipos y cuotas
- **ResultDisplay**: Muestra predicciones, probabilidades y análisis
- **Loading**: Spinner de carga
- **ErrorMessage**: Mensaje de error con opción de cerrar

### Hooks
- **useAPI**: Hook genérico para cualquier llamada API
- **usePrediction**: Hook especializado para predicciones

### Servicios
- **api.js**: Configuración de axios y endpoints disponibles

## 🎨 Estilo y Temas

- **Tema oscuro** por defecto (dark mode)
- **Variables CSS** centralizadas en App.css
- **Responsive design** para móvil, tablet y escritorio
- **Animaciones suaves** y transiciones

## 📱 Componentes Reutilizables

### Usar ErrorMessage
```jsx
import ErrorMessage from '../components/ErrorMessage/ErrorMessage'

<ErrorMessage
  message="Error al cargar"
  onDismiss={() => setError(null)}
/>
```

### Usar Loading
```jsx
import Loading from '../components/Loading/Loading'

<Loading message="Analizando..." />
```

### Usar hooks
```jsx
import { usePrediction } from '../hooks/usePrediction'

const { prediction, loading, error, predict, reset } = usePrediction()

// Hacer predicción
await predict({
  local: "Arsenal",
  visitante: "Chelsea",
  cuota_h: 2.1,
  cuota_d: 3.4,
  cuota_a: 3.2
})
```

## 🌐 API Endpoints

El frontend se conecta a estos endpoints:

- `GET /health` - Verificar estado de la API
- `GET /status` - Obtener estado detallado
- `POST /predictions` - Predicción simple
- `POST /predictions/detail` - Predicción detallada
- `GET /gameweek` - Jornada actual

## 📦 Dependencias

- **React**: Framework UI
- **Vite**: Build tool moderno
- **Axios**: Cliente HTTP

## 🚀 Deploy

### Netlify
```bash
# Desde CLI
npm run build
netlify deploy --prod --dir=dist

# O conectar repositorio y deploy automático
# Comando build: npm install && npm run build
# Directorio: dist
```

### Vercel
Similar a Netlify, autodetecta configuración Vite.

## 💡 Mejores Prácticas

1. ✅ **Componentes pequeños y reutilizables**
2. ✅ **Hooks para lógica compartida**
3. ✅ **CSS modular por componente**
4. ✅ **Servicios centralizados**
5. ✅ **Manejo de errores robusto**
6. ✅ **Responsive design**

## 🔧 Troubleshooting

### API no conecta
- Verificar que backend está corriendo en http://localhost:8000
- Revisar CORS en backend
- Revisar console del navegador

### Error de build
```bash
npm ci  # Limpiar y reinstalar
npm run build
```

### Port 5173 en uso
```bash
npm run dev -- --port 3000
```

## 📝 Licencia

MIT

---

**¿Necesitas ayuda?** Revisa los componentes en `src/components/` para ejemplos de implementación.
