# Setup en Windows - Guía Rápida

## 🐳 Paso 1: Instalar Docker Desktop

1. **Descargar** desde: https://www.docker.com/products/docker-desktop
2. **Instalar** y reiniciar Windows
3. **Abrir Docker Desktop** (busca en Inicio)
4. **Esperar 1-2 minutos** a que inicie completamente

## ✅ Verificar que Docker funciona

Abre **PowerShell** (como usuario normal) y ejecuta:

```powershell
docker --version
docker ps
```

Si ves un error como `The system cannot find the file specified`:
- ❌ Docker no está corriendo
- ✅ Abre Docker Desktop nuevamente
- ⏳ Espera 60 segundos
- 🔄 Intenta de nuevo

## 🚀 Opción 1: Ejecutar con Docker (Recomendado)

```powershell
# Navega a la carpeta del proyecto
cd C:\Users\LAP\Desktop\mi_predictor_pl

# Inicia los servicios
docker-compose up

# En otra terminal PowerShell, verifica que funciona:
curl http://localhost:8000/health
curl http://localhost:3000
```

**URLs:**
- 🌐 Página: http://localhost:3000
- 🔌 API: http://localhost:8000
- 📚 Documentación: http://localhost:8000/docs

**Para detener:**
```powershell
# Presiona Ctrl+C en la terminal donde corre docker-compose
# O en otra terminal:
docker-compose down
```

---

## 🚀 Opción 2: Ejecutar Manual (Sin Docker)

Si Docker no funciona en tu Windows, puedes ejecutar todo localmente:

### Backend (API)

```powershell
# Instalar dependencias Python
pip install -r requirements.txt
pip install -r requeriments-api.txt

# Iniciar API
python -m uvicorn api.main:app --reload
# Verás: "Uvicorn running on http://127.0.0.1:8000"
```

### Frontend (En otra terminal PowerShell)

```powershell
# Instalar dependencias Node.js
cd frontend
npm install

# Iniciar desarrollo
npm run dev
# Verás: "http://localhost:5173"
```

---

## 🐛 Troubleshooting

### ❌ "Docker daemon is not running"
```powershell
# Abre Docker Desktop nuevamente
# Espera 60 segundos
# Intenta de nuevo
```

### ❌ "Port 8000 already in use"
```powershell
# Encuentra el proceso usando el puerto
netstat -ano | findstr :8000

# Mata el proceso (reemplaza PID)
taskkill /PID <PID> /F
```

### ❌ "Port 3000 already in use"
```powershell
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

### ❌ "ModuleNotFoundError: No module named 'api'"
```powershell
# Asegúrate de estar en la carpeta raíz del proyecto
cd C:\Users\LAP\Desktop\mi_predictor_pl
python -m uvicorn api.main:app --reload
```

---

## 📝 Checklist Rápido

- [ ] Docker Desktop instalado y corriendo
- [ ] `docker --version` funciona
- [ ] Carpeta del proyecto abierta en PowerShell
- [ ] `docker-compose up` sin errores
- [ ] http://localhost:8000/docs carga
- [ ] http://localhost:3000 carga

---

## 💡 Comandos Útiles

```powershell
# Ver logs del API
docker-compose logs -f api

# Ver logs del Frontend
docker-compose logs -f frontend

# Ejecutar tests
docker-compose exec api pytest

# Limpiar todo (cuidado: elimina contenedores)
docker-compose down

# Reconstruir imágenes
docker-compose build --no-cache
```

---

## 🆘 ¿Aún no funciona?

Prueba estos pasos en orden:

1. **Reinicia Docker Desktop**
   - Ciérralo completamente
   - Vuelve a abrirlo
   - Espera 2 minutos

2. **Limpia Docker**
   ```powershell
   docker system prune -a
   ```

3. **Ejecuta modo manual** (sin Docker)
   - Backend: `python -m uvicorn api.main:app`
   - Frontend: `cd frontend && npm run dev`

4. **Contacta al desarrollador**
   - Comparte el error exacto
   - Incluye la salida de `docker --version`
   - Incluye la salida de `python --version`
